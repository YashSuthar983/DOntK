"""ApexNet: Dual-Scale Dead-Reckoned EgoMap + Neural Map + VIN Planner + Hierarchical GRU Policy.

All actions flow strictly from trained neural parameters. No hand-coded path search or aimbots.
Designed for strong zero-shot generalization to unseen maps:
- Texture-agnostic geometric EgoMap (fine 32x32 + wide 16x16) + 8-dim learned Neural Map writes
- K=6 shared-weight Value-Iteration-Network (VIN) conv planner over the 16x16 map
- Circular 1D ray CNN + 2-layer Entity Transformer + Vitals/Depth MLP -> GRU(256)
- Hierarchical 17-class waypoint direction head feeding the 6 factored action heads
- Legal loaded-weapon switch mask (never switches away from a weapon with ammo)
- Training-only asymmetric privileged critic + DFP/goal auxiliary heads
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from egomap import FINE_CROP, GRID_SIZE, N_FIXED_CH, VEC_DIM

torch.backends.mha.set_fastpath_enabled(False)

N_LEARNED_CH = 8
HEADS = (3, 3, 3, 2, 2, 2)  # Move(N/F/B), Strafe(N/L/R), Turn(N/L/R), Attack, Use, NextWpn
N_DIR_BINS = 17             # 16 ego-centric 22.5-deg bins + 'none'
N_GOAL_KINDS = 5            # none, exit, key, remote_switch, frontier
PRIV_DIM = 24               # matches FastDoomPackEnv priv_vec
MAP_CELLS = GRID_SIZE * GRID_SIZE + 1


def apex_factored_to_multibinary(factored: torch.Tensor) -> torch.Tensor:
    """Convert factored (..., 6) indices [Move(0=N,1=F,2=B), Strafe(0=N,1=L,2=R), Turn(0=N,1=L,2=R), Atk, Use, Wpn]
    to MultiBinary(9) int8 matching FrontierOracle / AEGIS3D convention."""
    out = torch.zeros((*factored.shape[:-1], 9), dtype=torch.int8, device=factored.device)
    out[..., 0] = (factored[..., 0] == 1).to(torch.int8)  # MOVE_FORWARD
    out[..., 1] = (factored[..., 0] == 2).to(torch.int8)  # MOVE_BACKWARD
    out[..., 2] = (factored[..., 1] == 1).to(torch.int8)  # MOVE_LEFT
    out[..., 3] = (factored[..., 1] == 2).to(torch.int8)  # MOVE_RIGHT
    out[..., 4] = (factored[..., 2] == 1).to(torch.int8)  # TURN_LEFT
    out[..., 5] = (factored[..., 2] == 2).to(torch.int8)  # TURN_RIGHT
    out[..., 6] = (factored[..., 3] == 1).to(torch.int8)  # ATTACK
    out[..., 7] = (factored[..., 4] == 1).to(torch.int8)  # USE
    out[..., 8] = (factored[..., 5] == 1).to(torch.int8)  # SELECT_NEXT_WEAPON
    return out


class ApexNet(nn.Module):
    def __init__(self, hid: int = 256, vin_k: int = 6, include_training_heads: bool = False) -> None:
        super().__init__()
        self.hid = hid
        self.vin_k = vin_k
        self.include_training_heads = include_training_heads
        self.detach_value = False

        # 1. Circular 1D Ray CNN: (B, 8, 64) -> 256
        self.ray_conv = nn.Sequential(
            nn.Conv1d(8, 32, 5, padding=2, padding_mode="circular"),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 48, 5, stride=2, padding=2, padding_mode="circular"),
            nn.ReLU(inplace=True),
            nn.Conv1d(48, 64, 5, stride=2, padding=2, padding_mode="circular"),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * 16, 256),
            nn.ReLU(inplace=True),
        )

        # 2. Vitals + PrevAction + Depth Profile MLP -> 128
        self.vec_mlp = nn.Sequential(
            nn.Linear(VEC_DIM, 128),
            nn.ReLU(inplace=True),
        )

        # 3. Entity Transformer (+ null token, masked mean + max pooling) -> 128
        self.etype_emb = nn.Embedding(12, 16)
        self.eproj = nn.Linear(6 + 16, 64)
        self.null_tok = nn.Parameter(torch.zeros(1, 1, 64))
        self.ent_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(64, 4, 128, dropout=0.0, batch_first=True),
            num_layers=2,
            enable_nested_tensor=False,
        )

        # 4. Fine 32x32 EgoMap (+ 8 learned Neural Map channels) + K-step VIN Planner
        fine_in = N_FIXED_CH + N_LEARNED_CH
        self.map_in = nn.Sequential(
            nn.Conv2d(fine_in, 32, 3, stride=2, padding=1),  # 32x32 -> 16x16
            nn.ReLU(inplace=True),
        )
        self.vin_conv = nn.Conv2d(64, 32, 3, padding=1)  # shared across vin_k iterations
        self.map_out = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=2, padding=1),       # 16x16 -> 8x8
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 64, 256),
            nn.ReLU(inplace=True),
        )
        # Local 8x8 planner readout centered on the agent in the 16x16 VIN grid
        self.plan_out = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, 64),
            nn.ReLU(inplace=True),
        )

        # 5. Wide 16x16 Zoomed-Out EgoMap (4096x4096u) -> 64
        self.wide_conv = nn.Sequential(
            nn.Conv2d(N_FIXED_CH, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 8x8
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),  # 4x4
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 16, 64),
            nn.ReLU(inplace=True),
        )

        # 6. Fusion + GRU + Neural Map Write Gate
        fuse_in = 256 + 128 + 128 + 256 + 64
        self.fuse = nn.Sequential(nn.Linear(fuse_in, hid), nn.ReLU(inplace=True))
        self.gru = nn.GRUCell(hid, hid)
        self.map_write = nn.Linear(hid, N_LEARNED_CH + 1)  # gate + 8-dim feature

        # 7. Hierarchical Direction Head + Factored Policy Head
        self.dir_head = nn.Sequential(
            nn.Linear(hid + 64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, N_DIR_BINS),
        )
        self.pi_head = nn.Sequential(
            nn.Linear(hid * 2 + 64 + N_DIR_BINS, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, sum(HEADS)),
        )
        nn.init.orthogonal_(self.pi_head[-1].weight, gain=0.01)
        nn.init.zeros_(self.pi_head[-1].bias)

        # 8. Training-only Asymmetric Critic + Auxiliary Heads
        if include_training_heads:
            self.v_priv = nn.Sequential(nn.Linear(PRIV_DIM, 64), nn.ReLU(inplace=True))
            self.v_head = nn.Sequential(
                nn.Linear(hid * 2 + 64, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 1),
            )
            nn.init.orthogonal_(self.v_head[-1].weight, gain=1.0)
            nn.init.zeros_(self.v_head[-1].bias)
            self.aux_goal = nn.Linear(hid, N_GOAL_KINDS)
            self.aux_dist = nn.Linear(hid, 1)

    def init_state(self, batch_size: int, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch_size, self.hid, dtype=torch.float32, device=device)
        L = torch.zeros(batch_size, N_LEARNED_CH, MAP_CELLS, dtype=torch.float32, device=device)
        return h, L

    def forward_step(
        self,
        obs: dict[str, torch.Tensor],
        state: tuple[torch.Tensor, torch.Tensor],
        priv_vec: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], tuple[torch.Tensor, torch.Tensor], dict[str, torch.Tensor]]:
        h, L = state
        B = h.shape[0]

        # Gather learned Neural Map channels into heading-aligned 32x32 fine crop
        crop_idx = obs["crop_idx"].long()  # (B, 1024)
        learned = torch.gather(L, 2, crop_idx.unsqueeze(1).expand(B, N_LEARNED_CH, -1))
        fine_in = torch.cat(
            [obs["fine_map"].float(), learned.view(B, N_LEARNED_CH, FINE_CROP, FINE_CROP)],
            dim=1,
        )
        x = self.map_in(fine_in)  # (B, 32, 16, 16)
        r = x
        for _ in range(self.vin_k):
            r = F.relu(self.vin_conv(torch.cat([x, r], dim=1)))
        z_map = self.map_out(r)
        z_plan = self.plan_out(r[:, :, 4:12, 4:12])  # 8x8 centered on agent

        z_wide = self.wide_conv(obs["wide_map"].float())
        z_ray = self.ray_conv(obs["rays"].float())
        z_vec = self.vec_mlp(obs["vec"].float())

        # Entity attention
        e = self.eproj(torch.cat([obs["ents"].float(), self.etype_emb(obs["etype"].long())], dim=-1))
        e = torch.cat([self.null_tok.expand(B, 1, 64), e], dim=1)
        pad = torch.cat(
            [torch.zeros(B, 1, dtype=torch.bool, device=e.device), obs["ents"][..., 5] < 0.5],
            dim=1,
        )
        e = self.ent_attn(e, src_key_padding_mask=pad)
        valid = (~pad).float().unsqueeze(-1)
        z_ent_mean = (e * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        z_ent_max = e.masked_fill(pad.unsqueeze(-1), 0.0).max(dim=1).values
        z_ent = torch.cat([z_ent_mean, z_ent_max], dim=-1)

        z = self.fuse(torch.cat([z_ray, z_vec, z_ent, z_map, z_wide], dim=-1))
        h_new = self.gru(z, h)

        # Differentiable gated Neural Map write at current dead-reckoned cell
        w = self.map_write(h_new)
        gate = torch.sigmoid(w[:, :1])
        pos = obs["pos_idx"].long().view(B, 1, 1).expand(B, N_LEARNED_CH, 1)
        old_val = torch.gather(L, 2, pos).squeeze(-1)
        new_val = (1.0 - gate) * old_val + gate * torch.tanh(w[:, 1:])
        L_new = L.scatter(2, pos, new_val.unsqueeze(-1))

        # Hierarchical direction + factored policy logits
        dir_logits = self.dir_head(torch.cat([h_new, z_plan], dim=-1))
        feat = torch.cat([h_new, z], dim=-1)
        pi_in = torch.cat([feat, z_plan, torch.softmax(dir_logits, dim=-1).detach()], dim=-1)
        raw_logits = self.pi_head(pi_in)

        # Legal weapon-switch mask: suppress NextWeapon=1 when current weapon has ammo
        if "wmask" in obs:
            pen = (1.0 - obs["wmask"].float().view(B, 1)) * 1e4
            raw_logits = torch.cat([raw_logits[:, :-1], raw_logits[:, -1:] - pen], dim=1)

        logits = list(torch.split(raw_logits, HEADS, dim=-1))
        extras: dict[str, torch.Tensor] = {"dir_logits": dir_logits}

        if self.include_training_heads:
            vfeat = feat.detach() if self.detach_value else feat
            if priv_vec is not None:
                vp = self.v_priv(priv_vec.float())
            else:
                vp = h_new.new_zeros(B, 64)
            extras["value"] = self.v_head(torch.cat([vfeat, vp], dim=-1)).squeeze(-1)
            extras["goal_logits"] = self.aux_goal(h_new)
            extras["dist_pred"] = self.aux_dist(h_new).squeeze(-1)

        return logits, (h_new, L_new), extras

    def strip_for_submission(self) -> dict[str, torch.Tensor]:
        """Return CPU state_dict excluding training-only critic/aux heads."""
        drop = ("v_priv.", "v_head.", "aux_goal.", "aux_dist.")
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if not k.startswith(drop)}
