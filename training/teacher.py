"""Phase 2 Fast Omniscient Privileged Teacher Policy (Training Only — Never Shipped).

Runs on `render_buffers=False` (10,000+ steps/s on CPU) using legal structured sensors
PLUS the 24-dim privileged geodesic & map oracle vector from `FastDoomPackEnv`.
Outputs the exact same 6 factored conflict-free action heads as AEGIS-3D for clean
per-head KL distillation in Phase 3 & Phase 4.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_SUB_DIR = str(Path(__file__).resolve().parent.parent / "submission")
if _SUB_DIR not in sys.path:
    sys.path.insert(0, _SUB_DIR)

from aegis3d import ACTION_DIMS, EntitySetNet


class OmniscientTeacher(nn.Module):
    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hit_emb = nn.Embedding(6, 6)
        self.ray_conv = nn.Sequential(
            nn.Conv1d(7, 32, kernel_size=5, padding=2, padding_mode="circular"),
            nn.SiLU(inplace=True),
            nn.Conv1d(32, 48, kernel_size=5, stride=2, padding=2, padding_mode="circular"),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
            nn.Linear(48 * 8, 128),
            nn.LayerNorm(128),
            nn.SiLU(inplace=True),
        )
        self.entity_net = EntitySetNet()
        self.priv_vitals = nn.Sequential(
            nn.Linear(17 + 24 + 9, 128),
            nn.LayerNorm(128),
            nn.SiLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Linear(128 + 96 + 128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.action_heads = nn.ModuleList([nn.Linear(hidden_dim, d) for d in ACTION_DIMS])
        self.value_head = nn.Linear(hidden_dim, 1)

    def init_hidden(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        return torch.zeros((batch_size, self.hidden_dim), device=device)

    def encode(
        self,
        obs: dict[str, torch.Tensor],
        priv_vec: torch.Tensor,
        prev_action: torch.Tensor,
    ) -> torch.Tensor:
        ray_d = obs["ray_distance"].to(torch.float32).unsqueeze(1)  # (B, 1, 64)
        ray_h = self.hit_emb(obs["ray_hit_type"].to(torch.int64).clamp(0, 5)).transpose(1, 2)  # (B, 6, 64)
        ray_feat = self.ray_conv(torch.cat([ray_d, ray_h], dim=1))

        vitals = obs["vitals"].to(torch.float32)
        combat_ctx, _ = self.entity_net(
            obs["ent_features"].to(torch.float32),
            obs["ent_type"].to(torch.int64),
            obs["ent_mask"].to(torch.float32),
            vitals[:, 2],
        )
        v_norm = torch.cat(
            [
                vitals[:, 0:2] / 100.0,
                vitals[:, 2:3] / 7.0,
                vitals[:, 3:8] / 100.0,
                vitals[:, 8:14],
                vitals[:, 14:17] / 20.0,
                priv_vec.to(torch.float32),
                prev_action.to(torch.float32),
            ],
            dim=-1,
        )
        pv_feat = self.priv_vitals(v_norm)
        return self.fusion(torch.cat([ray_feat, combat_ctx, pv_feat], dim=-1))

    def head_from_h(self, h: torch.Tensor, prev_action: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        logits = [head(h) for head in self.action_heads]
        prev_use = prev_action[..., 7] > 0.5
        if prev_use.any():
            m = torch.zeros_like(logits[4])
            m[..., 1] = torch.where(prev_use, torch.full_like(m[..., 1], -1e9), torch.zeros_like(m[..., 1]))
            logits[4] = logits[4] + m
        value = self.value_head(h).squeeze(-1)
        return logits, value

    def forward_step(
        self,
        obs: dict[str, torch.Tensor],
        priv_vec: torch.Tensor,
        prev_action: torch.Tensor,
        h: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        x = self.encode(obs, priv_vec, prev_action)
        new_h = self.gru(x, h)
        logits, value = self.head_from_h(new_h, prev_action)
        return logits, value, new_h

    def forward_sequence(
        self,
        obs_seq: dict[str, torch.Tensor],
        priv_seq: torch.Tensor,
        prev_act_seq: torch.Tensor,
        done_seq: torch.Tensor,
        init_h: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Batched encoder across (T*N) + fast GRU rollout across T."""
        t_len, n_env = priv_seq.shape[0], priv_seq.shape[1]
        flat_obs = {k: v.reshape(t_len * n_env, *v.shape[2:]) for k, v in obs_seq.items()}
        x_seq = self.encode(
            flat_obs,
            priv_seq.reshape(t_len * n_env, -1),
            prev_act_seq.reshape(t_len * n_env, -1),
        ).view(t_len, n_env, self.hidden_dim)

        h_list = []
        cur_h = init_h
        for t in range(t_len):
            cur_h = self.gru(x_seq[t], cur_h)
            h_list.append(cur_h)
            cur_h = cur_h * (1.0 - done_seq[t]).unsqueeze(-1)
        h_all = torch.stack(h_list, dim=0)  # (T, N, H)
        return self.head_from_h(h_all, prev_act_seq)
