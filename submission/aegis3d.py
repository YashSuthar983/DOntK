"""AEGIS-3D: Embodied 3D Neural BEV Occupancy & Tri-Scale Episodic Memory Agent.

100% custom PyTorch neural architecture built from scratch for Track 1 ("Can It Play DOOM?").
Zero hardcoded action trees, zero classical pathfinders (A*/Dijkstra/BFS), zero atan2 aimbots.
Every action decision flows strictly through trained neural network parameters.

Empirically calibrated constants (verified via training/calibration_probe.py on vizdoom==1.3.3):
- depth_buffer uint8: 1..62 linear (d = 7.11 * u8 + 8.9 map units); 63 = saturated far (>= 456u)
- 160x120 pinhole intrinsics: fx = 80.0, fy = 105.77, eye_height = 41.0 map units
- Freedoom 2 SELECTED_WEAPON IDs: 1=Fist/Chainsaw, 2=Pistol, 3=Shotgun/SSG, 4=Chaingun, 5=RL, 6=Plasma, 7=BFG
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Empirically measured ViZDoom 1.3.3 + Freedoom 2 camera & depth constants
DEPTH_SLOPE = 7.11
DEPTH_INTERCEPT = 8.9
DEPTH_SAT_U8 = 63
FX = 80.0
FY = 105.77
EYE_HEIGHT = 41.0

BEV_SIZE = 64
BEV_RES = 32.0  # 64 * 32 = 2048 map units (-1024..+1024 around player)
BEV_CHANNELS = 24
NUM_SLOTS = 32
SLOT_DIM = 64
NUM_KEYFRAMES = 32
KF_DIM = 96
E3B_DIM = 64
GRU_DIM = 256

# Factored conflict-free action head cardinalities:
# MoveFB(3) x StrafeLR(3) x TurnLR(3) x Attack(2) x Use(2) x NextWeapon(2)
ACTION_DIMS = (3, 3, 3, 2, 2, 2)


def factored_to_multibinary(factored: torch.Tensor) -> torch.Tensor:
    """Convert (..., 6) factored indices -> (..., 9) MultiBinary action tensor."""
    out = torch.zeros((*factored.shape[:-1], 9), dtype=torch.int8, device=factored.device)
    fb = factored[..., 0]
    lr = factored[..., 1]
    tn = factored[..., 2]
    out[..., 0] = (fb == 1).to(torch.int8)  # MOVE_FORWARD
    out[..., 1] = (fb == 2).to(torch.int8)  # MOVE_BACKWARD
    out[..., 2] = (lr == 1).to(torch.int8)  # MOVE_LEFT
    out[..., 3] = (lr == 2).to(torch.int8)  # MOVE_RIGHT
    out[..., 4] = (tn == 1).to(torch.int8)  # TURN_LEFT
    out[..., 5] = (tn == 2).to(torch.int8)  # TURN_RIGHT
    out[..., 6] = factored[..., 3].to(torch.int8)  # ATTACK
    out[..., 7] = factored[..., 4].to(torch.int8)  # USE
    out[..., 8] = factored[..., 5].to(torch.int8)  # SELECT_NEXT_WEAPON
    return out


@dataclass
class AgentMemoryState:
    gru_h: torch.Tensor          # (B, GRU_DIM)
    bev_map: torch.Tensor        # (B, BEV_CHANNELS, BEV_SIZE, BEV_SIZE)
    slot_xy: torch.Tensor        # (B, NUM_SLOTS, 2) ego (fwd, right) in map units
    slot_conf: torch.Tensor      # (B, NUM_SLOTS, 1)
    slot_feat: torch.Tensor      # (B, NUM_SLOTS, SLOT_DIM)
    kf_feat: torch.Tensor        # (B, NUM_KEYFRAMES, KF_DIM)
    kf_mask: torch.Tensor        # (B, NUM_KEYFRAMES)
    kf_ptr: torch.Tensor         # (B,) int64 ring pointer
    e3b_cov_inv: torch.Tensor    # (B, E3B_DIM, E3B_DIM)
    ray_hist: torch.Tensor       # (B, 4, 64)
    ray_type_hist: torch.Tensor  # (B, 4, 64) int64
    prev_actions: torch.Tensor   # (B, 4, 9)
    prev_vitals: torch.Tensor    # (B, 17)
    wpn_cooldown: torch.Tensor   # (B,) int64

    def detach_(self) -> "AgentMemoryState":
        return AgentMemoryState(
            gru_h=self.gru_h.detach(),
            bev_map=self.bev_map.detach(),
            slot_xy=self.slot_xy.detach(),
            slot_conf=self.slot_conf.detach(),
            slot_feat=self.slot_feat.detach(),
            kf_feat=self.kf_feat.detach(),
            kf_mask=self.kf_mask.detach(),
            kf_ptr=self.kf_ptr.detach(),
            e3b_cov_inv=self.e3b_cov_inv.detach(),
            ray_hist=self.ray_hist.detach(),
            ray_type_hist=self.ray_type_hist.detach(),
            prev_actions=self.prev_actions.detach(),
            prev_vitals=self.prev_vitals.detach(),
            wpn_cooldown=self.wpn_cooldown.detach(),
        )


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size, stride=stride, padding=padding, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.pw(self.dw(x))))


class CalibratedRGBDStem(nn.Module):
    """Pinhole 3D unprojector + lightweight stem -> Frustum (15x20x48) + 20 Horizon Tokens (20x96)."""

    def __init__(self) -> None:
        super().__init__()
        # Precompute pixel ray-direction grids for 120x160
        u = torch.arange(160, dtype=torch.float32)
        v = torch.arange(120, dtype=torch.float32)
        vv, uu = torch.meshgrid(v, u, indexing="ij")
        self.register_buffer("dir_right", (uu - 79.5) / FX, persistent=False)  # (120, 160)
        self.register_buffer("dir_up", (59.5 - vv) / FY, persistent=False)     # (120, 160)

        # Downsampled 15x20 cell centers for fast frustum-to-BEV splatting
        u_s = torch.linspace(4.0, 156.0, 20)
        self.register_buffer("frust_dir_right", (u_s - 79.5) / FX, persistent=False)  # (20,)

        # 10 channels -> 30x40 -> 15x20
        self.conv1 = nn.Sequential(
            nn.Conv2d(10, 32, kernel_size=4, stride=4, padding=0, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
        )
        self.conv2 = DepthwiseSeparableConv2d(32, 48, kernel_size=3, stride=2, padding=1)
        self.conv3 = DepthwiseSeparableConv2d(48, 48, kernel_size=3, stride=1, padding=1)

        # Vertical attention pooling: (B, 48, 15, 20) -> (B, 20, 96)
        self.vert_attn = nn.Conv2d(48, 1, kernel_size=1)
        self.horizon_proj = nn.Sequential(nn.Linear(48, 96), nn.LayerNorm(96), nn.SiLU(inplace=True))
        self.frust_to_bev = nn.Conv2d(48, BEV_CHANNELS, kernel_size=1)

    def forward(self, rgb: torch.Tensor, depth_u8: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """rgb: (B, 120, 160, 3) float/uint8; depth_u8: (B, 120, 160) float/uint8."""
        b = rgb.shape[0]
        rgb_f = rgb.to(torch.float32).permute(0, 3, 1, 2) / 255.0  # (B, 3, 120, 160)
        dep_f = depth_u8.to(torch.float32)  # (B, 120, 160)

        valid_near = (dep_f >= 1.0) & (dep_f < float(DEPTH_SAT_U8))
        d_units = torch.where(valid_near, dep_f * DEPTH_SLOPE + DEPTH_INTERCEPT, torch.full_like(dep_f, 512.0))
        y_right = d_units * self.dir_right.unsqueeze(0)
        z_floor = EYE_HEIGHT + d_units * self.dir_up.unsqueeze(0)

        # Explicit 3D traversability channels (cures sensors.py abs(step) > 24 bug!)
        step_ok = ((z_floor.abs() <= 24.0) & valid_near).to(torch.float32)
        dropoff = ((z_floor < -24.0) & valid_near).to(torch.float32)
        high_obs = ((z_floor > 24.0) & valid_near).to(torch.float32)

        x = torch.cat(
            [
                rgb_f,
                (d_units / 512.0).unsqueeze(1),
                (1.0 / (1.0 + d_units / 128.0)).unsqueeze(1),
                (y_right.clamp(-512.0, 512.0) / 512.0).unsqueeze(1),
                (z_floor.clamp(-128.0, 128.0) / 128.0).unsqueeze(1),
                step_ok.unsqueeze(1),
                dropoff.unsqueeze(1),
                high_obs.unsqueeze(1),
            ],
            dim=1,
        )  # (B, 10, 120, 160)

        feat = self.conv3(self.conv2(self.conv1(x)))  # (B, 48, 15, 20)

        # Vertical attention pooling -> 20 horizontal FOV tokens
        v_w = torch.softmax(self.vert_attn(feat), dim=2)  # (B, 1, 15, 20)
        h_col = (feat * v_w).sum(dim=2).transpose(1, 2)   # (B, 20, 48)
        horizon_tokens = self.horizon_proj(h_col)         # (B, 20, 96)

        # Unproject non-saturated frustum features (15x20) into local (64x64) Ego-BEV grid
        bev_feat = self.frust_to_bev(feat)  # (B, C, 15, 20)
        d_pool = F.avg_pool2d(d_units.unsqueeze(1), kernel_size=8, stride=8).squeeze(1)  # (B, 15, 20)
        near_pool = F.avg_pool2d(valid_near.to(torch.float32).unsqueeze(1), kernel_size=8, stride=8).squeeze(1) > 0.5

        x_fwd = d_pool  # (B, 15, 20)
        y_lat = d_pool * self.frust_dir_right.view(1, 1, 20)
        gx = torch.floor(x_fwd / BEV_RES + BEV_SIZE / 2.0).to(torch.int64)
        gy = torch.floor(y_lat / BEV_RES + BEV_SIZE / 2.0).to(torch.int64)
        inb = near_pool & (gx >= 0) & (gx < BEV_SIZE) & (gy >= 0) & (gy < BEV_SIZE)

        bev_splat = torch.zeros((b, BEV_CHANNELS, BEV_SIZE * BEV_SIZE), dtype=feat.dtype, device=feat.device)
        flat_idx = (gx.clamp(0, BEV_SIZE - 1) * BEV_SIZE + gy.clamp(0, BEV_SIZE - 1)).view(b, -1)  # (B, 300)
        src = (bev_feat.view(b, BEV_CHANNELS, -1) * inb.view(b, 1, -1).to(feat.dtype))
        bev_splat.scatter_add_(2, flat_idx.unsqueeze(1).expand(-1, BEV_CHANNELS, -1), src)
        bev_splat = bev_splat.view(b, BEV_CHANNELS, BEV_SIZE, BEV_SIZE)
        return horizon_tokens, bev_splat, feat


class CircularRayAndOdometry(nn.Module):
    """360-deg circular ray encoder + Differentiable Circular FFT Polar Scan-Matcher for ego-pose delta."""

    def __init__(self) -> None:
        super().__init__()
        self.hit_emb = nn.Embedding(6, 8)
        # Input per ray: 4 stacked distances + 8 hit emb + 1 angular diff + 1 temporal diff = 14
        self.conv = nn.Sequential(
            nn.Conv1d(14, 48, kernel_size=5, padding=2, padding_mode="circular", bias=False),
            nn.GroupNorm(8, 48),
            nn.SiLU(inplace=True),
            nn.Conv1d(48, 64, kernel_size=5, padding=2, padding_mode="circular", bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
        )
        self.ray_to_bev = nn.Conv1d(64, BEV_CHANNELS, kernel_size=1)
        self.ray_pool = nn.Sequential(nn.Linear(64 * 2, 96), nn.LayerNorm(96), nn.SiLU(inplace=True))

        # Odometry head: 64 circular cross-corr bins + 3 vel + 36 prev_actions -> (dx, dy, u_sin, v_cos)
        self.odom_mlp = nn.Sequential(
            nn.Linear(64 + 3 + 36, 96),
            nn.SiLU(inplace=True),
            nn.Linear(96, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 4),
        )
        # Initialize cos(dtheta) bias to +1.0 so untrained startup is identity rotation (zero spin!)
        nn.init.zeros_(self.odom_mlp[-1].weight)
        with torch.no_grad():
            self.odom_mlp[-1].bias.copy_(torch.tensor([0.0, 0.0, 0.0, 1.0]))

        angles = torch.arange(64, dtype=torch.float32) * (2.0 * math.pi / 64.0)
        # Ego frame: angle=0 is +fwd, angle=+pi/2 is left (-right)
        self.register_buffer("cos_a", torch.cos(angles), persistent=False)
        self.register_buffer("sin_a", torch.sin(angles), persistent=False)

    def forward(
        self,
        ray_hist: torch.Tensor,
        ray_type: torch.Tensor,
        vel_xyz: torch.Tensor,
        prev_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b = ray_hist.shape[0]
        cur_d = ray_hist[:, 0]    # (B, 64)
        prev_d = ray_hist[:, 1]   # (B, 64)
        ang_diff = cur_d - torch.roll(cur_d, shifts=1, dims=-1)
        tmp_diff = cur_d - prev_d
        h_emb = self.hit_emb(ray_type.clamp(0, 5)).transpose(1, 2)  # (B, 8, 64)
        x = torch.cat([ray_hist, h_emb, ang_diff.unsqueeze(1), tmp_diff.unsqueeze(1)], dim=1)  # (B, 14, 64)
        polar = self.conv(x)  # (B, 64, 64)

        # Global polar summary (mean + max over 360 deg)
        ray_global = self.ray_pool(torch.cat([polar.mean(dim=-1), polar.max(dim=-1).values], dim=-1))

        # Splat 360-deg ray endpoints into 64x64 Ego-BEV grid (covers 270 deg rear/flank + far 456..1024u!)
        r_units = cur_d * 1024.0
        xf = r_units * self.cos_a.unsqueeze(0)
        yr = -r_units * self.sin_a.unsqueeze(0)  # +sin is left, so right = -sin
        gx = torch.floor(xf / BEV_RES + BEV_SIZE / 2.0).to(torch.int64)
        gy = torch.floor(yr / BEV_RES + BEV_SIZE / 2.0).to(torch.int64)
        valid = (cur_d < 0.999) & (gx >= 0) & (gx < BEV_SIZE) & (gy >= 0) & (gy < BEV_SIZE)
        flat_idx = gx.clamp(0, BEV_SIZE - 1) * BEV_SIZE + gy.clamp(0, BEV_SIZE - 1)
        ray_bev_feat = self.ray_to_bev(polar) * valid.unsqueeze(1).to(polar.dtype)  # (B, C, 64)
        bev_splat = torch.zeros((b, BEV_CHANNELS, BEV_SIZE * BEV_SIZE), dtype=polar.dtype, device=polar.device)
        bev_splat.scatter_add_(2, flat_idx.unsqueeze(1).expand(-1, BEV_CHANNELS, -1), ray_bev_feat)
        bev_splat = bev_splat.view(b, BEV_CHANNELS, BEV_SIZE, BEV_SIZE)

        # Differentiable circular FFT cross-correlation between prev_d and cur_d for exact dtheta
        # Run in fp32 so cuFFT never fails under CUDA AMP (bf16/fp16)
        with torch.amp.autocast(device_type=cur_d.device.type, enabled=False):
            p32 = prev_d.float()
            c32 = cur_d.float()
            f_prev = torch.fft.rfft(p32 - p32.mean(dim=-1, keepdim=True), dim=-1)
            f_cur = torch.fft.rfft(c32 - c32.mean(dim=-1, keepdim=True), dim=-1)
            xcorr = torch.fft.irfft(f_cur * torch.conj(f_prev), n=64, dim=-1)  # (B, 64)

        odom_in = torch.cat([xcorr, vel_xyz / 20.0, prev_actions.reshape(b, -1)], dim=-1)
        raw_odom = self.odom_mlp(odom_in)
        dx_dy = raw_odom[:, :2] * 32.0  # map units per 4-tic step
        uv = raw_odom[:, 2:4]
        uv = uv / (uv.norm(dim=-1, keepdim=True) + 1e-6)  # orthonormal (sin, cos)
        odom = torch.cat([dx_dy, uv], dim=-1)  # (B, 4): [dx_fwd, dy_right, sin_dtheta, cos_dtheta]
        return polar.transpose(1, 2), ray_global, bev_splat, odom


class EntitySetNet(nn.Module):
    """Permutation-invariant Masked Set-Transformer over up to 32 visible entities + TTC/CPA kinematics."""

    def __init__(self) -> None:
        super().__init__()
        self.type_emb = nn.Embedding(12, 16)
        # 5 raw features + 3 derived (ttc, cpa, splash_risk) + 16 type_emb = 24
        self.in_proj = nn.Sequential(nn.Linear(24, 96), nn.LayerNorm(96), nn.SiLU(inplace=True))
        layer = nn.TransformerEncoderLayer(
            d_model=96, nhead=4, dim_feedforward=192, dropout=0.0, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.query = nn.Parameter(torch.randn(1, 1, 96) * 0.02)
        self.pool_attn = nn.MultiheadAttention(96, num_heads=4, batch_first=True)
        self.ent_to_bev = nn.Linear(96, BEV_CHANNELS)

    def forward(
        self,
        features: torch.Tensor,
        etype: torch.Tensor,
        mask: torch.Tensor,
        selected_weapon: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b = features.shape[0]
        dist_u = features[..., 0] * 1024.0
        sin_a = features[..., 1]
        cos_a = features[..., 2]
        v_fwd = features[..., 3]
        v_right = features[..., 4]

        # Closing speed & Time-To-Collision / Closest Point of Approach
        v_closing = -(v_fwd * cos_a - v_right * sin_a)
        ttc = torch.clamp(dist_u / torch.clamp(v_closing, min=0.5), 0.0, 100.0) / 100.0
        cpa = torch.abs(dist_u * sin_a + v_right * ttc * 100.0) / 256.0
        is_rl = (selected_weapon.round() == 5.0).unsqueeze(1)
        splash_risk = (is_rl & (dist_u < 160.0) & (mask > 0)).to(torch.float32)

        x = torch.cat(
            [
                features[..., :3],
                (features[..., 3:5] / 25.0).clamp(-4.0, 4.0),
                ttc.unsqueeze(-1),
                cpa.clamp(0.0, 4.0).unsqueeze(-1),
                splash_risk.unsqueeze(-1),
                self.type_emb(etype.clamp(0, 11)),
            ],
            dim=-1,
        )  # (B, 32, 24)
        tok = self.in_proj(x)
        key_pad = mask <= 0  # True = ignored
        # Prevent all-True NaN in TransformerEncoder by unmasking slot 0 when empty
        any_valid = (~key_pad).any(dim=1)  # (B,)
        safe_pad = key_pad.clone()
        safe_pad[~any_valid, 0] = False
        enc = self.encoder(tok, src_key_padding_mask=safe_pad)

        q = self.query.expand(b, -1, -1)
        pooled, _ = self.pool_attn(q, enc, enc, key_padding_mask=safe_pad, need_weights=False)
        combat_ctx = pooled.squeeze(1) * any_valid.unsqueeze(-1).to(tok.dtype)  # (B, 96)

        # Splat visible entities into 64x64 Ego-BEV
        xf = dist_u * cos_a
        yr = -dist_u * sin_a
        gx = torch.floor(xf / BEV_RES + BEV_SIZE / 2.0).to(torch.int64)
        gy = torch.floor(yr / BEV_RES + BEV_SIZE / 2.0).to(torch.int64)
        inb = (mask > 0) & (gx >= 0) & (gx < BEV_SIZE) & (gy >= 0) & (gy < BEV_SIZE)
        flat_idx = gx.clamp(0, BEV_SIZE - 1) * BEV_SIZE + gy.clamp(0, BEV_SIZE - 1)
        ent_bev = self.ent_to_bev(enc).transpose(1, 2) * inb.unsqueeze(1).to(enc.dtype)  # (B, C, 32)
        bev_splat = torch.zeros((b, BEV_CHANNELS, BEV_SIZE * BEV_SIZE), dtype=enc.dtype, device=enc.device)
        bev_splat.scatter_add_(2, flat_idx.unsqueeze(1).expand(-1, BEV_CHANNELS, -1), ent_bev)
        return combat_ctx, bev_splat.view(b, BEV_CHANNELS, BEV_SIZE, BEV_SIZE)


class NeuralBEVMapModule(nn.Module):
    """Differentiable SE(2) grid_sample ego-warped Neural BEV Occupancy Map + ConvGRU + FPN Readout."""

    def __init__(self) -> None:
        super().__init__()
        # Depthwise-separable ConvGRU update on (BEV_CHANNELS, 64, 64)
        self.gates = nn.Conv2d(BEV_CHANNELS * 2, BEV_CHANNELS * 2, kernel_size=3, padding=1)
        self.cand = nn.Conv2d(BEV_CHANNELS * 2, BEV_CHANNELS, kernel_size=3, padding=1)
        # Multi-scale neural spatial readout (zero A*/BFS!)
        self.readout = nn.Sequential(
            DepthwiseSeparableConv2d(BEV_CHANNELS, 32, stride=2),  # 32x32
            DepthwiseSeparableConv2d(32, 48, stride=2),            # 16x16
            DepthwiseSeparableConv2d(48, 64, stride=2),            # 8x8
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(64 * 4, 128),
            nn.LayerNorm(128),
            nn.SiLU(inplace=True),
        )

    @staticmethod
    def warp_bev(bev: torch.Tensor, odom: torch.Tensor) -> torch.Tensor:
        """Rigidly warp previous ego-BEV into new ego frame using odom=[dx_fwd, dy_right, sin, cos]."""
        b = bev.shape[0]
        o32 = odom.float()
        dx = o32[:, 0] / (BEV_SIZE * BEV_RES / 2.0)  # normalized [-1, 1]
        dy = o32[:, 1] / (BEV_SIZE * BEV_RES / 2.0)
        s = o32[:, 2]
        c = o32[:, 3]
        # grid_sample uses (x=col=right, y=row=fwd)
        # Inverse transform: old_pt = R(+dtheta) * new_pt + [dx, dy]
        theta = torch.zeros((b, 2, 3), dtype=torch.float32, device=bev.device)
        theta[:, 0, 0] = c
        theta[:, 0, 1] = -s
        theta[:, 0, 2] = dy
        theta[:, 1, 0] = s
        theta[:, 1, 1] = c
        theta[:, 1, 2] = dx
        grid = F.affine_grid(theta, bev.size(), align_corners=False)
        return F.grid_sample(bev.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False).to(bev.dtype)

    def forward(self, prev_bev: torch.Tensor, obs_splat: torch.Tensor, odom: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        warped = self.warp_bev(prev_bev, odom)
        z, r = torch.chunk(torch.sigmoid(self.gates(torch.cat([warped, obs_splat], dim=1))), 2, dim=1)
        h_tilde = torch.tanh(self.cand(torch.cat([r * warped, obs_splat], dim=1)))
        new_bev = (1.0 - z) * warped + z * h_tilde
        nav_ctx = self.readout(new_bev)
        return new_bev, nav_ctx


class SlotLandmarkMemory(nn.Module):
    """Exact 2x2 SE(2)-rotated discrete neural landmark slots (Doors, Locks, Switches, Exits, Items)."""

    def __init__(self) -> None:
        super().__init__()
        self.write_gate = nn.Sequential(nn.Linear(192, 64), nn.SiLU(inplace=True), nn.Linear(64, 3 + SLOT_DIM))
        self.query_proj = nn.Linear(192 + 6, SLOT_DIM)
        self.key_proj = nn.Linear(SLOT_DIM + 3, SLOT_DIM)
        self.val_proj = nn.Linear(SLOT_DIM + 3, SLOT_DIM)
        self.out_proj = nn.Sequential(nn.Linear(SLOT_DIM, 96), nn.LayerNorm(96), nn.SiLU(inplace=True))

    def forward(
        self,
        slot_xy: torch.Tensor,
        slot_conf: torch.Tensor,
        slot_feat: torch.Tensor,
        odom: torch.Tensor,
        scene_ctx: torch.Tensor,
        keys_held: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. Exact analytical ego-frame transform of all stored (fwd, right) landmark coordinates
        dx, dy, s, c = odom[:, 0:1], odom[:, 1:2], odom[:, 2:3], odom[:, 3:4]
        xf = slot_xy[..., 0] - dx
        yr = slot_xy[..., 1] - dy
        new_xf = xf * c - yr * s
        new_yr = xf * s + yr * c
        new_xy = torch.stack([new_xf, new_yr], dim=-1).clamp(-4096.0, 4096.0)

        # 2. Learned gated write into the lowest-confidence slot when salient landmark is observed
        w_raw = self.write_gate(scene_ctx)  # (B, 3 + SLOT_DIM)
        w_prob = torch.sigmoid(w_raw[:, 0:1])  # write strength in [0, 1]
        cand_xy = torch.tanh(w_raw[:, 1:3]) * 256.0  # local landmark offset in front/around player
        cand_feat = w_raw[:, 3:]  # (B, SLOT_DIM)

        # Soft-argmin over existing slot confidences for differentiable slot allocation
        slot_weights = torch.softmax(-slot_conf.squeeze(-1) * 5.0, dim=-1).unsqueeze(-1) * w_prob.unsqueeze(-1)  # (B, K, 1)
        new_xy = (1.0 - slot_weights) * new_xy + slot_weights * cand_xy.unsqueeze(1)
        new_feat = (1.0 - slot_weights) * slot_feat + slot_weights * cand_feat.unsqueeze(1)
        new_conf = (1.0 - slot_weights) * (slot_conf * 0.998) + slot_weights * torch.ones_like(slot_conf)

        # 3. Key-Inventory-Conditioned Cross-Attention readout
        slot_geom = torch.cat([new_xy / 512.0, new_conf], dim=-1)  # (B, K, 3)
        slot_in = torch.cat([new_feat, slot_geom], dim=-1)
        q = self.query_proj(torch.cat([scene_ctx, keys_held], dim=-1)).unsqueeze(1)  # (B, 1, D)
        k = self.key_proj(slot_in)
        v = self.val_proj(slot_in)
        attn = torch.softmax((q @ k.transpose(1, 2)) / math.sqrt(SLOT_DIM), dim=-1)
        readout = self.out_proj((attn @ v).squeeze(1))
        return new_xy, new_conf, new_feat, readout


class KeyframeSMT(nn.Module):
    """Drift-free Scene Memory Transformer storing 32 salience-gated visual-polar keyframes."""

    def __init__(self) -> None:
        super().__init__()
        self.to_kf = nn.Linear(192, KF_DIM)
        self.salience = nn.Linear(192 + 1, 1)
        self.attn = nn.MultiheadAttention(KF_DIM, num_heads=4, batch_first=True)
        self.out_proj = nn.Sequential(nn.Linear(KF_DIM, 96), nn.LayerNorm(96), nn.SiLU(inplace=True))

    def forward(
        self,
        kf_feat: torch.Tensor,
        kf_mask: torch.Tensor,
        kf_ptr: torch.Tensor,
        scene_ctx: torch.Tensor,
        novelty: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b = scene_ctx.shape[0]
        cur_kf = self.to_kf(scene_ctx)  # (B, KF_DIM)
        sal = torch.sigmoid(self.salience(torch.cat([scene_ctx, novelty], dim=-1))).squeeze(-1)  # (B,)
        should_write = (sal > 0.5) | (kf_mask.sum(dim=-1) < 1.0)

        new_feat = kf_feat.clone()
        new_mask = kf_mask.clone()
        new_ptr = kf_ptr.clone()
        b_idx = torch.arange(b, device=scene_ctx.device)
        if should_write.any():
            w_b = b_idx[should_write]
            w_p = kf_ptr[should_write]
            new_feat[w_b, w_p] = cur_kf[should_write].to(new_feat.dtype)
            new_mask[w_b, w_p] = 1.0
            new_ptr[w_b] = (w_p + 1) % NUM_KEYFRAMES

        key_pad = new_mask <= 0.5
        out, _ = self.attn(cur_kf.unsqueeze(1), new_feat.to(cur_kf.dtype), new_feat.to(cur_kf.dtype), key_padding_mask=key_pad, need_weights=False)
        return new_feat, new_mask, new_ptr, self.out_proj(out.squeeze(1))


class E3BModule(nn.Module):
    """Live Elliptical Episodic Bonus (Henaff et al., NeurIPS 2022) via Sherman-Morrison O(d^2) update."""

    def __init__(self, ridge: float = 0.1) -> None:
        super().__init__()
        self.ridge = ridge
        self.phi = nn.Sequential(nn.Linear(192, 128), nn.SiLU(inplace=True), nn.Linear(128, E3B_DIM))

    def forward(self, scene_ctx: torch.Tensor, cov_inv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.amp.autocast(device_type=scene_ctx.device.type, enabled=False):
            emb = F.normalize(self.phi(scene_ctx.float()), dim=-1)  # (B, D)
            c_inv = cov_inv.float()
            # u = C^-1 @ emb
            u = torch.bmm(c_inv, emb.unsqueeze(-1)).squeeze(-1)  # (B, D)
            bonus = (emb * u).sum(dim=-1, keepdim=True).clamp(0.0, 20.0)  # (B, 1)
            # Sherman-Morrison rank-1 update: C_new^-1 = C^-1 - (u u^T) / (1 + bonus)
            new_cov_inv = c_inv - torch.bmm(u.unsqueeze(2), u.unsqueeze(1)) / (1.0 + bonus.unsqueeze(-1))
        return emb, bonus, new_cov_inv


class AEGIS3D(nn.Module):
    """Full AEGIS-3D Student Policy (+ optional Training-Only Asymmetric Critic & Grounding Heads)."""

    def __init__(self, include_training_heads: bool = False) -> None:
        super().__init__()
        self.include_training_heads = include_training_heads

        self.rgbd_stem = CalibratedRGBDStem()
        self.ray_odom = CircularRayAndOdometry()
        self.entity_net = EntitySetNet()
        self.vitals_net = nn.Sequential(
            nn.Linear(17 + 6 + 2 + 36, 96),
            nn.LayerNorm(96),
            nn.SiLU(inplace=True),
        )

        # Dual-stream memory
        self.bev_module = NeuralBEVMapModule()
        self.slot_mem = SlotLandmarkMemory()
        self.kf_smt = KeyframeSMT()
        self.e3b = E3BModule()

        # Fine horizon attention (queries 20 RGB-D FOV columns with combat+vitals context)
        self.horizon_q = nn.Linear(192, 96)
        self.horizon_attn = nn.MultiheadAttention(96, num_heads=4, batch_first=True)

        # Fast Tactical GRU core
        # Inputs: horizon_ctx(96) + combat_ctx(96) + ray_global(96) + vitals(96) + bev(128) + slot(96) + smt(96) + e3b(1) = 705
        self.gru_in = nn.Sequential(nn.Linear(705, 256), nn.LayerNorm(256), nn.SiLU(inplace=True))
        self.gru = nn.GRUCell(256, GRU_DIM)

        # Conflict-free factored policy head
        self.action_heads = nn.ModuleList([nn.Linear(GRU_DIM, d) for d in ACTION_DIMS])

        if include_training_heads:
            # 1. Asymmetric Omniscient Critic V(gru_h, priv_vec)
            self.critic = nn.Sequential(
                nn.Linear(GRU_DIM + 24, 256),
                nn.SiLU(inplace=True),
                nn.Linear(256, 128),
                nn.SiLU(inplace=True),
                nn.Linear(128, 1),
            )
            # 2. Grounding head: BEV occupancy reconstruction (24 -> 4 channels)
            self.aux_bev = nn.Conv2d(BEV_CHANNELS, 4, kernel_size=1)
            # 3. Grounding head: 20-col visual affordances (96 -> 5 classes)
            self.aux_afford = nn.Linear(96, 5)
            # 4. E3B Inverse Dynamics Model: (phi_t, phi_{t+1}) -> factored action logits
            self.e3b_idm = nn.ModuleList(
                [nn.Sequential(nn.Linear(E3B_DIM * 2, 64), nn.SiLU(inplace=True), nn.Linear(64, d)) for d in ACTION_DIMS]
            )

    def init_state(self, batch_size: int, device: torch.device | str) -> AgentMemoryState:
        eye = torch.eye(E3B_DIM, device=device, dtype=torch.float32).unsqueeze(0).expand(batch_size, -1, -1).clone() / 0.1
        return AgentMemoryState(
            gru_h=torch.zeros((batch_size, GRU_DIM), device=device),
            bev_map=torch.zeros((batch_size, BEV_CHANNELS, BEV_SIZE, BEV_SIZE), device=device),
            slot_xy=torch.zeros((batch_size, NUM_SLOTS, 2), device=device),
            slot_conf=torch.zeros((batch_size, NUM_SLOTS, 1), device=device),
            slot_feat=torch.zeros((batch_size, NUM_SLOTS, SLOT_DIM), device=device),
            kf_feat=torch.zeros((batch_size, NUM_KEYFRAMES, KF_DIM), device=device),
            kf_mask=torch.zeros((batch_size, NUM_KEYFRAMES), device=device),
            kf_ptr=torch.zeros((batch_size,), dtype=torch.int64, device=device),
            e3b_cov_inv=eye,
            ray_hist=torch.ones((batch_size, 4, 64), device=device),
            ray_type_hist=torch.zeros((batch_size, 4, 64), dtype=torch.int64, device=device),
            prev_actions=torch.zeros((batch_size, 4, 9), device=device),
            prev_vitals=torch.zeros((batch_size, 17), device=device),
            wpn_cooldown=torch.zeros((batch_size,), dtype=torch.int64, device=device),
        )

    def forward_step(
        self,
        obs: dict[str, torch.Tensor],
        state: AgentMemoryState,
        *,
        ray_dropout_p: float = 0.0,
    ) -> tuple[list[torch.Tensor], AgentMemoryState, dict[str, torch.Tensor]]:
        b = obs["vitals"].shape[0]
        vitals = obs["vitals"].to(torch.float32)  # (B, 17)
        ray_d = obs["ray_distance"].to(torch.float32)  # (B, 64)
        ray_t = obs["ray_hit_type"].to(torch.int64)    # (B, 64)

        # Scheduled modality dropout during training (front 90-deg rays: indices 0..8 and 56..63)
        if ray_dropout_p > 0.0 and self.training:
            drop = (torch.rand((b, 1), device=ray_d.device) < ray_dropout_p).to(torch.float32)
            front_mask = torch.zeros((1, 64), device=ray_d.device)
            front_mask[:, :9] = 1.0
            front_mask[:, 56:] = 1.0
            m = drop * front_mask
            ray_d = ray_d * (1.0 - m) + m
            ray_t = torch.where(m > 0.5, torch.zeros_like(ray_t), ray_t)

        new_ray_hist = torch.cat([ray_d.unsqueeze(1), state.ray_hist[:, :3]], dim=1)
        new_ray_type_hist = torch.cat([ray_t.unsqueeze(1), state.ray_type_hist[:, :3]], dim=1)

        # 1. Encoders
        horizon_tokens, frust_bev, _ = self.rgbd_stem(obs["rgb"], obs["depth"])
        _polar_tok, ray_global, ray_bev, odom = self.ray_odom(
            new_ray_hist, ray_t, vitals[:, 14:17], state.prev_actions
        )
        combat_ctx, ent_bev = self.entity_net(
            obs["ent_features"].to(torch.float32),
            obs["ent_type"].to(torch.int64),
            obs["ent_mask"].to(torch.float32),
            vitals[:, 2],
        )

        # Vitals + event deltas (key_just_picked_up, hp_delta, speed_vs_move_cmd stuck signal)
        key_delta = (vitals[:, 8:14] - state.prev_vitals[:, 8:14]).clamp(0.0, 1.0)
        hp_delta = ((vitals[:, 0] - state.prev_vitals[:, 0]) / 50.0).clamp(-2.0, 2.0).unsqueeze(-1)
        speed = torch.hypot(vitals[:, 14], vitals[:, 15]).unsqueeze(-1) / 20.0
        v_norm = torch.cat(
            [
                vitals[:, 0:2] / 100.0,
                vitals[:, 2:3] / 7.0,
                vitals[:, 3:8] / 100.0,
                vitals[:, 8:14],
                vitals[:, 14:17] / 20.0,
                key_delta,
                hp_delta,
                speed,
                state.prev_actions.reshape(b, -1),
            ],
            dim=-1,
        )
        vitals_ctx = self.vitals_net(v_norm)  # (B, 96)

        # 2. Tri-Scale Memory Updates
        obs_splat = frust_bev + ray_bev + ent_bev
        new_bev, bev_ctx = self.bev_module(state.bev_map, obs_splat, odom)

        scene_ctx = torch.cat([ray_global, vitals_ctx], dim=-1)  # (B, 192)
        e3b_emb, e3b_bonus, new_cov_inv = self.e3b(scene_ctx, state.e3b_cov_inv)
        new_sxy, new_sconf, new_sfeat, slot_ctx = self.slot_mem(
            state.slot_xy, state.slot_conf, state.slot_feat, odom, scene_ctx, vitals[:, 8:14]
        )
        new_kff, new_kfm, new_kfp, smt_ctx = self.kf_smt(
            state.kf_feat, state.kf_mask, state.kf_ptr, scene_ctx, e3b_bonus
        )

        # 3. Fine Horizon Attention & Fast Tactical GRU
        hq = self.horizon_q(torch.cat([combat_ctx, vitals_ctx], dim=-1)).unsqueeze(1)
        h_ctx, _ = self.horizon_attn(hq, horizon_tokens, horizon_tokens, need_weights=False)
        h_ctx = h_ctx.squeeze(1)

        gru_feat = self.gru_in(
            torch.cat([h_ctx, combat_ctx, ray_global, vitals_ctx, bev_ctx, slot_ctx, smt_ctx, e3b_bonus], dim=-1)
        )
        new_h = self.gru(gru_feat, state.gru_h)

        # 4. Conflict-Free Factored Policy Logits + Legal Edge Masks
        logits = [head(new_h) for head in self.action_heads]
        # Legal button-down edge reset: if prev USE == 1, mask USE = 1 so DOOM edge-trigger resets cleanly
        prev_use = state.prev_actions[:, 0, 7] > 0.5
        if prev_use.any():
            use_mask = torch.zeros_like(logits[4])
            use_mask[prev_use, 1] = -1e9
            logits[4] = logits[4] + use_mask
        # Legal weapon-switch cooldown mask (prevents spinning through weapons during 24-tic raise anim)
        wpn_cd = state.wpn_cooldown > 0
        if wpn_cd.any():
            wpn_mask = torch.zeros_like(logits[5])
            wpn_mask[wpn_cd, 1] = -1e9
            logits[5] = logits[5] + wpn_mask

        next_state = AgentMemoryState(
            gru_h=new_h,
            bev_map=new_bev,
            slot_xy=new_sxy,
            slot_conf=new_sconf,
            slot_feat=new_sfeat,
            kf_feat=new_kff,
            kf_mask=new_kfm,
            kf_ptr=new_kfp,
            e3b_cov_inv=new_cov_inv,
            ray_hist=new_ray_hist,
            ray_type_hist=new_ray_type_hist,
            prev_actions=state.prev_actions,  # updated after sampling action
            prev_vitals=vitals,
            wpn_cooldown=(state.wpn_cooldown - 1).clamp(min=0),
        )

        extras = {
            "odom": odom,
            "e3b_emb": e3b_emb,
            "e3b_bonus": e3b_bonus,
            "horizon_tokens": horizon_tokens,
            "bev_map": new_bev,
            "gru_h": new_h,
        }
        return logits, next_state, extras

    @staticmethod
    def update_state_with_action(state: AgentMemoryState, action_mb: torch.Tensor) -> AgentMemoryState:
        """Record the executed MultiBinary(9) action into state.prev_actions & weapon cooldown."""
        a_f = action_mb.to(torch.float32)
        new_prev = torch.cat([a_f.unsqueeze(1), state.prev_actions[:, :3]], dim=1)
        new_cd = torch.where(a_f[:, 8] > 0.5, torch.full_like(state.wpn_cooldown, 5), state.wpn_cooldown)
        state.prev_actions = new_prev
        state.wpn_cooldown = new_cd
        return state

    def strip_for_submission(self) -> dict[str, torch.Tensor]:
        """Return state_dict containing ONLY inference-time student parameters (no Critic/Aux/Teacher)."""
        drop_prefixes = ("critic.", "aux_bev.", "aux_afford.", "e3b_idm.")
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if not k.startswith(drop_prefixes)}
