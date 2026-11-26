"""Ultra-Fast Parallel Online DAgger + Asymmetric Recurrent PPO Trainer for AEGIS-3D v2.

Designed to reach 90%+ zero-shot exit rate within a single 4-6h Colab GPU session:
- Multiprocessing `SubprocDoomVecEnv`: runs ViZDoom + Fog-of-War + 32x32 BEV targets +
  FrontierOracle in parallel worker processes.
- Vectorized `AEGIS3D.forward_sequence()`: batches RGBD CNN, Circular Ray+FFT,
  Entity Set-Transformer, and Vitals across (L*B) in ONE GPU kernel (20x-30x faster!).
- Stage 1 (`--mode dagger`): Online DAgger from the Fog-of-War FrontierOracle with
  class-balanced factored CE + Observed-Only BEV/Odometry/Affordance/IDM Grounding +
  Asymmetric Critic warm-start.
- Stage 2 (`--mode ppo`): Chunked Minibatched Asymmetric Recurrent PPO + Live E3B
  Curiosity + Annealed KL anchor to the DAgger checkpoint (zero ray-dropout mismatch!).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_ROOT / "env"), str(_ROOT / "submission")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from aegis3d import (
    ACTION_DIMS,
    AEGIS3D,
    AFFORD_COLS,
    AgentMemoryState,
    BEV_SIZE,
    factored_to_multibinary,
)
from training.frontier_oracle import FrontierOracle
from training.wad_oracle import FastDoomPackEnv


# ---------------------------------------------------------------------------
# Multiprocessing Vectorized Environment (ViZDoom + Fog-of-War + FrontierOracle)
# ---------------------------------------------------------------------------
def _worker_loop(conn, wad_paths: list[str], episode_ticks: int, gamma: float, seed_base: int, use_oracle: bool) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    envs = [
        FastDoomPackEnv(
            wp,
            episode_ticks=episode_ticks,
            render_buffers=True,
            gamma=gamma,
            compute_spatial_targets=True,
        )
        for wp in wad_paths
    ]
    oracles = [FrontierOracle() for _ in envs] if use_oracle else None

    def _pack(obs_l, inf_l, o_acts, rews, dones, exits):
        return {
            "rgb": np.stack([o["rgb"] for o in obs_l]),
            "depth": np.stack([o["depth"] for o in obs_l]),
            "vitals": np.stack([o["vitals"] for o in obs_l]),
            "ray_distance": np.stack([o["rays"]["distance"] for o in obs_l]),
            "ray_hit_type": np.stack([o["rays"]["hit_type"] for o in obs_l]),
            "ent_features": np.stack([o["visible_entities"]["features"] for o in obs_l]),
            "ent_type": np.stack([o["visible_entities"]["entity_type"] for o in obs_l]),
            "ent_mask": np.stack([o["visible_entities"]["mask"] for o in obs_l]),
            "priv_vec": np.stack([inf["priv_vec"] for inf in inf_l]),
            "odom_target": np.stack([inf["odom_target"] for inf in inf_l]),
            "bev_target": np.stack([inf["bev_target"] for inf in inf_l]),
            "afford_target": np.stack([inf["afford_target"] for inf in inf_l]),
            "oracle_act": o_acts,
            "reward": rews,
            "done": dones,
            "exit_reached": exits,
        }

    try:
        while True:
            cmd, data = conn.recv()
            if cmd == "reset":
                obs_l, inf_l = [], []
                o_acts = np.zeros((len(envs), 6), dtype=np.int64)
                for i, e in enumerate(envs):
                    o, inf = e.reset(seed=seed_base + i)
                    obs_l.append(o)
                    inf_l.append(inf)
                    if oracles is not None:
                        oracles[i].reset()
                        o_acts[i], _ = oracles[i].act(o, e)
                conn.send(_pack(
                    obs_l, inf_l, o_acts,
                    np.zeros(len(envs), dtype=np.float32),
                    np.zeros(len(envs), dtype=np.float32),
                    np.zeros(len(envs), dtype=np.float32),
                ))
            elif cmd == "step":
                acts_mb = data  # (K, 9) int8
                obs_l, inf_l = [], []
                o_acts = np.zeros((len(envs), 6), dtype=np.int64)
                rews = np.zeros(len(envs), dtype=np.float32)
                dones = np.zeros(len(envs), dtype=np.float32)
                exits = np.zeros(len(envs), dtype=np.float32)
                for i, e in enumerate(envs):
                    o, r, term, trunc, inf = e.step(acts_mb[i])
                    rews[i] = r
                    if term or trunc:
                        dones[i] = 1.0
                        exits[i] = 1.0 if inf.get("exit_reached") else 0.0
                        o, inf = e.reset()
                        if oracles is not None:
                            oracles[i].reset()
                    if oracles is not None:
                        o_acts[i], _ = oracles[i].act(o, e)
                    obs_l.append(o)
                    inf_l.append(inf)
                pkt = _pack(obs_l, inf_l, o_acts, rews, dones, exits)
                conn.send(pkt)
            elif cmd == "close":
                break
    finally:
        for e in envs:
            e.close()
        conn.close()


class SubprocDoomVecEnv:
    """Spawns `num_workers` processes, each hosting `envs_per_worker` ViZDoom envs."""

    def __init__(
        self,
        wads: list[str],
        num_envs: int,
        num_workers: int,
        episode_ticks: int,
        gamma: float,
        use_oracle: bool,
        seed: int = 1000,
    ) -> None:
        self.num_envs = num_envs
        self.num_workers = max(1, min(num_workers, num_envs))
        ctx = mp.get_context("spawn")
        self.conns = []
        self.procs = []

        # Distribute envs across workers evenly
        buckets: list[list[str]] = [[] for _ in range(self.num_workers)]
        for i in range(num_envs):
            buckets[i % self.num_workers].append(wads[i % len(wads)])

        self.worker_sizes = [len(b) for b in buckets]
        offset = 0
        for w_idx, w_wads in enumerate(buckets):
            parent_c, child_c = ctx.Pipe()
            p = ctx.Process(
                target=_worker_loop,
                args=(child_c, w_wads, episode_ticks, gamma, seed + offset * 17, use_oracle),
                daemon=True,
            )
            p.start()
            child_c.close()
            self.conns.append(parent_c)
            self.procs.append(p)
            offset += len(w_wads)

    def _merge(self, pkts: list[dict]) -> dict[str, np.ndarray]:
        return {k: np.concatenate([p[k] for p in pkts], axis=0) for k in pkts[0]}

    def reset(self) -> dict[str, np.ndarray]:
        for c in self.conns:
            c.send(("reset", None))
        return self._merge([c.recv() for c in self.conns])

    def step(self, actions_mb: np.ndarray) -> dict[str, np.ndarray]:
        start = 0
        for c, sz in zip(self.conns, self.worker_sizes):
            c.send(("step", actions_mb[start : start + sz]))
            start += sz
        return self._merge([c.recv() for c in self.conns])

    def close(self) -> None:
        for c in self.conns:
            try:
                c.send(("close", None))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5.0)
            if p.is_alive():
                p.terminate()


# ---------------------------------------------------------------------------
# Tensor helpers & losses
# ---------------------------------------------------------------------------
def _pkt_obs_to_torch(pkt: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "rgb": torch.from_numpy(pkt["rgb"]).to(device, non_blocking=True),
        "depth": torch.from_numpy(pkt["depth"]).to(device, non_blocking=True),
        "vitals": torch.from_numpy(pkt["vitals"]).to(device, non_blocking=True),
        "ray_distance": torch.from_numpy(pkt["ray_distance"]).to(device, non_blocking=True),
        "ray_hit_type": torch.from_numpy(pkt["ray_hit_type"]).to(device, non_blocking=True),
        "ent_features": torch.from_numpy(pkt["ent_features"]).to(device, non_blocking=True),
        "ent_type": torch.from_numpy(pkt["ent_type"]).to(device, non_blocking=True),
        "ent_mask": torch.from_numpy(pkt["ent_mask"]).to(device, non_blocking=True),
    }


def _reset_masked_state(state: AgentMemoryState, done_mask: torch.Tensor, fresh: AgentMemoryState) -> AgentMemoryState:
    if not (done_mask > 0.5).any():
        return state
    m = done_mask > 0.5
    m2 = m.view(-1, 1)
    m3 = m.view(-1, 1, 1)
    m4 = m.view(-1, 1, 1, 1)
    return AgentMemoryState(
        gru_h=torch.where(m2, fresh.gru_h, state.gru_h),
        bev_map=torch.where(m4, fresh.bev_map, state.bev_map),
        slot_xy=torch.where(m3, fresh.slot_xy, state.slot_xy),
        slot_conf=torch.where(m3, fresh.slot_conf, state.slot_conf),
        slot_feat=torch.where(m3, fresh.slot_feat, state.slot_feat),
        e3b_cov_inv=torch.where(m3, fresh.e3b_cov_inv, state.e3b_cov_inv),
        ray_hist=torch.where(m3, fresh.ray_hist, state.ray_hist),
        prev_actions=torch.where(m3, fresh.prev_actions, state.prev_actions),
        prev_vitals=torch.where(m2, fresh.prev_vitals, state.prev_vitals),
        wpn_cooldown=torch.where(m, fresh.wpn_cooldown, state.wpn_cooldown),
        steps_since_use=torch.where(m, fresh.steps_since_use, state.steps_since_use),
    )


def _slice_state(state: AgentMemoryState, idx: torch.Tensor) -> AgentMemoryState:
    return AgentMemoryState(
        gru_h=state.gru_h[idx],
        bev_map=state.bev_map[idx],
        slot_xy=state.slot_xy[idx],
        slot_conf=state.slot_conf[idx],
        slot_feat=state.slot_feat[idx],
        e3b_cov_inv=state.e3b_cov_inv[idx],
        ray_hist=state.ray_hist[idx],
        prev_actions=state.prev_actions[idx],
        prev_vitals=state.prev_vitals[idx],
        wpn_cooldown=state.wpn_cooldown[idx],
        steps_since_use=state.steps_since_use[idx],
    )


def _concat_states(states: list[AgentMemoryState]) -> AgentMemoryState:
    return AgentMemoryState(
        gru_h=torch.cat([s.gru_h for s in states], dim=0),
        bev_map=torch.cat([s.bev_map for s in states], dim=0),
        slot_xy=torch.cat([s.slot_xy for s in states], dim=0),
        slot_conf=torch.cat([s.slot_conf for s in states], dim=0),
        slot_feat=torch.cat([s.slot_feat for s in states], dim=0),
        e3b_cov_inv=torch.cat([s.e3b_cov_inv for s in states], dim=0),
        ray_hist=torch.cat([s.ray_hist for s in states], dim=0),
        prev_actions=torch.cat([s.prev_actions for s in states], dim=0),
        prev_vitals=torch.cat([s.prev_vitals for s in states], dim=0),
        wpn_cooldown=torch.cat([s.wpn_cooldown for s in states], dim=0),
        steps_since_use=torch.cat([s.steps_since_use for s in states], dim=0),
    )


def _kl_factored(anchor_logits: list[torch.Tensor], cur_logits: list[torch.Tensor]) -> torch.Tensor:
    total = torch.zeros((), device=cur_logits[0].device)
    for al, cl in zip(anchor_logits, cur_logits):
        valid = (cl > -1e6) & (al > -1e6)
        al_m = torch.where(valid, al.detach(), torch.full_like(al, -1e9))
        cl_m = torch.where(valid, cl, torch.full_like(cl, -1e9))
        ap = torch.softmax(al_m, dim=-1)
        log_ap = torch.log_softmax(al_m, dim=-1)
        log_cp = torch.log_softmax(cl_m, dim=-1)
        kl = torch.where(valid, ap * (log_ap - log_cp), torch.zeros_like(ap))
        total = total + kl.sum(dim=-1).mean()
    return total


def _dagger_ce_loss(
    logits: list[torch.Tensor],
    targets: torch.Tensor,
    head_weights: list[torch.Tensor],
) -> torch.Tensor:
    """Class-balanced factored cross-entropy safely ignoring legally masked steps."""
    total = torch.zeros((), device=targets.device)
    for i, (lg, w) in enumerate(zip(logits, head_weights)):
        # lg: (L, B, C), tgt: (L, B)
        lg_f = lg.float().reshape(-1, lg.shape[-1])
        tgt_f = targets[:, :, i].reshape(-1)
        # Skip rows where target action is legally masked (-1e9)
        picked_logit = lg_f.gather(1, tgt_f.unsqueeze(1)).squeeze(1)
        valid = picked_logit > -1e5
        if valid.any():
            ce = F.cross_entropy(lg_f[valid], tgt_f[valid], weight=w)
            total = total + ce
    return total


def _grounding_loss(
    model: AEGIS3D,
    extras: dict[str, torch.Tensor],
    odom_tgt: torch.Tensor,
    bev_tgt: torch.Tensor,
    aff_tgt: torch.Tensor,
    act_f: torch.Tensor,
    done_seq: torch.Tensor,
    aff_pos_weight: torch.Tensor,
) -> torch.Tensor:
    # 1. Balanced SE(2) Odometry SmoothL1 (equal scale for translation/16 and sharp angular heading!)
    l_odom_xy = F.smooth_l1_loss(extras["odom"][..., :2].float() / 16.0, odom_tgt[..., :2].float() / 16.0)
    l_odom_ang = F.smooth_l1_loss(extras["odom"][..., 2:4].float(), odom_tgt[..., 2:4].float(), beta=0.1)
    l_odom = l_odom_xy + 4.0 * l_odom_ang

    # 2. Observed-Only Masked 32x32 BEV BCE (zero hallucination behind walls!)
    L, B = done_seq.shape
    bev_pred = model.aux_bev(extras["bev_map"].reshape(L * B, *extras["bev_map"].shape[2:])).view(L, B, 5, BEV_SIZE, BEV_SIZE)
    obs_mask = bev_tgt[:, :, 0:1]  # (L, B, 1, 32, 32)
    bev_y = bev_tgt[:, :, 1:6]     # (L, B, 5, 32, 32)
    raw_bev_bce = F.binary_cross_entropy_with_logits(bev_pred.float(), bev_y.float(), reduction="none")
    l_bev = (raw_bev_bce * obs_mask.float()).sum() / (obs_mask.float().sum() * 5.0 + 1.0)

    # 3. 16-col Visual Affordances (pos-weighted for rare switches/exits/locks)
    aff_pred = model.aux_afford(extras["horizon_tokens"])
    l_aff = F.binary_cross_entropy_with_logits(aff_pred.float(), aff_tgt.float(), pos_weight=aff_pos_weight)

    # 4. E3B Inverse Dynamics Model (masked across episode resets)
    emb = extras["e3b_emb"]  # (L, B, D)
    l_idm = torch.zeros((), device=emb.device)
    if L > 1:
        keep = (done_seq[:-1] < 0.5).reshape(-1)
        if keep.any():
            idm_in = torch.cat([emb[:-1], emb[1:]], dim=-1).reshape((L - 1) * B, -1)[keep]
            for i, head in enumerate(model.e3b_idm):
                l_idm = l_idm + F.cross_entropy(head(idm_in).float(), act_f[:-1, :, i].reshape(-1)[keep])
            l_idm = l_idm / len(model.e3b_idm)

    return 2.0 * l_odom + 2.0 * l_bev + l_aff + 0.35 * l_idm


# ---------------------------------------------------------------------------
# Main Trainer
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("dagger", "ppo"))
    ap.add_argument("--wad", nargs="+", required=True)
    ap.add_argument("--num-envs", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--episode-ticks", type=int, default=2800)
    ap.add_argument("--rollout-steps", type=int, default=64)
    ap.add_argument("--chunk-len", type=int, default=32)
    ap.add_argument("--minibatches", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--total-steps", type=int, default=1_500_000)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip-eps", type=float, default=0.18)
    ap.add_argument("--ent-coef", type=float, default=0.008)
    ap.add_argument("--e3b-coef", type=float, default=0.015)
    ap.add_argument("--kl-coef", type=float, default=0.05)
    ap.add_argument("--ground-coef", type=float, default=0.35)
    ap.add_argument("--beta-start", type=float, default=0.90, help="DAgger initial expert prob")
    ap.add_argument("--beta-end", type=float, default=0.10, help="DAgger final expert prob")
    ap.add_argument("--target-kl", type=float, default=0.035)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--save", default="training/checkpoints/aegis_v2.pt")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--init-policy", default=None, help="Optional stripped policy checkpoint overlaid after --resume (keeps warmed critic/aux!)")
    ap.add_argument("--reset-step", action="store_true", help="Load weights only from --resume and start step/beta/LR from 0")
    ap.add_argument("--anchor-ckpt", default=None, help="Frozen DAgger anchor checkpoint for PPO KL")
    ap.add_argument("--export-submission", default="submission/weights/aegis3d.pt")
    ap.add_argument("--save-every", type=int, default=15)
    ap.add_argument("--milestone-every", type=int, default=250000, help="Save stripped milestone copy every N steps")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else (args.device if args.device != "auto" else "cpu")
    )
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))
    print(
        f"[AEGIS-3D v2 | {args.mode.upper()}] Device={device} AMP={use_amp}({amp_dtype}) "
        f"Envs={args.num_envs} Workers={args.num_workers} T={args.rollout_steps} L={args.chunk_len}",
        flush=True,
    )

    model = AEGIS3D(include_training_heads=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    global_step = 0
    update_idx = 0

    if args.resume and Path(args.resume).is_file():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        if not args.reset_step and "opt" in ckpt and ckpt.get("mode") == args.mode:
            opt.load_state_dict(ckpt["opt"])
            global_step = int(ckpt.get("global_step", 0))
            update_idx = int(ckpt.get("update_idx", 0))
        print(f"[AEGIS-3D v2] Resumed from {args.resume} (prev_mode={ckpt.get('mode')}, reset_step={args.reset_step}, step={global_step:,})", flush=True)

    if args.init_policy and Path(args.init_policy).is_file():
        pckpt = torch.load(args.init_policy, map_location=device, weights_only=False)
        psd = pckpt["model"] if isinstance(pckpt, dict) and "model" in pckpt else pckpt
        model.load_state_dict(psd, strict=False)
        print(f"[AEGIS-3D v2] Overlaid policy/backbone weights <- {args.init_policy}", flush=True)

    anchor: AEGIS3D | None = None
    if args.mode == "ppo" and args.anchor_ckpt and Path(args.anchor_ckpt).is_file():
        anchor = AEGIS3D(include_training_heads=False).to(device)
        ackpt = torch.load(args.anchor_ckpt, map_location=device, weights_only=False)
        sd = ackpt["model"] if "model" in ackpt else ackpt
        anchor.load_state_dict({k: v for k, v in sd.items() if not k.startswith(("critic.", "aux_bev.", "aux_afford.", "e3b_idm."))}, strict=True)
        anchor.eval()
        for p in anchor.parameters():
            p.requires_grad_(False)
        print(f"[AEGIS-3D v2] Loaded frozen DAgger KL Anchor <- {args.anchor_ckpt}", flush=True)

    # Class weights for DAgger balanced cross-entropy (upweights sparse high-impact actions!)
    head_weights = [
        torch.tensor([1.0, 1.0, 1.3], device=device),   # MoveFB
        torch.tensor([1.0, 1.4, 1.4], device=device),   # StrafeLR
        torch.tensor([1.0, 1.2, 1.2], device=device),   # TurnLR
        torch.tensor([1.0, 2.5], device=device),        # Attack
        torch.tensor([1.0, 4.0], device=device),        # Use
        torch.tensor([1.0, 3.0], device=device),        # NextWeapon
    ]
    aff_pos_weight = torch.full((5,), 8.0, device=device)

    vec_env = SubprocDoomVecEnv(
        wads=args.wad,
        num_envs=args.num_envs,
        num_workers=args.num_workers,
        episode_ticks=args.episode_ticks,
        gamma=args.gamma,
        use_oracle=(args.mode == "dagger"),
        seed=args.seed,
    )

    N = args.num_envs
    T = args.rollout_steps
    L = min(args.chunk_len, T)
    assert T % L == 0, f"rollout_steps ({T}) must be divisible by chunk_len ({L})"
    n_chunks_t = T // L
    total_chunks = n_chunks_t * N

    state = model.init_state(N, device)
    fresh_N = model.init_state(N, device)
    anchor_state = anchor.init_state(N, device) if anchor is not None else None
    fresh_mb = model.init_state(max(1, total_chunks // args.minibatches) + total_chunks, device)

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    export_path = Path(args.export_submission) if args.export_submission else None
    if export_path is not None:
        export_path.parent.mkdir(parents=True, exist_ok=True)

    pkt = vec_env.reset()
    ep_exits: list[float] = []
    ep_returns = np.zeros(N, dtype=np.float64)
    recent_returns: list[float] = []
    next_milestone = ((global_step // max(1, args.milestone_every)) + 1) * args.milestone_every if args.milestone_every > 0 else 0
    t_start = time.perf_counter()

    try:
        while global_step < args.total_steps:
            update_idx += 1
            progress = min(1.0, global_step / max(1, args.total_steps))
            # Cosine LR decay down to 15% of initial LR
            cur_lr = args.lr * (0.15 + 0.85 * 0.5 * (1.0 + np.cos(np.pi * progress)))
            for pg in opt.param_groups:
                pg["lr"] = cur_lr

            beta = args.beta_start + (args.beta_end - args.beta_start) * progress if args.mode == "dagger" else 0.0

            # Pre-allocate rollout buffers
            buf_rgb = torch.zeros((T, N, 120, 160, 3), dtype=torch.uint8, device=device)
            buf_dep = torch.zeros((T, N, 120, 160), dtype=torch.uint8, device=device)
            buf_vit = torch.zeros((T, N, 17), dtype=torch.float32, device=device)
            buf_rd = torch.zeros((T, N, 64), dtype=torch.float32, device=device)
            buf_rt = torch.zeros((T, N, 64), dtype=torch.int64, device=device)
            buf_ef = torch.zeros((T, N, 32, 5), dtype=torch.float32, device=device)
            buf_et = torch.zeros((T, N, 32), dtype=torch.int64, device=device)
            buf_em = torch.zeros((T, N, 32), dtype=torch.int8, device=device)

            buf_ray_hist = torch.zeros((T, N, 4, 64), dtype=torch.float32, device=device)
            buf_prev_act = torch.zeros((T, N, 4, 9), dtype=torch.float32, device=device)
            buf_prev_vit = torch.zeros((T, N, 17), dtype=torch.float32, device=device)
            buf_wpn_cd = torch.zeros((T, N), dtype=torch.int64, device=device)
            buf_since_use = torch.zeros((T, N), dtype=torch.int64, device=device)

            buf_priv = torch.zeros((T, N, 24), dtype=torch.float32, device=device)
            buf_odom = torch.zeros((T, N, 4), dtype=torch.float32, device=device)
            buf_bev = torch.zeros((T, N, 6, BEV_SIZE, BEV_SIZE), dtype=torch.float32, device=device)
            buf_aff = torch.zeros((T, N, AFFORD_COLS, 5), dtype=torch.float32, device=device)
            buf_oracle = torch.zeros((T, N, 6), dtype=torch.int64, device=device)

            buf_act_f = torch.zeros((T, N, 6), dtype=torch.int64, device=device)
            buf_logp = torch.zeros((T, N), dtype=torch.float32, device=device)
            buf_val = torch.zeros((T, N), dtype=torch.float32, device=device)
            buf_rew = torch.zeros((T, N), dtype=torch.float32, device=device)
            buf_done = torch.zeros((T, N), dtype=torch.float32, device=device)
            buf_anchor_logits: list[list[torch.Tensor]] | None = [[] for _ in ACTION_DIMS] if anchor is not None else None

            chunk_init_states: list[AgentMemoryState] = []

            model.eval()
            with torch.inference_mode():
                for t in range(T):
                    if t % L == 0:
                        chunk_init_states.append(state.detach_())

                    obs_t = _pkt_obs_to_torch(pkt, device)
                    priv_t = torch.from_numpy(pkt["priv_vec"]).to(device, non_blocking=True)
                    oracle_t = torch.from_numpy(pkt["oracle_act"]).to(device, non_blocking=True)

                    # Record exact inputs to forward_sequence at step t
                    buf_rgb[t] = obs_t["rgb"]
                    buf_dep[t] = obs_t["depth"]
                    buf_vit[t] = obs_t["vitals"]
                    buf_rd[t] = obs_t["ray_distance"]
                    buf_rt[t] = obs_t["ray_hit_type"]
                    buf_ef[t] = obs_t["ent_features"]
                    buf_et[t] = obs_t["ent_type"]
                    buf_em[t] = obs_t["ent_mask"]
                    buf_ray_hist[t] = torch.cat([obs_t["ray_distance"].unsqueeze(1), state.ray_hist[:, :3]], dim=1)
                    buf_prev_act[t] = state.prev_actions
                    buf_prev_vit[t] = state.prev_vitals
                    buf_wpn_cd[t] = state.wpn_cooldown
                    buf_since_use[t] = state.steps_since_use
                    buf_priv[t] = priv_t
                    buf_oracle[t] = oracle_t
                    # Issue #6 fix: align odom/BEV/affordance grounding targets with observation t (from the SAME packet!)
                    buf_odom[t] = torch.from_numpy(pkt["odom_target"]).to(device, non_blocking=True)
                    buf_bev[t] = torch.from_numpy(pkt["bev_target"]).to(device, non_blocking=True)
                    buf_aff[t] = torch.from_numpy(pkt["afford_target"]).to(device, non_blocking=True)

                    with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                        s_logits, next_state, extras = model.forward_step(obs_t, state)
                        val_t = model.critic(torch.cat([extras["gru_h"], priv_t], dim=-1)).squeeze(-1)

                    if anchor is not None and anchor_state is not None and buf_anchor_logits is not None:
                        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                            a_logits, next_a_state, _ = anchor.forward_step(obs_t, anchor_state)
                        for hi, al in enumerate(a_logits):
                            buf_anchor_logits[hi].append(al.float())
                    else:
                        next_a_state = None

                    s_dists = [Categorical(logits=lg.float()) for lg in s_logits]
                    if args.mode == "dagger":
                        student_f = torch.stack([d.sample() for d in s_dists], dim=-1)
                        use_exp = (torch.rand((N, 1), device=device) < beta)
                        acts_f = torch.where(use_exp, oracle_t, student_f)
                    else:
                        acts_f = torch.stack([d.sample() for d in s_dists], dim=-1)

                    logp = torch.stack([d.log_prob(acts_f[:, i]) for i, d in enumerate(s_dists)], dim=-1).sum(dim=-1)
                    acts_mb = factored_to_multibinary(acts_f)
                    next_state = model.update_state_with_action(next_state, acts_mb)
                    if next_a_state is not None:
                        next_a_state = anchor.update_state_with_action(next_a_state, acts_mb)

                    buf_act_f[t] = acts_f
                    buf_logp[t] = logp
                    buf_val[t] = val_t.float()

                    pkt = vec_env.step(acts_mb.cpu().numpy())
                    rew_np = pkt["reward"]
                    if args.mode == "ppo" and args.e3b_coef > 0:
                        rew_np = rew_np + args.e3b_coef * extras["e3b_bonus"][:, 0].float().cpu().numpy()
                    done_np = pkt["done"]
                    exit_np = pkt["exit_reached"]

                    ep_returns += rew_np
                    for i in range(N):
                        if done_np[i] > 0.5:
                            ep_exits.append(float(exit_np[i]))
                            recent_returns.append(float(ep_returns[i]))
                            ep_returns[i] = 0.0

                    done_t = torch.from_numpy(done_np).to(device, non_blocking=True)
                    buf_rew[t] = torch.from_numpy(rew_np).to(device, non_blocking=True)
                    buf_done[t] = done_t

                    state = _reset_masked_state(next_state, done_t, fresh_N)
                    if next_a_state is not None:
                        anchor_state = _reset_masked_state(next_a_state, done_t, fresh_N)
                    global_step += N

                # Bootstrap final value
                last_obs = _pkt_obs_to_torch(pkt, device)
                last_priv = torch.from_numpy(pkt["priv_vec"]).to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    _, _, last_ext = model.forward_step(last_obs, state)
                    next_val = model.critic(torch.cat([last_ext["gru_h"], last_priv], dim=-1)).squeeze(-1).float()

            # Compute GAE returns & advantages
            adv = torch.zeros((T, N), device=device)
            last_gae = torch.zeros(N, device=device)
            for t in reversed(range(T)):
                nonterm = 1.0 - buf_done[t]
                nv = next_val if t == T - 1 else buf_val[t + 1]
                delta = buf_rew[t] + args.gamma * nv * nonterm - buf_val[t]
                last_gae = delta + args.gamma * args.gae_lambda * nonterm * last_gae
                adv[t] = last_gae
            ret = adv + buf_val
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            # Reshape rollout into (L, total_chunks, ...) for fast chunked minibatched sequence updates!
            def _chunk(x: torch.Tensor) -> torch.Tensor:
                tail = x.shape[2:]
                return x.view(n_chunks_t, L, N, *tail).permute(1, 0, 2, *range(3, 3 + len(tail))).reshape(L, total_chunks, *tail)

            obs_c = {
                "rgb": _chunk(buf_rgb),
                "depth": _chunk(buf_dep),
                "vitals": _chunk(buf_vit),
                "ray_distance": _chunk(buf_rd),
                "ray_hit_type": _chunk(buf_rt),
                "ent_features": _chunk(buf_ef),
                "ent_type": _chunk(buf_et),
                "ent_mask": _chunk(buf_em),
            }
            rh_c = _chunk(buf_ray_hist)
            pa_c = _chunk(buf_prev_act)
            pv_c = _chunk(buf_prev_vit)
            wcd_c = _chunk(buf_wpn_cd)
            su_c = _chunk(buf_since_use)
            priv_c = _chunk(buf_priv)
            odom_c = _chunk(buf_odom)
            bev_c = _chunk(buf_bev)
            aff_c = _chunk(buf_aff)
            orc_c = _chunk(buf_oracle)
            act_c = _chunk(buf_act_f)
            logp_c = _chunk(buf_logp)
            val_c = _chunk(buf_val)
            adv_c = _chunk(adv)
            ret_c = _chunk(ret)
            done_c = _chunk(buf_done)
            init_c = _concat_states(chunk_init_states)
            anchor_c = [_chunk(torch.stack(buf_anchor_logits[hi], dim=0)) for hi in range(len(ACTION_DIMS))] if buf_anchor_logits is not None else None

            # Optimization epochs
            model.train()
            mb_size = max(1, total_chunks // args.minibatches)
            kl_beta = args.kl_coef * (0.1 + 0.9 * (1.0 - progress))
            log_loss, log_ce, log_gnd, log_ppo, log_kl, log_ent = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            n_mbs = 0
            stop_early = False

            for _ in range(args.epochs):
                perm = torch.randperm(total_chunks, device=device)
                for start in range(0, total_chunks, mb_size):
                    idx = perm[start : start + mb_size]
                    mb_obs = {k: v[:, idx] for k, v in obs_c.items()}
                    mb_init = _slice_state(init_c, idx)

                    opt.zero_grad(set_to_none=True)
                    with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                        s_logits, ext = model.forward_sequence(
                            mb_obs,
                            rh_c[:, idx],
                            pa_c[:, idx],
                            pv_c[:, idx],
                            wcd_c[:, idx],
                            su_c[:, idx],
                            done_c[:, idx],
                            mb_init,
                            fresh_mb,
                        )
                        val_new = model.critic(torch.cat([ext["gru_h"], priv_c[:, idx]], dim=-1)).squeeze(-1)
                        l_ground = _grounding_loss(
                            model, ext, odom_c[:, idx], bev_c[:, idx], aff_c[:, idx], act_c[:, idx], done_c[:, idx], aff_pos_weight
                        )

                        if args.mode == "dagger":
                            l_ce = _dagger_ce_loss(s_logits, orc_c[:, idx], head_weights)
                            l_val = F.smooth_l1_loss(val_new.float(), ret_c[:, idx].float())
                            loss = l_ce + args.ground_coef * l_ground + 0.15 * l_val
                            l_ppo_v, l_kl_v, ent_v = 0.0, 0.0, 0.0
                        else:
                            dists = [Categorical(logits=lg.float()) for lg in s_logits]
                            logp_new = torch.stack([d.log_prob(act_c[:, idx, i]) for i, d in enumerate(dists)], dim=-1).sum(dim=-1)
                            ent_mean = torch.stack([d.entropy() for d in dists], dim=-1).sum(dim=-1).mean()
                            log_ratio = logp_new - logp_c[:, idx]
                            ratio = torch.exp(log_ratio)
                            with torch.no_grad():
                                approx_kl = float(((ratio - 1.0) - log_ratio).mean().item())
                            if approx_kl > args.target_kl:
                                stop_early = True
                                break
                            mb_adv = adv_c[:, idx]
                            surr1 = ratio * mb_adv
                            surr2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * mb_adv
                            l_ppo = -torch.min(surr1, surr2).mean()
                            # Huber/SmoothL1 value loss so +49 exit jackpots NEVER crush PPO policy gradients!
                            l_val = F.smooth_l1_loss(val_new.float(), ret_c[:, idx].float(), beta=2.0)
                            cur_ent_coef = args.ent_coef * (0.4 + 0.6 * (1.0 - progress))
                            l_kl = _kl_factored([a[:, idx] for a in anchor_c], s_logits) if anchor_c is not None else torch.zeros((), device=device)
                            loss = l_ppo + 0.25 * l_val - cur_ent_coef * ent_mean + kl_beta * l_kl + args.ground_coef * l_ground
                            l_ce = torch.zeros((), device=device)
                            l_ppo_v, l_kl_v, ent_v = float(l_ppo.item()), float(l_kl.item()), float(ent_mean.item())

                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    scaler.step(opt)
                    scaler.update()

                    log_loss += float(loss.item())
                    log_ce += float(l_ce.item())
                    log_gnd += float(l_ground.item())
                    log_ppo += l_ppo_v
                    log_kl += l_kl_v
                    log_ent += ent_v
                    n_mbs += 1
                if stop_early:
                    break

            denom = max(1, n_mbs)
            fps = global_step / max(1e-3, time.perf_counter() - t_start)
            ex_rate = float(np.mean(ep_exits[-80:])) if ep_exits else 0.0
            mean_ret = float(np.mean(recent_returns[-80:])) if recent_returns else 0.0
            if args.mode == "dagger":
                print(
                    f"DAGGER upd={update_idx:04d} step={global_step:8d} fps={fps:5.0f} beta={beta:.2f} "
                    f"ret={mean_ret:6.2f} exit={ex_rate*100:5.1f}% loss={log_loss/denom:.3f} "
                    f"ce={log_ce/denom:.3f} gnd={log_gnd/denom:.3f}",
                    flush=True,
                )
            else:
                print(
                    f"PPO    upd={update_idx:04d} step={global_step:8d} fps={fps:5.0f} "
                    f"ret={mean_ret:6.2f} exit={ex_rate*100:5.1f}% loss={log_loss/denom:.3f} "
                    f"ppo={log_ppo/denom:+.3f} kl={log_kl/denom:.3f} gnd={log_gnd/denom:.3f} ent={log_ent/denom:.2f}",
                    flush=True,
                )

            if update_idx % args.save_every == 0 or global_step >= args.total_steps:
                tmp = save_path.with_suffix(".tmp")
                torch.save(
                    {
                        "mode": args.mode,
                        "model": model.state_dict(),
                        "opt": opt.state_dict(),
                        "global_step": global_step,
                        "update_idx": update_idx,
                    },
                    tmp,
                )
                tmp.replace(save_path)
                if export_path is not None:
                    etmp = export_path.with_suffix(".tmp")
                    stripped = model.strip_for_submission()
                    torch.save({"model": stripped}, etmp)
                    etmp.replace(export_path)
                    if next_milestone > 0 and global_step >= next_milestone:
                        mpath = save_path.parent / f"{save_path.stem}_{global_step // 1000}k.pt"
                        torch.save({"model": stripped}, mpath)
                        print(f"[AEGIS-3D v2] Milestone -> {mpath}", flush=True)
                        next_milestone += args.milestone_every
                print(f"[AEGIS-3D v2] Saved -> {save_path} | Exported -> {export_path}", flush=True)
    finally:
        vec_env.close()


if __name__ == "__main__":
    main()
