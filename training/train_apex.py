"""ApexNet Trainer: Kickstart DAgger -> Joint DAgger+PPO -> PPO with Exact EgoMap & ApexTeacher.

Combines:
- Subprocess vectorized ViZDoom workers running EgoMapBuilder + ApexTeacher (executed-action tracking)
- Truncated BPTT (32-step chunks) carrying both GRU hidden state and 8-dim Neural Map memory
- Asymmetric privileged critic (training-only) + detached value warm-up during DAgger
- Hierarchical 17-bin waypoint direction supervision + goal-kind & geodesic-distance aux heads
- 3-phase schedule:
    --mode dagger    : DAgger (beta_start -> beta_end), teacher CE + aux, detached value warm-up
    --mode kickstart : Joint Recurrent PPO + decaying teacher CE (student drives 100%)
    --mode ppo       : Recurrent PPO (optional light teacher anchor)
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
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "env"), str(ROOT / "submission")):
    if p not in sys.path:
        sys.path.insert(0, p)

from apex_net import ApexNet, HEADS, apex_factored_to_multibinary
from egomap import EgoMapBuilder, FINE_CROP, N_FIXED_CH, VEC_DIM, WIDE_CROP
from training.apex_teacher import ApexTeacher, GOAL_KIND_TO_ID, dir_rel_to_bin
from training.frontier_oracle import factored_np_to_multibinary
from training.wad_oracle import FastDoomPackEnv


def _worker_loop(
    conn,
    wad_paths: list[str],
    episode_ticks: int,
    gamma: float,
    seed_base: int,
    use_teacher: bool,
    cell_bonus: float,
) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    envs = [
        FastDoomPackEnv(
            wp,
            episode_ticks=episode_ticks,
            gamma=gamma,
            compute_spatial_targets=False,
        )
        for wp in wad_paths
    ]
    egomaps = [EgoMapBuilder() for _ in envs]
    teachers = [ApexTeacher() for _ in envs] if use_teacher else None

    def _pack(feats_l, priv_l, o_acts, dir_bins, goal_ids, goal_dists, rews, dones, exits):
        return {
            "vec": np.stack([f["vec"] for f in feats_l]),
            "rays": np.stack([f["rays"] for f in feats_l]),
            "ents": np.stack([f["ents"] for f in feats_l]),
            "etype": np.stack([f["etype"] for f in feats_l]),
            "fine_map": np.stack([f["fine_map"] for f in feats_l]),
            "wide_map": np.stack([f["wide_map"] for f in feats_l]),
            "crop_idx": np.stack([f["crop_idx"] for f in feats_l]),
            "pos_idx": np.array([f["pos_idx"] for f in feats_l], dtype=np.int64),
            "wmask": np.array([f["wmask"] for f in feats_l], dtype=np.float32),
            "priv_vec": np.stack(priv_l),
            "oracle_act": o_acts,
            "dir_bin": dir_bins,
            "goal_id": goal_ids,
            "goal_dist": goal_dists,
            "reward": rews,
            "done": dones,
            "exit": exits,
        }

    ep_counter = 0
    try:
        while True:
            cmd, payload = conn.recv()
            if cmd == "reset":
                feats_l, priv_l = [], []
                o_acts = np.zeros((len(envs), 6), dtype=np.int64)
                dir_bins = np.full(len(envs), 16, dtype=np.int64)
                goal_ids = np.zeros(len(envs), dtype=np.int64)
                goal_dists = np.zeros(len(envs), dtype=np.float32)
                for i, e in enumerate(envs):
                    ep_counter += 1
                    obs, inf = e.reset(seed=seed_base + ep_counter * 31 + i)
                    egomaps[i].reset()
                    f, _ = egomaps[i].step(obs, None)
                    feats_l.append(f)
                    priv_l.append(inf["priv_vec"])
                    if teachers is not None:
                        teachers[i].reset()
                        o_acts[i], _ = teachers[i].act(obs, e, assume_executed=False)
                        dir_bins[i] = dir_rel_to_bin(teachers[i].last_dir_rel_deg)
                        goal_ids[i] = GOAL_KIND_TO_ID.get(teachers[i].last_goal_kind, 0)
                        goal_dists[i] = min(teachers[i].last_goal_dist / 1024.0, 4.0)
                conn.send(_pack(
                    feats_l, priv_l, o_acts, dir_bins, goal_ids, goal_dists,
                    np.zeros(len(envs), dtype=np.float32),
                    np.zeros(len(envs), dtype=bool),
                    np.zeros(len(envs), dtype=bool),
                ))
            elif cmd == "step":
                acts_mb = payload  # (K, 9) int8
                feats_l, priv_l = [], []
                o_acts = np.zeros((len(envs), 6), dtype=np.int64)
                dir_bins = np.full(len(envs), 16, dtype=np.int64)
                goal_ids = np.zeros(len(envs), dtype=np.int64)
                goal_dists = np.zeros(len(envs), dtype=np.float32)
                rews = np.zeros(len(envs), dtype=np.float32)
                dones = np.zeros(len(envs), dtype=bool)
                exits = np.zeros(len(envs), dtype=bool)
                for i, e in enumerate(envs):
                    obs, rew, term, trunc, inf = e.step(acts_mb[i])
                    done = bool(term or trunc)
                    dones[i] = done
                    exits[i] = bool(inf.get("exit_reached", False))
                    if done:
                        rews[i] = float(rew)
                        ep_counter += 1
                        obs, inf = e.reset(seed=seed_base + ep_counter * 31 + i)
                        egomaps[i].reset()
                        f, _ = egomaps[i].step(obs, None)
                        if teachers is not None:
                            teachers[i].reset()
                    else:
                        f, new_cell = egomaps[i].step(obs, acts_mb[i])
                        rews[i] = float(rew) + (cell_bonus if new_cell else 0.0)
                        if teachers is not None:
                            teachers[i].record_action(acts_mb[i])
                    if teachers is not None:
                        o_acts[i], _ = teachers[i].act(obs, e, assume_executed=False)
                        dir_bins[i] = dir_rel_to_bin(teachers[i].last_dir_rel_deg)
                        goal_ids[i] = GOAL_KIND_TO_ID.get(teachers[i].last_goal_kind, 0)
                        goal_dists[i] = min(teachers[i].last_goal_dist / 1024.0, 4.0)
                    feats_l.append(f)
                    priv_l.append(inf["priv_vec"])
                conn.send(_pack(feats_l, priv_l, o_acts, dir_bins, goal_ids, goal_dists, rews, dones, exits))
            elif cmd == "close":
                break
    finally:
        for e in envs:
            e.close()
        conn.close()


class SubprocApexVecEnv:
    def __init__(
        self,
        wad_paths: list[str],
        num_envs: int,
        num_workers: int,
        episode_ticks: int,
        gamma: float,
        use_teacher: bool,
        seed: int = 1000,
        cell_bonus: float = 0.02,
    ) -> None:
        self.num_envs = num_envs
        self.num_workers = max(1, min(num_workers, num_envs))
        ctx = mp.get_context("spawn")
        self.conns = []
        self.procs = []
        base, rem = divmod(num_envs, self.num_workers)
        offset = 0
        for w in range(self.num_workers):
            k = base + (1 if w < rem else 0)
            w_wads = [wad_paths[(offset + j) % len(wad_paths)] for j in range(k)]
            parent_c, child_c = ctx.Pipe()
            p = ctx.Process(
                target=_worker_loop,
                args=(child_c, w_wads, episode_ticks, gamma, seed + offset * 17, use_teacher, cell_bonus),
                daemon=True,
            )
            p.start()
            child_c.close()
            self.conns.append(parent_c)
            self.procs.append(p)
            offset += k

    def _merge(self, parts: list[dict]) -> dict[str, np.ndarray]:
        return {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}

    def reset(self) -> dict[str, np.ndarray]:
        for c in self.conns:
            c.send(("reset", None))
        return self._merge([c.recv() for c in self.conns])

    def step(self, actions_mb: np.ndarray) -> dict[str, np.ndarray]:
        base, rem = divmod(self.num_envs, self.num_workers)
        offset = 0
        for w, c in enumerate(self.conns):
            k = base + (1 if w < rem else 0)
            c.send(("step", actions_mb[offset : offset + k]))
            offset += k
        return self._merge([c.recv() for c in self.conns])

    def close(self) -> None:
        for c in self.conns:
            try:
                c.send(("close", None))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()


def _sample_factored(logits: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    acts, logps, ents = [], [], []
    for lg in logits:
        lp = F.log_softmax(lg.float(), dim=-1)
        probs = lp.exp()
        a = torch.multinomial(probs, num_samples=1).squeeze(-1)
        acts.append(a)
        logps.append(lp.gather(-1, a.unsqueeze(-1)).squeeze(-1))
        ents.append(-(probs * lp).sum(dim=-1))
    return torch.stack(acts, dim=-1), sum(logps), sum(ents)


def _eval_factored(logits: list[torch.Tensor], acts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    logps, ents = [], []
    for i, lg in enumerate(logits):
        lp = F.log_softmax(lg.float(), dim=-1)
        probs = lp.exp()
        logps.append(lp.gather(-1, acts[..., i : i + 1]).squeeze(-1))
        ents.append(-(probs * lp).sum(dim=-1))
    return sum(logps), sum(ents)


def _teacher_ce(logits: list[torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
    weights = (1.0, 0.85, 1.15, 1.30, 1.75, 0.45)
    return sum(w * F.cross_entropy(lg.float(), targets[..., i]) for i, (w, lg) in enumerate(zip(weights, logits)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("dagger", "kickstart", "ppo"))
    ap.add_argument("--wad", nargs="+", required=True)
    ap.add_argument("--num-envs", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--episode-ticks", type=int, default=3200)
    ap.add_argument("--rollout-steps", type=int, default=64)
    ap.add_argument("--chunk-len", type=int, default=32)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--beta-start", type=float, default=0.5)
    ap.add_argument("--beta-end", type=float, default=0.0)
    ap.add_argument("--teacher-coef-start", type=float, default=0.8)
    ap.add_argument("--teacher-coef-end", type=float, default=0.05)
    ap.add_argument("--aux-coef", type=float, default=0.25)
    ap.add_argument("--total-steps", type=int, default=1_000_000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--save", default="training/checkpoints/apex.pt")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--reset-step", action="store_true")
    ap.add_argument("--export-submission", default=None)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else (args.device if args.device != "auto" else "cpu")
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    model = ApexNet(include_training_heads=True).to(device)
    model.detach_value = (args.mode == "dagger")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-5, weight_decay=1e-4)

    global_step = 0
    update_idx = 0
    if args.resume and Path(args.resume).is_file():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        if not args.reset_step and "opt" in ckpt and ckpt.get("mode") == args.mode:
            opt.load_state_dict(ckpt["opt"])
            global_step = int(ckpt.get("global_step", 0))
            update_idx = int(ckpt.get("update_idx", 0))
        print(f"[Apex] Resumed {args.resume} (mode={args.mode}, step={global_step:,})", flush=True)

    use_teacher = args.mode in ("dagger", "kickstart") or args.teacher_coef_end > 0.0
    vec = SubprocApexVecEnv(
        wad_paths=args.wad,
        num_envs=args.num_envs,
        num_workers=args.num_workers,
        episode_ticks=args.episode_ticks,
        gamma=args.gamma,
        use_teacher=use_teacher,
        seed=args.seed,
    )

    N, T, C = args.num_envs, args.rollout_steps, args.chunk_len
    assert T % C == 0
    n_chunks = T // C
    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    export_path = Path(args.export_submission) if args.export_submission else None
    if export_path is not None:
        export_path.parent.mkdir(parents=True, exist_ok=True)

    pkt = vec.reset()
    h_cur, L_cur = model.init_state(N, device)
    ep_ret = np.zeros(N, dtype=np.float32)
    recent_rets: list[float] = []
    recent_exits: list[float] = []
    t0 = time.perf_counter()

    def _pkt_to_tensors(p: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        return {
            "vec": torch.from_numpy(p["vec"]).to(device, non_blocking=True),
            "rays": torch.from_numpy(p["rays"]).to(device, non_blocking=True),
            "ents": torch.from_numpy(p["ents"]).to(device, non_blocking=True),
            "etype": torch.from_numpy(p["etype"]).to(device, non_blocking=True),
            "fine_map": torch.from_numpy(p["fine_map"]).to(device, non_blocking=True),
            "wide_map": torch.from_numpy(p["wide_map"]).to(device, non_blocking=True),
            "crop_idx": torch.from_numpy(p["crop_idx"]).to(device, non_blocking=True),
            "pos_idx": torch.from_numpy(p["pos_idx"]).to(device, non_blocking=True),
            "wmask": torch.from_numpy(p["wmask"]).to(device, non_blocking=True),
        }

    try:
        while global_step < args.total_steps:
            frac = min(1.0, global_step / max(1, args.total_steps))
            beta = args.beta_start + frac * (args.beta_end - args.beta_start) if args.mode == "dagger" else 0.0
            t_coef = (
                1.0 if args.mode == "dagger"
                else args.teacher_coef_start + frac * (args.teacher_coef_end - args.teacher_coef_start)
            )
            lr_now = args.lr * (0.2 + 0.8 * (1.0 - frac))
            for pg in opt.param_groups:
                pg["lr"] = lr_now

            # Rollout storage
            buf_vec = torch.zeros((T, N, VEC_DIM), device=device)
            buf_rays = torch.zeros((T, N, 8, 64), device=device)
            buf_ents = torch.zeros((T, N, 32, 6), device=device)
            buf_etype = torch.zeros((T, N, 32), dtype=torch.int64, device=device)
            buf_fine = torch.zeros((T, N, N_FIXED_CH, FINE_CROP, FINE_CROP), dtype=torch.float16, device=device)
            buf_wide = torch.zeros((T, N, N_FIXED_CH, WIDE_CROP, WIDE_CROP), dtype=torch.float16, device=device)
            buf_cidx = torch.zeros((T, N, FINE_CROP * FINE_CROP), dtype=torch.int64, device=device)
            buf_pidx = torch.zeros((T, N), dtype=torch.int64, device=device)
            buf_wmask = torch.zeros((T, N), device=device)
            buf_priv = torch.zeros((T, N, 24), device=device)

            buf_h = torch.zeros((n_chunks, N, model.hid), device=device)
            buf_L = torch.zeros((n_chunks, N, L_cur.shape[1], L_cur.shape[2]), device=device)
            buf_act = torch.zeros((T, N, 6), dtype=torch.int64, device=device)
            buf_logp = torch.zeros((T, N), device=device)
            buf_val = torch.zeros((T, N), device=device)
            buf_rew = torch.zeros((T, N), device=device)
            buf_done = torch.zeros((T, N), device=device)
            buf_orc = torch.zeros((T, N, 6), dtype=torch.int64, device=device)
            buf_dir = torch.zeros((T, N), dtype=torch.int64, device=device)
            buf_goal = torch.zeros((T, N), dtype=torch.int64, device=device)
            buf_dist = torch.zeros((T, N), device=device)

            model.eval()
            with torch.inference_mode():
                for t in range(T):
                    if t % C == 0:
                        buf_h[t // C].copy_(h_cur)
                        buf_L[t // C].copy_(L_cur)
                    obs_t = _pkt_to_tensors(pkt)
                    priv_t = torch.from_numpy(pkt["priv_vec"]).to(device, non_blocking=True)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        logits, (h_nxt, L_nxt), extras = model.forward_step(obs_t, (h_cur, L_cur), priv_vec=priv_t)
                    act_f, logp, _ = _sample_factored(logits)

                    buf_vec[t] = obs_t["vec"]
                    buf_rays[t] = obs_t["rays"]
                    buf_ents[t] = obs_t["ents"]
                    buf_etype[t] = obs_t["etype"]
                    buf_fine[t] = obs_t["fine_map"]
                    buf_wide[t] = obs_t["wide_map"]
                    buf_cidx[t] = obs_t["crop_idx"]
                    buf_pidx[t] = obs_t["pos_idx"]
                    buf_wmask[t] = obs_t["wmask"]
                    buf_priv[t] = priv_t
                    buf_act[t] = act_f
                    buf_logp[t] = logp
                    buf_val[t] = extras["value"].float()
                    buf_orc[t] = torch.from_numpy(pkt["oracle_act"]).to(device, non_blocking=True)
                    buf_dir[t] = torch.from_numpy(pkt["dir_bin"]).to(device, non_blocking=True)
                    buf_goal[t] = torch.from_numpy(pkt["goal_id"]).to(device, non_blocking=True)
                    buf_dist[t] = torch.from_numpy(pkt["goal_dist"]).to(device, non_blocking=True)

                    exec_mb = apex_factored_to_multibinary(act_f).cpu().numpy()
                    if beta > 0.0:
                        use_orc = rng.random(N) < beta
                        for i in np.flatnonzero(use_orc):
                            exec_mb[i] = factored_np_to_multibinary(pkt["oracle_act"][i])

                    pkt = vec.step(exec_mb)
                    rew_np, done_np, exit_np = pkt["reward"], pkt["done"], pkt["exit"]
                    buf_rew[t] = torch.from_numpy(rew_np).to(device, non_blocking=True)
                    buf_done[t] = torch.from_numpy(done_np.astype(np.float32)).to(device, non_blocking=True)
                    ep_ret += rew_np
                    for i in range(N):
                        if done_np[i]:
                            recent_rets.append(float(ep_ret[i]))
                            recent_exits.append(float(exit_np[i]))
                            ep_ret[i] = 0.0
                    keep = torch.from_numpy((~done_np).astype(np.float32)).to(device)
                    h_cur = h_nxt * keep.unsqueeze(-1)
                    L_cur = L_nxt * keep.view(N, 1, 1)

                # Bootstrap value
                obs_last = _pkt_to_tensors(pkt)
                priv_last = torch.from_numpy(pkt["priv_vec"]).to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    _, _, ex_last = model.forward_step(obs_last, (h_cur, L_cur), priv_vec=priv_last)
                next_val = ex_last["value"].float()

            # GAE
            buf_adv = torch.zeros((T, N), device=device)
            gae = torch.zeros(N, device=device)
            for t in reversed(range(T)):
                nt = 1.0 - buf_done[t]
                nv = next_val if t == T - 1 else buf_val[t + 1]
                delta = buf_rew[t] + args.gamma * nv * nt - buf_val[t]
                gae = delta + args.gamma * args.gae_lambda * nt * gae
                buf_adv[t] = gae
            buf_ret = buf_adv + buf_val
            buf_adv = (buf_adv - buf_adv.mean()) / (buf_adv.std() + 1e-8)

            def _chunk(x: torch.Tensor) -> torch.Tensor:
                return x.view(n_chunks, C, N, *x.shape[2:]).permute(1, 0, 2, *range(3, x.ndim + 1)).reshape(C, n_chunks * N, *x.shape[2:])

            vec_c, rays_c, ents_c, etype_c = _chunk(buf_vec), _chunk(buf_rays), _chunk(buf_ents), _chunk(buf_etype)
            fine_c, wide_c, cidx_c, pidx_c = _chunk(buf_fine), _chunk(buf_wide), _chunk(buf_cidx), _chunk(buf_pidx)
            wmask_c, priv_c = _chunk(buf_wmask), _chunk(buf_priv)
            act_c, logp_c, ret_c, adv_c = _chunk(buf_act), _chunk(buf_logp), _chunk(buf_ret), _chunk(buf_adv)
            done_c, orc_c, dir_c, goal_c, dist_c = _chunk(buf_done), _chunk(buf_orc), _chunk(buf_dir), _chunk(buf_goal), _chunk(buf_dist)
            h0_c = buf_h.reshape(n_chunks * N, model.hid)
            L0_c = buf_L.reshape(n_chunks * N, L_cur.shape[1], L_cur.shape[2])

            total_chunks = n_chunks * N
            mb_size = max(1, total_chunks // args.minibatches)

            model.train()
            last_loss = 0.0
            for _ in range(args.epochs):
                perm = torch.randperm(total_chunks, device=device)
                for start in range(0, total_chunks, mb_size):
                    idx = perm[start : start + mb_size]
                    hc, Lc = h0_c[idx], L0_c[idx]
                    opt.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        logits_seq = [[] for _ in HEADS]
                        val_seq, dir_seq, goal_seq, dist_seq = [], [], [], []
                        for s in range(C):
                            obs_s = {
                                "vec": vec_c[s, idx], "rays": rays_c[s, idx],
                                "ents": ents_c[s, idx], "etype": etype_c[s, idx],
                                "fine_map": fine_c[s, idx], "wide_map": wide_c[s, idx],
                                "crop_idx": cidx_c[s, idx], "pos_idx": pidx_c[s, idx],
                                "wmask": wmask_c[s, idx],
                            }
                            lgs, (hc, Lc), ex = model.forward_step(obs_s, (hc, Lc), priv_vec=priv_c[s, idx])
                            for k, lg in enumerate(lgs):
                                logits_seq[k].append(lg)
                            val_seq.append(ex["value"])
                            dir_seq.append(ex["dir_logits"])
                            goal_seq.append(ex["goal_logits"])
                            dist_seq.append(ex["dist_pred"])
                            keep = 1.0 - done_c[s, idx]
                            hc = hc * keep.unsqueeze(-1)
                            Lc = Lc * keep.view(-1, 1, 1)

                        s_logits = [torch.stack(ls, dim=0) for ls in logits_seq]
                        s_val = torch.stack(val_seq, dim=0)
                        s_dir = torch.stack(dir_seq, dim=0)
                        s_goal = torch.stack(goal_seq, dim=0)
                        s_dist = torch.stack(dist_seq, dim=0)

                        l_val = F.smooth_l1_loss(s_val.float(), ret_c[:, idx].float(), beta=2.0)
                        l_aux = (
                            F.cross_entropy(s_dir.float().reshape(-1, s_dir.shape[-1]), dir_c[:, idx].reshape(-1))
                            + 0.5 * F.cross_entropy(s_goal.float().reshape(-1, s_goal.shape[-1]), goal_c[:, idx].reshape(-1))
                            + 0.25 * F.smooth_l1_loss(s_dist.float(), dist_c[:, idx].float())
                        )

                        if args.mode == "dagger":
                            l_ce = _teacher_ce([lg.reshape(-1, lg.shape[-1]) for lg in s_logits], orc_c[:, idx].reshape(-1, 6))
                            loss = l_ce + 0.25 * l_val + args.aux_coef * l_aux
                        else:
                            new_lp, ent = _eval_factored(s_logits, act_c[:, idx])
                            ratio = (new_lp - logp_c[:, idx]).clamp(-10.0, 10.0).exp()
                            adv_mb = adv_c[:, idx]
                            surr1 = ratio * adv_mb
                            surr2 = ratio.clamp(1.0 - args.clip_eps, 1.0 + args.clip_eps) * adv_mb
                            l_ppo = -torch.min(surr1, surr2).mean()
                            loss = l_ppo + 0.5 * l_val - args.ent_coef * ent.mean() + args.aux_coef * l_aux
                            if use_teacher and t_coef > 0.0:
                                l_ce = _teacher_ce([lg.reshape(-1, lg.shape[-1]) for lg in s_logits], orc_c[:, idx].reshape(-1, 6))
                                loss = loss + t_coef * l_ce

                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    scaler.step(opt)
                    scaler.update()
                    last_loss = float(loss.item())

            global_step += N * T
            update_idx += 1
            sps = global_step / max(1e-3, time.perf_counter() - t0)
            avg_r = float(np.mean(recent_rets[-40:])) if recent_rets else 0.0
            avg_ex = float(np.mean(recent_exits[-40:]) * 100.0) if recent_exits else 0.0
            print(
                f"[{args.mode:9s}] step={global_step:8,d} | sps={sps:6.0f} | loss={last_loss:.3f} | "
                f"ret={avg_r:+6.2f} | exit={avg_ex:5.1f}% | beta={beta:.2f} | t_coef={t_coef:.2f}",
                flush=True,
            )

            if update_idx % args.save_every == 0 or global_step >= args.total_steps:
                tmp = save_path.with_suffix(".tmp")
                torch.save(
                    {
                        "arch": "apex",
                        "mode": args.mode,
                        "global_step": global_step,
                        "update_idx": update_idx,
                        "model": model.state_dict(),
                        "opt": opt.state_dict(),
                    },
                    tmp,
                )
                tmp.replace(save_path)
                if export_path is not None:
                    etmp = export_path.with_suffix(".tmp")
                    torch.save({"arch": "apex", "model": model.strip_for_submission()}, etmp)
                    etmp.replace(export_path)
    finally:
        vec.close()


if __name__ == "__main__":
    main()
