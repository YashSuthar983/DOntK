"""Phase 2: Fast Vectorized Recurrent PPO Trainer for OmniscientTeacher (render_buffers=False).

Runs efficiently on both multi-core CPU (e.g. Ryzen 5 5600H) and CUDA GPU (Colab).
Saves atomic resumable checkpoints containing model + optimizer + step count.

Example:
    PYTHONPATH=. .venv/bin/python training/train_teacher.py \
        --wad training/wads/train/tier1_train.wad \
        --num-envs 8 --rollout-steps 128 --total-steps 500000 \
        --save training/checkpoints/teacher.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_ROOT / "submission")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from aegis3d import factored_to_multibinary
from training.teacher import OmniscientTeacher
from training.wad_oracle import FastDoomPackEnv


def _obs_batch_to_torch(obs_list: list[dict], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "vitals": torch.from_numpy(np.stack([o["vitals"] for o in obs_list])).to(device),
        "ray_distance": torch.from_numpy(np.stack([o["rays"]["distance"] for o in obs_list])).to(device),
        "ray_hit_type": torch.from_numpy(np.stack([o["rays"]["hit_type"] for o in obs_list])).to(device),
        "ent_features": torch.from_numpy(np.stack([o["visible_entities"]["features"] for o in obs_list])).to(device),
        "ent_type": torch.from_numpy(np.stack([o["visible_entities"]["entity_type"] for o in obs_list])).to(device),
        "ent_mask": torch.from_numpy(np.stack([o["visible_entities"]["mask"] for o in obs_list])).to(device),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wad", nargs="+", required=True, help="One or more packed .wad shards")
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--episode-ticks", type=int, default=2100)
    ap.add_argument("--rollout-steps", type=int, default=128)
    ap.add_argument("--total-steps", type=int, default=200_000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip-eps", type=float, default=0.18)
    ap.add_argument("--ent-coef", type=float, default=0.004)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--chunk-len", type=int, default=32)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--target-kl", type=float, default=0.03)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--save", default="training/checkpoints/teacher.pt")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--save-every", type=int, default=10)
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else (args.device if args.device != "auto" else "cpu"))
    print(f"[Teacher Phase 2] Device={device} | Envs={args.num_envs} | WADs={args.wad}")

    envs = [
        FastDoomPackEnv(
            args.wad[i % len(args.wad)],
            episode_ticks=args.episode_ticks,
            render_buffers=False,
            gamma=args.gamma,
            compute_spatial_targets=False,
        )
        for i in range(args.num_envs)
    ]

    model = OmniscientTeacher().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    global_step = 0
    update_idx = 0

    if args.resume and Path(args.resume).is_file():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        global_step = int(ckpt.get("global_step", 0))
        update_idx = int(ckpt.get("update_idx", 0))
        print(f"[Teacher] Resumed from {args.resume} at step {global_step:,}")

    obs_list, priv_list = [], []
    for i, e in enumerate(envs):
        o, inf = e.reset(seed=100 + i)
        obs_list.append(o)
        priv_list.append(inf["priv_vec"])

    h = model.init_hidden(args.num_envs, device)
    prev_act = torch.zeros((args.num_envs, 9), dtype=torch.float32, device=device)

    T = args.rollout_steps
    N = args.num_envs
    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    ep_exits: list[float] = []
    ep_returns = np.zeros(N, dtype=np.float64)
    recent_returns: list[float] = []
    t_start = time.perf_counter()

    try:
        while global_step < args.total_steps:
            update_idx += 1
            # Rollout storage
            buf_obs: list[dict[str, torch.Tensor]] = []
            buf_priv = torch.zeros((T, N, 24), device=device)
            buf_prev_act = torch.zeros((T, N, 9), device=device)
            buf_act = torch.zeros((T, N, 6), dtype=torch.int64, device=device)
            buf_logp = torch.zeros((T, N), device=device)
            buf_val = torch.zeros((T, N), device=device)
            buf_rew = torch.zeros((T, N), device=device)
            buf_done = torch.zeros((T, N), device=device)
            buf_h = torch.zeros((T, N, model.hidden_dim), device=device)

            model.eval()
            with torch.no_grad():
                for t in range(T):
                    buf_h[t] = h
                    obs_t = _obs_batch_to_torch(obs_list, device)
                    priv_t = torch.from_numpy(np.stack(priv_list)).to(device)
                    logits, val, next_h = model.forward_step(obs_t, priv_t, prev_act, h)

                    dists = [Categorical(logits=lg) for lg in logits]
                    acts_f = torch.stack([d.sample() for d in dists], dim=-1)  # (N, 6)
                    logp = torch.stack([d.log_prob(acts_f[:, i]) for i, d in enumerate(dists)], dim=-1).sum(dim=-1)
                    acts_mb = factored_to_multibinary(acts_f)  # (N, 9) int8

                    buf_obs.append(obs_t)
                    buf_priv[t] = priv_t
                    buf_prev_act[t] = prev_act
                    buf_act[t] = acts_f
                    buf_logp[t] = logp
                    buf_val[t] = val

                    acts_np = acts_mb.cpu().numpy()
                    next_obs_list, next_priv_list = [], []
                    dones = np.zeros(N, dtype=np.float32)
                    rews = np.zeros(N, dtype=np.float32)
                    for i, e in enumerate(envs):
                        o, r, term, trunc, inf = e.step(acts_np[i])
                        rews[i] = r
                        ep_returns[i] += r
                        done = term or trunc
                        if done:
                            dones[i] = 1.0
                            ep_exits.append(1.0 if inf.get("exit_reached") else 0.0)
                            recent_returns.append(float(ep_returns[i]))
                            ep_returns[i] = 0.0
                            o, inf = e.reset()
                        next_obs_list.append(o)
                        next_priv_list.append(inf["priv_vec"])

                    buf_rew[t] = torch.from_numpy(rews).to(device)
                    done_t = torch.from_numpy(dones).to(device)
                    buf_done[t] = done_t
                    # Zero hidden & prev_action on episode boundaries
                    keep = (1.0 - done_t).unsqueeze(-1)
                    h = next_h * keep
                    prev_act = acts_mb.to(torch.float32) * keep
                    obs_list, priv_list = next_obs_list, next_priv_list
                    global_step += N

                # Bootstrap value
                last_obs = _obs_batch_to_torch(obs_list, device)
                last_priv = torch.from_numpy(np.stack(priv_list)).to(device)
                _, next_val, _ = model.forward_step(last_obs, last_priv, prev_act, h)

            # Compute GAE advantages
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

            # Chunked minibatched Recurrent PPO update (5x more gradient updates per rollout!)
            model.train()
            L = min(args.chunk_len, T)
            assert T % L == 0, f"rollout_steps ({T}) must be divisible by chunk_len ({L})"
            n_chunks_t = T // L
            total_chunks = n_chunks_t * N  # treat each (time_chunk, env) as an independent sequence of length L
            obs_seq = {
                k: torch.stack([step_o[k] for step_o in buf_obs], dim=0)
                .view(n_chunks_t, L, N, *buf_obs[0][k].shape[1:])
                .permute(1, 0, 2, *range(3, 3 + len(buf_obs[0][k].shape[1:])))
                .reshape(L, total_chunks, *buf_obs[0][k].shape[1:])
                for k in buf_obs[0]
            }
            priv_c = buf_priv.view(n_chunks_t, L, N, -1).permute(1, 0, 2, 3).reshape(L, total_chunks, -1)
            pa_c = buf_prev_act.view(n_chunks_t, L, N, -1).permute(1, 0, 2, 3).reshape(L, total_chunks, -1)
            act_c = buf_act.view(n_chunks_t, L, N, -1).permute(1, 0, 2, 3).reshape(L, total_chunks, -1)
            logp_c = buf_logp.view(n_chunks_t, L, N).permute(1, 0, 2).reshape(L, total_chunks)
            adv_c = adv.view(n_chunks_t, L, N).permute(1, 0, 2).reshape(L, total_chunks)
            ret_c = ret.view(n_chunks_t, L, N).permute(1, 0, 2).reshape(L, total_chunks)
            done_c = buf_done.view(n_chunks_t, L, N).permute(1, 0, 2).reshape(L, total_chunks)
            h0_c = buf_h[::L].reshape(total_chunks, model.hidden_dim)

            mb_size = max(1, total_chunks // args.minibatches)
            p_loss_v, v_loss_v, ent_v = 0.0, 0.0, 0.0
            stop_early = False
            for _ in range(args.epochs):
                perm = torch.randperm(total_chunks, device=device)
                for start in range(0, total_chunks, mb_size):
                    idx = perm[start : start + mb_size]
                    mb_obs = {k: v[:, idx] for k, v in obs_seq.items()}
                    logits_seq, val_new = model.forward_sequence(mb_obs, priv_c[:, idx], pa_c[:, idx], done_c[:, idx], h0_c[idx])
                    dists = [Categorical(logits=lg) for lg in logits_seq]
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
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = F.mse_loss(val_new, ret_c[:, idx])
                    loss = policy_loss + 0.5 * value_loss - args.ent_coef * ent_mean

                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    opt.step()
                    p_loss_v, v_loss_v, ent_v = float(policy_loss.item()), float(value_loss.item()), float(ent_mean.item())
                if stop_early:
                    break

            fps = global_step / max(1e-3, time.perf_counter() - t_start)
            ex_rate = float(np.mean(ep_exits[-50:])) if ep_exits else 0.0
            mean_ret = float(np.mean(recent_returns[-50:])) if recent_returns else 0.0
            print(
                f"upd={update_idx:04d} step={global_step:8d} fps={fps:6.0f} "
                f"ret={mean_ret:6.2f} exit={ex_rate*100:5.1f}% "
                f"pl={p_loss_v:+.3f} vl={v_loss_v:.3f} ent={ent_v:.2f}",
                flush=True,
            )

            if update_idx % args.save_every == 0 or global_step >= args.total_steps:
                tmp_path = save_path.with_suffix(".tmp")
                torch.save(
                    {"model": model.state_dict(), "opt": opt.state_dict(), "global_step": global_step, "update_idx": update_idx},
                    tmp_path,
                )
                tmp_path.replace(save_path)
                print(f"[Teacher] Saved checkpoint -> {save_path}", flush=True)
    finally:
        for e in envs:
            e.close()


if __name__ == "__main__":
    main()
