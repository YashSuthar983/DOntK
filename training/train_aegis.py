"""Unified Phases 1, 3, and 4 Trainer for AEGIS-3D (Colab GPU + Local CPU).

Supports:
- Phase 1 (`--phase 1`): Supervised Sensor & Memory Bootstrapping
    L_ground = L_odom + L_BEV + L_afford + L_E3B_IDM
- Phase 3 (`--phase 3`): Online Student DAgger KL Distillation from Phase-2 OmniscientTeacher
    L = KL(pi_teacher || pi_student) + 0.5 * L_ground
- Phase 4 (`--phase 4`): Asymmetric Recurrent PPO + Annealed Teacher KL + E3B Curiosity + Grounding
    L = L_PPO + 0.5 * L_critic_asym - c_H * H + beta * KL + 0.25 * L_ground

Includes:
- Scheduled 15% front-ray modality dropout to prevent CNN collapse
- Atomic resumable checkpointing (Colab Drive resilient)
- Clean student-only weight export (`--export-submission submission/weights/aegis3d.pt`)
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

from aegis3d import AEGIS3D, AgentMemoryState, factored_to_multibinary
from training.checkpoint import restore_optimizer
from training.teacher import OmniscientTeacher
from training.wad_oracle import FastDoomPackEnv


def _obs_batch_to_torch(obs_list: list[dict], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "rgb": torch.from_numpy(np.stack([o["rgb"] for o in obs_list])).to(device, non_blocking=True),
        "depth": torch.from_numpy(np.stack([o["depth"] for o in obs_list])).to(device, non_blocking=True),
        "vitals": torch.from_numpy(np.stack([o["vitals"] for o in obs_list])).to(device, non_blocking=True),
        "ray_distance": torch.from_numpy(np.stack([o["rays"]["distance"] for o in obs_list])).to(device, non_blocking=True),
        "ray_hit_type": torch.from_numpy(np.stack([o["rays"]["hit_type"] for o in obs_list])).to(device, non_blocking=True),
        "ent_features": torch.from_numpy(np.stack([o["visible_entities"]["features"] for o in obs_list])).to(device, non_blocking=True),
        "ent_type": torch.from_numpy(np.stack([o["visible_entities"]["entity_type"] for o in obs_list])).to(device, non_blocking=True),
        "ent_mask": torch.from_numpy(np.stack([o["visible_entities"]["mask"] for o in obs_list])).to(device, non_blocking=True),
    }


def _reset_masked_state(state: AgentMemoryState, done_mask: torch.Tensor, fresh: AgentMemoryState) -> AgentMemoryState:
    """Reset per-env memory tensors wherever done_mask[i] > 0.5."""
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
        kf_feat=torch.where(m3, fresh.kf_feat, state.kf_feat),
        kf_mask=torch.where(m2, fresh.kf_mask, state.kf_mask),
        kf_ptr=torch.where(m, fresh.kf_ptr, state.kf_ptr),
        e3b_cov_inv=torch.where(m3, fresh.e3b_cov_inv, state.e3b_cov_inv),
        ray_hist=torch.where(m3, fresh.ray_hist, state.ray_hist),
        ray_type_hist=torch.where(m3, fresh.ray_type_hist, state.ray_type_hist),
        prev_actions=torch.where(m3, fresh.prev_actions, state.prev_actions),
        prev_vitals=torch.where(m2, fresh.prev_vitals, state.prev_vitals),
        wpn_cooldown=torch.where(m, fresh.wpn_cooldown, state.wpn_cooldown),
    )


def _kl_factored(teacher_logits: list[torch.Tensor], student_logits: list[torch.Tensor]) -> torch.Tensor:
    total = torch.zeros((), device=student_logits[0].device)
    for tl, sl in zip(teacher_logits, student_logits):
        valid = (sl > -1e6) & (tl.detach() > -1e6)
        tl_m = torch.where(valid, tl.detach(), torch.full_like(tl, -1e9))
        sl_m = torch.where(valid, sl, torch.full_like(sl, -1e9))
        tp = torch.softmax(tl_m, dim=-1)
        log_tp = torch.log_softmax(tl_m, dim=-1)
        log_sp = torch.log_softmax(sl_m, dim=-1)
        kl_elem = torch.where(valid, tp * (log_tp - log_sp), torch.zeros_like(tp))
        total = total + kl_elem.sum(dim=-1).mean()
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, required=True, choices=(1, 3, 4))
    ap.add_argument("--wad", nargs="+", required=True)
    ap.add_argument("--teacher-ckpt", default=None, help="Required for Phase 3; optional KL anchor for Phase 4")
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--episode-ticks", type=int, default=2100)
    ap.add_argument("--rollout-steps", type=int, default=64)
    ap.add_argument("--total-steps", type=int, default=200_000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip-eps", type=float, default=0.15)
    ap.add_argument("--ent-coef", type=float, default=0.008)
    ap.add_argument("--e3b-coef", type=float, default=0.02)
    ap.add_argument("--kl-coef", type=float, default=0.15)
    ap.add_argument("--teacher-mix", type=float, default=0.50, help="Phase 3 initial teacher rollout mixing prob")
    ap.add_argument("--ray-dropout", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--target-kl", type=float, default=0.03)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--save", default="training/checkpoints/aegis3d_train.pt")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--export-submission", default="submission/weights/aegis3d.pt")
    ap.add_argument("--save-every", type=int, default=5)
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else (args.device if args.device != "auto" else "cpu"))
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))
    print(f"[AEGIS-3D Phase {args.phase}] Device={device} | AMP={use_amp} | Envs={args.num_envs}")

    envs = [
        FastDoomPackEnv(args.wad[i % len(args.wad)], episode_ticks=args.episode_ticks, render_buffers=True, gamma=args.gamma)
        for i in range(args.num_envs)
    ]

    model = AEGIS3D(include_training_heads=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    global_step = 0
    update_idx = 0

    if args.resume and Path(args.resume).is_file():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        if "opt" in ckpt and ckpt.get("phase") == args.phase:
            restore_optimizer(opt, model, ckpt)
            global_step = int(ckpt.get("global_step", 0))
            update_idx = int(ckpt.get("update_idx", 0))
        print(f"[AEGIS-3D] Loaded weights from {args.resume} (phase={ckpt.get('phase')}, step={global_step:,})")

    teacher: OmniscientTeacher | None = None
    if args.teacher_ckpt and Path(args.teacher_ckpt).is_file():
        teacher = OmniscientTeacher().to(device)
        tckpt = torch.load(args.teacher_ckpt, map_location=device, weights_only=False)
        teacher.load_state_dict(tckpt["model"])
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        print(f"[AEGIS-3D] Loaded frozen OmniscientTeacher <- {args.teacher_ckpt}")
    elif args.phase == 3:
        raise ValueError("Phase 3 (DAgger Distillation) requires --teacher-ckpt")

    obs_list, info_list = [], []
    for i, e in enumerate(envs):
        o, inf = e.reset(seed=200 + i)
        obs_list.append(o)
        info_list.append(inf)

    N = args.num_envs
    T = args.rollout_steps
    state = model.init_state(N, device)
    fresh_template = model.init_state(N, device)
    teacher_h = teacher.init_hidden(N, device) if teacher is not None else None
    teacher_prev_act = torch.zeros((N, 9), device=device)

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    export_path = Path(args.export_submission) if args.export_submission else None
    if export_path is not None:
        export_path.parent.mkdir(parents=True, exist_ok=True)

    ep_exits: list[float] = []
    ep_returns = np.zeros(N, dtype=np.float64)
    recent_returns: list[float] = []
    t_start = time.perf_counter()

    try:
        while global_step < args.total_steps:
            update_idx += 1
            init_state = state.detach_()
            buf_obs: list[dict[str, torch.Tensor]] = []
            buf_priv = torch.zeros((T, N, 24), device=device)
            buf_odom = torch.zeros((T, N, 4), device=device)
            buf_bev = torch.zeros((T, N, 4, 64, 64), device=device)
            buf_aff = torch.zeros((T, N, 20, 5), device=device)
            buf_act_f = torch.zeros((T, N, 6), dtype=torch.int64, device=device)
            buf_act_mb = torch.zeros((T, N, 9), dtype=torch.int8, device=device)
            buf_logp = torch.zeros((T, N), device=device)
            buf_val = torch.zeros((T, N), device=device)
            buf_rew = torch.zeros((T, N), device=device)
            buf_done = torch.zeros((T, N), device=device)
            buf_t_logits: list[list[torch.Tensor]] = []
            buf_t_val = torch.zeros((T, N), device=device)

            progress = min(1.0, global_step / max(1, args.total_steps))
            mix_p = args.teacher_mix * max(0.05, 1.0 - progress * 1.1) if args.phase == 3 else 0.0

            model.eval()
            with torch.no_grad():
                for t in range(T):
                    obs_t = _obs_batch_to_torch(obs_list, device)
                    priv_t = torch.from_numpy(np.stack([inf["priv_vec"] for inf in info_list])).to(device)
                    s_logits, next_state, extras = model.forward_step(obs_t, state)
                    val_t = model.critic(torch.cat([extras["gru_h"], priv_t], dim=-1)).squeeze(-1)

                    t_logits_step: list[torch.Tensor] | None = None
                    if teacher is not None and teacher_h is not None:
                        t_logits_step, t_val_step, teacher_h = teacher.forward_step(obs_t, priv_t, teacher_prev_act, teacher_h)
                        buf_t_logits.append(t_logits_step)
                        buf_t_val[t] = t_val_step.detach()

                    s_dists = [Categorical(logits=lg) for lg in s_logits]
                    if t_logits_step is not None and np.random.random() < mix_p:
                        t_dists = [Categorical(logits=lg) for lg in t_logits_step]
                        acts_f = torch.stack([d.sample() for d in t_dists], dim=-1)
                    else:
                        acts_f = torch.stack([d.sample() for d in s_dists], dim=-1)

                    logp = torch.stack([d.log_prob(acts_f[:, i]) for i, d in enumerate(s_dists)], dim=-1).sum(dim=-1)
                    acts_mb = factored_to_multibinary(acts_f)
                    next_state = model.update_state_with_action(next_state, acts_mb)

                    buf_obs.append(obs_t)
                    buf_priv[t] = priv_t
                    buf_act_f[t] = acts_f
                    buf_act_mb[t] = acts_mb
                    buf_logp[t] = logp
                    buf_val[t] = val_t

                    acts_np = acts_mb.cpu().numpy()
                    next_obs, next_info = [], []
                    dones = np.zeros(N, dtype=np.float32)
                    rews = np.zeros(N, dtype=np.float32)
                    for i, e in enumerate(envs):
                        o, r, term, trunc, inf = e.step(acts_np[i])
                        # Add live E3B episodic novelty bonus during Phase 4
                        r_tot = r + (args.e3b_coef * float(extras["e3b_bonus"][i, 0].item()) if args.phase == 4 else 0.0)
                        rews[i] = r_tot
                        ep_returns[i] += r_tot
                        buf_odom[t, i] = torch.from_numpy(inf["odom_target"]).to(device)
                        buf_bev[t, i] = torch.from_numpy(inf["bev_target"]).to(device)
                        buf_aff[t, i] = torch.from_numpy(inf["afford_target"]).to(device)
                        if term or trunc:
                            dones[i] = 1.0
                            ep_exits.append(1.0 if inf.get("exit_reached") else 0.0)
                            recent_returns.append(float(ep_returns[i]))
                            ep_returns[i] = 0.0
                            o, inf = e.reset()
                        next_obs.append(o)
                        next_info.append(inf)

                    done_t = torch.from_numpy(dones).to(device)
                    buf_rew[t] = torch.from_numpy(rews).to(device)
                    buf_done[t] = done_t
                    state = _reset_masked_state(next_state, done_t, fresh_template)
                    if teacher_h is not None:
                        keep = (1.0 - done_t).unsqueeze(-1)
                        teacher_h = teacher_h * keep
                        teacher_prev_act = acts_mb.to(torch.float32) * keep
                    obs_list, info_list = next_obs, next_info
                    global_step += N

                # Bootstrap value for Phase 4 GAE
                last_obs = _obs_batch_to_torch(obs_list, device)
                last_priv = torch.from_numpy(np.stack([inf["priv_vec"] for inf in info_list])).to(device)
                _, _, last_ext = model.forward_step(last_obs, state)
                next_val = model.critic(torch.cat([last_ext["gru_h"], last_priv], dim=-1)).squeeze(-1)

            # GAE for Phase 4
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

            # Training step over rollout chunk
            model.train()
            kl_beta = args.kl_coef * (1.0 - progress) if args.phase == 4 else 1.0
            metrics = {"loss": 0.0, "ground": 0.0, "kl": 0.0, "ppo": 0.0}

            for _ in range(args.epochs):
                cur_state = init_state
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    l_odom_acc = torch.zeros((), device=device)
                    l_bev_acc = torch.zeros((), device=device)
                    l_aff_acc = torch.zeros((), device=device)
                    l_idm_acc = torch.zeros((), device=device)
                    l_kl_acc = torch.zeros((), device=device)
                    l_vdist_acc = torch.zeros((), device=device)
                    logp_new_list, val_new_list, ent_list = [], [], []
                    prev_e3b_emb: torch.Tensor | None = None

                    for t in range(T):
                        s_logits, nxt_s, ext = model.forward_step(buf_obs[t], cur_state, ray_dropout_p=args.ray_dropout)
                        # Grounding losses (active in ALL phases!)
                        l_odom_acc = l_odom_acc + F.smooth_l1_loss(ext["odom"], buf_odom[t])
                        l_bev_acc = l_bev_acc + F.binary_cross_entropy_with_logits(model.aux_bev(ext["bev_map"]), buf_bev[t])
                        l_aff_acc = l_aff_acc + F.binary_cross_entropy_with_logits(model.aux_afford(ext["horizon_tokens"]), buf_aff[t])
                        if prev_e3b_emb is not None and t > 0:
                            idm_in = torch.cat([prev_e3b_emb, ext["e3b_emb"]], dim=-1)
                            idm_step = sum(F.cross_entropy(head(idm_in), buf_act_f[t - 1, :, i]) for i, head in enumerate(model.e3b_idm))
                            l_idm_acc = l_idm_acc + idm_step / len(model.e3b_idm)
                        prev_e3b_emb = ext["e3b_emb"]

                        if buf_t_logits:
                            l_kl_acc = l_kl_acc + _kl_factored(buf_t_logits[t], s_logits)
                            if args.phase == 3:
                                val_d = model.critic(torch.cat([ext["gru_h"], buf_priv[t]], dim=-1)).squeeze(-1)
                                l_vdist_acc = l_vdist_acc + F.mse_loss(val_d, buf_t_val[t])

                        if args.phase == 4:
                            dists = [Categorical(logits=lg) for lg in s_logits]
                            lp = torch.stack([d.log_prob(buf_act_f[t, :, i]) for i, d in enumerate(dists)], dim=-1).sum(dim=-1)
                            en = torch.stack([d.entropy() for d in dists], dim=-1).sum(dim=-1)
                            val_t = model.critic(torch.cat([ext["gru_h"], buf_priv[t]], dim=-1)).squeeze(-1)
                            logp_new_list.append(lp)
                            val_new_list.append(val_t)
                            ent_list.append(en)

                        nxt_s = model.update_state_with_action(nxt_s, buf_act_mb[t])
                        cur_state = _reset_masked_state(nxt_s, buf_done[t], fresh_template)

                    l_ground = (l_odom_acc + l_bev_acc + l_aff_acc + 0.5 * l_idm_acc) / T
                    l_kl = l_kl_acc / T
                    l_vdist = l_vdist_acc / T

                    if args.phase == 1:
                        loss = l_ground
                        l_ppo = torch.zeros((), device=device)
                    elif args.phase == 3:
                        loss = l_kl + 0.25 * l_vdist + 0.5 * l_ground
                        l_ppo = torch.zeros((), device=device)
                    else:
                        logp_new = torch.stack(logp_new_list, dim=0)
                        val_new = torch.stack(val_new_list, dim=0)
                        ent_mean = torch.stack(ent_list, dim=0).mean()
                        log_ratio = logp_new - buf_logp
                        ratio = torch.exp(log_ratio)
                        with torch.no_grad():
                            approx_kl = float(((ratio - 1.0) - log_ratio).mean().item())
                        if approx_kl > args.target_kl:
                            break
                        surr1 = ratio * adv
                        surr2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * adv
                        l_ppo = -torch.min(surr1, surr2).mean()
                        l_val = F.mse_loss(val_new, ret)
                        loss = l_ppo + 0.5 * l_val - args.ent_coef * ent_mean + kl_beta * l_kl + 0.25 * l_ground

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                scaler.step(opt)
                scaler.update()
                metrics = {
                    "loss": float(loss.item()),
                    "ground": float(l_ground.item()),
                    "kl": float(l_kl.item()),
                    "ppo": float(l_ppo.item()),
                }

            fps = global_step / max(1e-3, time.perf_counter() - t_start)
            ex_rate = float(np.mean(ep_exits[-50:])) if ep_exits else 0.0
            mean_ret = float(np.mean(recent_returns[-50:])) if recent_returns else 0.0
            print(
                f"P{args.phase} upd={update_idx:04d} step={global_step:8d} fps={fps:5.0f} "
                f"ret={mean_ret:6.2f} exit={ex_rate*100:5.1f}% "
                f"loss={metrics['loss']:.3f} gnd={metrics['ground']:.3f} kl={metrics['kl']:.3f}",
                flush=True,
            )

            if update_idx % args.save_every == 0 or global_step >= args.total_steps:
                tmp = save_path.with_suffix(".tmp")
                torch.save(
                    {
                        "phase": args.phase,
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
                    torch.save({"model": model.strip_for_submission()}, etmp)
                    etmp.replace(export_path)
                print(f"[AEGIS-3D] Saved checkpoint -> {save_path} | Exported -> {export_path}", flush=True)
    finally:
        for e in envs:
            e.close()


if __name__ == "__main__":
    main()
