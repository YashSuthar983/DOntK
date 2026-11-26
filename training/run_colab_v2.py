"""One-Command Full AEGIS-3D v2 Training Orchestrator (Colab GPU / Linux GPU).

Runs the complete 8-hour winning pipeline automatically with resumable checkpoints:
  0. Build 400 Diverse 100%-Solvable DoomForge V2 Train Maps + 40-Map Locked Eval Suite
     + mix in 32 real Freedoom2 campaign maps.
  1A. Online DAgger Warmup (1.0M steps, beta 0.90 -> 0.15) from Fog-of-War FrontierOracle
  1B. Student-Rollout DAgger Hardening (600k steps, beta 0.25 -> 0.00)
  2.  Asymmetric Recurrent PPO + Live E3B + Annealed DAgger KL Anchor (3.5M steps)
  3.  Inference Decoding Temperature Calibration on Locked 40-Map Suite + Official validate.py

Usage:
    python training/run_colab_v2.py --ckpt-dir /content/drive/MyDrive/aegis3d_v2_ckpts
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    env = {**os.environ, "PYTHONPATH": f"{ROOT}:{ROOT / 'env'}:{ROOT / 'submission'}"}
    rc = subprocess.call(cmd, cwd=ROOT, env=env)
    if rc != 0:
        raise SystemExit(f"Command failed ({rc}): {' '.join(cmd)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="training/checkpoints/v2")
    ap.add_argument("--wad-dir", default="training/wads")
    ap.add_argument("--num-envs", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=max(2, min(4, os.cpu_count() or 2)))
    ap.add_argument("--dagger1-steps", type=int, default=800_000)
    ap.add_argument("--dagger2-steps", type=int, default=600_000)
    ap.add_argument("--ppo-steps", type=int, default=3_500_000)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    py = sys.executable
    ckpts = Path(args.ckpt_dir)
    ckpts.mkdir(parents=True, exist_ok=True)
    wads = Path(args.wad_dir)
    export = ROOT / "submission" / "weights" / "aegis3d.pt"

    train_wads = [wads / "train" / f"tier{t}_train.wad" for t in (1, 2, 3, 4)]
    eval_manifest = wads / "eval_locked" / "eval_suite.json"
    if not all(p.is_file() for p in train_wads) or not eval_manifest.is_file():
        _run([
            py, "training/build_dataset.py",
            "--out-dir", str(wads),
            "--train-maps-per-tier", "100",
            "--eval-maps-per-tier", "10",
        ])

    import vizdoom
    freedoom2 = str(Path(vizdoom.__file__).parent / "freedoom2.wad")
    # Weighted 16-env mix: 3x T1, 4x T2, 4x T3, 4x T4, 1x Freedoom2 (or balanced across all 5!)
    # Passing [T1, T2, T3, T4, T2, T3, T4, Freedoom2] puts 2 envs on Freedoom2 and heavy weight on T2-T4!
    all_wads = [
        str(train_wads[0]),
        str(train_wads[1]),
        str(train_wads[2]),
        str(train_wads[3]),
        str(train_wads[1]),
        str(train_wads[2]),
        str(train_wads[3]),
        freedoom2,
    ]
    hard_wads = [
        str(train_wads[1]),
        str(train_wads[2]),
        str(train_wads[3]),
        freedoom2,
    ]

    d1_ckpt = ckpts / "aegis_v2_dagger1.pt"
    d2_ckpt = ckpts / "aegis_v2_dagger2.pt"
    ppo_ckpt = ckpts / "aegis_v2_ppo.pt"

    # Stage 1A: Online DAgger Warmup (beta 0.90 -> 0.15)
    _run([
        py, "training/train_dagger_ppo.py",
        "--mode", "dagger",
        "--wad", *all_wads,
        "--num-envs", str(args.num_envs),
        "--num-workers", str(args.num_workers),
        "--episode-ticks", "2800",
        "--rollout-steps", "64",
        "--chunk-len", "32",
        "--minibatches", "2",
        "--epochs", "3",
        "--lr", "4e-4",
        "--beta-start", "0.90",
        "--beta-end", "0.15",
        "--total-steps", str(args.dagger1_steps),
        "--device", args.device,
        "--save", str(d1_ckpt),
        "--export-submission", str(export),
        "--save-every", "20",
        *(["--resume", str(d1_ckpt)] if d1_ckpt.is_file() else []),
    ])

    # Stage 1B: Pure-Student DAgger Hardening on Hard Tiers + Freedoom2 (beta 0.25 -> 0.00)
    d2_resume = d2_ckpt if d2_ckpt.is_file() else d1_ckpt
    _run([
        py, "training/train_dagger_ppo.py",
        "--mode", "dagger",
        "--wad", *hard_wads,
        "--num-envs", str(args.num_envs),
        "--num-workers", str(args.num_workers),
        "--episode-ticks", "3500",
        "--rollout-steps", "64",
        "--chunk-len", "32",
        "--minibatches", "2",
        "--epochs", "3",
        "--lr", "2.5e-4",
        "--beta-start", "0.25",
        "--beta-end", "0.00",
        "--total-steps", str(args.dagger2_steps),
        "--device", args.device,
        "--resume", str(d2_resume),
        "--save", str(d2_ckpt),
        "--export-submission", str(export),
        "--save-every", "20",
    ])

    # Quick eval snapshot after DAgger before PPO
    _run([
        py, "training/eval_and_calibrate.py",
        "--weights", str(export),
        "--device", args.device,
        "--out-json", str(ckpts / "eval_after_dagger.json"),
    ])

    # Stage 2: Asymmetric Recurrent PPO + Live E3B + Annealed DAgger KL Anchor
    ppo_resume = ppo_ckpt if ppo_ckpt.is_file() else d2_ckpt
    _run([
        py, "training/train_dagger_ppo.py",
        "--mode", "ppo",
        "--wad", *all_wads,
        "--num-envs", str(args.num_envs),
        "--num-workers", str(args.num_workers),
        "--episode-ticks", "3500",
        "--rollout-steps", "64",
        "--chunk-len", "32",
        "--minibatches", "2",
        "--epochs", "3",
        "--lr", "1.8e-4",
        "--ent-coef", "0.007",
        "--e3b-coef", "0.015",
        "--kl-coef", "0.04",
        "--ground-coef", "0.20",
        "--total-steps", str(args.ppo_steps),
        "--device", args.device,
        "--resume", str(ppo_resume),
        "--anchor-ckpt", str(d2_ckpt),
        "--save", str(ppo_ckpt),
        "--export-submission", str(export),
        "--save-every", "25",
    ])

    # Stage 3: Calibrate Inference Decoding Temps + Full Locked 40-Map Eval + Official Validator
    _run([
        py, "training/eval_and_calibrate.py",
        "--weights", str(export),
        "--device", args.device,
        "--calibrate",
        "--out-json", str(ckpts / "eval_final_report.json"),
    ])
    val_dev = "cuda" if args.device in ("auto", "cuda") else "cpu"
    _run([py, "validate.py", "--device", val_dev])


if __name__ == "__main__":
    main()
