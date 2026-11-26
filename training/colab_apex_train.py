"""One-shot Colab T4 GPU trainer for ApexNet (Phase A DAgger -> Phase B Kickstart PPO -> Phase C PPO).

Usage on Colab T4:
    !git clone https://github.com/YashSuthar983/DOntK.git /content/DOntK
    !cd /content/DOntK && python training/colab_apex_train.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import torch
import vizdoom

ROOT = Path(__file__).resolve().parents[1]
WADS = ROOT / "training" / "wads" / "train"
CKPT = ROOT / "training" / "checkpoints"
EXPORT = ROOT / "submission" / "weights" / "aegis3d.pt"
FREEDOOM2 = str(Path(vizdoom.__file__).parent / "freedoom2.wad")


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    env = {**os.environ, "PYTHONPATH": f"{ROOT}:{ROOT / 'env'}:{ROOT / 'submission'}", "PYTHONUNBUFFERED": "1"}
    rc = subprocess.call(cmd, cwd=ROOT, env=env)
    if rc != 0:
        raise SystemExit(f"Command failed ({rc}): {cmd}")


def ensure_wads() -> list[str]:
    required = [
        WADS / "tier1_train.wad",
        WADS / "tier2_train.wad",
        WADS / "tier3_train.wad",
        WADS / "tier4_train.wad",
    ]
    if not all(p.is_file() for p in required):
        run([
            sys.executable, "training/build_dataset.py",
            "--out-dir", "training/wads",
            "--train-maps-per-tier", "100",
            "--eval-maps-per-tier", "10",
        ])
    t1, t2, t3, t4 = (str(p) for p in required)
    wild = str(WADS / "wild_train.wad") if (WADS / "wild_train.wad").is_file() else t3
    skills = str(WADS / "skills_train.wad") if (WADS / "skills_train.wad").is_file() else t2
    # 24-env maximum-generalization OOD mix (Tier 1-4 + WildForge + Skills + Freedoom2)
    return [
        skills, wild, t1, t3, skills, wild, t2, t4,
        skills, wild, FREEDOOM2, t3, skills, wild, t1, t4,
        skills, wild, t2, t3, skills, wild, FREEDOOM2, t4,
    ]


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    workers = str(max(2, min(6, os.cpu_count() or 2)))
    wads = ensure_wads()
    CKPT.mkdir(parents=True, exist_ok=True)

    p_dagger = CKPT / "apex_dagger.pt"
    p_kick = CKPT / "apex_kickstart.pt"
    p_ppo = CKPT / "apex_ppo.pt"

    # Phase A: DAgger with ApexTeacher (beta: 0.5 -> 0.0, detached value warm-up)
    run([
        sys.executable, "training/train_apex.py",
        "--mode", "dagger", "--wad", *wads,
        "--num-envs", "24", "--num-workers", workers,
        "--episode-ticks", "3200", "--rollout-steps", "64", "--chunk-len", "32",
        "--minibatches", "4", "--epochs", "3", "--lr", "3e-4",
        "--beta-start", "0.5", "--beta-end", "0.0",
        "--total-steps", "500000", "--device", dev,
        "--save", str(p_dagger), "--export-submission", str(EXPORT), "--save-every", "20",
    ])

    # Phase B: Joint Kickstart PPO (student drives 100%, teacher CE decays 0.8 -> 0.1)
    run([
        sys.executable, "training/train_apex.py",
        "--mode", "kickstart", "--wad", *wads,
        "--num-envs", "24", "--num-workers", workers,
        "--episode-ticks", "3200", "--rollout-steps", "64", "--chunk-len", "32",
        "--minibatches", "4", "--epochs", "3", "--lr", "2.5e-4",
        "--teacher-coef-start", "0.8", "--teacher-coef-end", "0.1",
        "--total-steps", "2000000", "--device", dev,
        "--resume", str(p_dagger), "--reset-step",
        "--save", str(p_kick), "--export-submission", str(EXPORT), "--save-every", "20",
    ])

    # Phase C: Light-anchor PPO (surpass teacher on unseen maps & Tier 4)
    run([
        sys.executable, "training/train_apex.py",
        "--mode", "ppo", "--wad", *wads,
        "--num-envs", "24", "--num-workers", workers,
        "--episode-ticks", "3200", "--rollout-steps", "64", "--chunk-len", "32",
        "--minibatches", "4", "--epochs", "3", "--lr", "1.5e-4",
        "--teacher-coef-start", "0.05", "--teacher-coef-end", "0.0",
        "--total-steps", "2000000", "--device", dev,
        "--resume", str(p_kick), "--reset-step",
        "--save", str(p_ppo), "--export-submission", str(EXPORT), "--save-every", "20",
    ])


if __name__ == "__main__":
    main()
