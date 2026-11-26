"""One-Command Full 4-Phase Colab GPU / Local Training Orchestrator for AEGIS-3D.

Designed for resilient, resumable training on Google Colab (T4/L4/A100) with Google Drive
checkpoint persistence, OR on a multi-core local machine.

Example on Google Colab (after cloning repo & mounting /content/drive):
    python training/colab_train.py \
        --ckpt-dir /content/drive/MyDrive/aegis3d_ckpts \
        --num-envs 8 \
        --p1-steps 500000 \
        --p2-steps 5000000 \
        --p3-steps 3000000 \
        --p4-steps 10000000
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
        raise SystemExit(f"Command failed with exit code {rc}: {' '.join(cmd)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="training/checkpoints", help="Checkpoint dir (point to Google Drive on Colab!)")
    ap.add_argument("--wad-dir", default="training/wads")
    ap.add_argument("--train-maps-per-tier", type=int, default=100)
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--p1-steps", type=int, default=200_000)
    ap.add_argument("--p2-steps", type=int, default=2_000_000)
    ap.add_argument("--p3-steps", type=int, default=1_500_000)
    ap.add_argument("--p4-steps", type=int, default=5_000_000)
    args = ap.parse_args()

    py = sys.executable
    ckpts = Path(args.ckpt_dir)
    ckpts.mkdir(parents=True, exist_ok=True)
    wads = Path(args.wad_dir)

    # 0. Generate packed training shards + locked 40-map eval suite if missing
    train_wads = [wads / "train" / f"tier{t}_train.wad" for t in (1, 2, 3, 4)]
    if not all(p.is_file() for p in train_wads):
        _run(
            [
                py,
                "training/build_dataset.py",
                "--out-dir",
                str(wads),
                "--train-maps-per-tier",
                str(args.train_maps_per_tier),
                "--eval-maps-per-tier",
                "10",
            ]
        )

    all_wads = [str(p) for p in train_wads]
    t_ckpt = ckpts / "teacher.pt"
    p1_ckpt = ckpts / "aegis_p1.pt"
    p3_ckpt = ckpts / "aegis_p3.pt"
    p4_ckpt = ckpts / "aegis_p4.pt"
    export = ROOT / "submission" / "weights" / "aegis3d.pt"

    # Phase 2: Fast Omniscient Teacher PPO (render_buffers=False)
    _run(
        [
            py,
            "training/train_teacher.py",
            "--wad",
            *all_wads,
            "--num-envs",
            str(args.num_envs),
            "--rollout-steps",
            "128",
            "--total-steps",
            str(args.p2_steps),
            "--save",
            str(t_ckpt),
            *(["--resume", str(t_ckpt)] if t_ckpt.is_file() else []),
        ]
    )

    # Phase 1: Supervised Sensor & Memory Bootstrapping
    _run(
        [
            py,
            "training/train_aegis.py",
            "--phase",
            "1",
            "--wad",
            *all_wads,
            "--num-envs",
            str(args.num_envs),
            "--rollout-steps",
            "64",
            "--total-steps",
            str(args.p1_steps),
            "--save",
            str(p1_ckpt),
            "--export-submission",
            str(export),
            *(["--resume", str(p1_ckpt)] if p1_ckpt.is_file() else []),
        ]
    )

    # Phase 3: Online Student DAgger KL Distillation + Privileged Grounding
    p3_resume = p3_ckpt if p3_ckpt.is_file() else p1_ckpt
    _run(
        [
            py,
            "training/train_aegis.py",
            "--phase",
            "3",
            "--wad",
            *all_wads,
            "--teacher-ckpt",
            str(t_ckpt),
            "--resume",
            str(p3_resume),
            "--num-envs",
            str(args.num_envs),
            "--rollout-steps",
            "64",
            "--total-steps",
            str(args.p3_steps),
            "--save",
            str(p3_ckpt),
            "--export-submission",
            str(export),
        ]
    )

    # Phase 4: Asymmetric Recurrent PPO + E3B Curiosity + Annealed Teacher KL
    p4_resume = p4_ckpt if p4_ckpt.is_file() else p3_ckpt
    _run(
        [
            py,
            "training/train_aegis.py",
            "--phase",
            "4",
            "--wad",
            *all_wads,
            "--teacher-ckpt",
            str(t_ckpt),
            "--resume",
            str(p4_resume),
            "--num-envs",
            str(args.num_envs),
            "--rollout-steps",
            "64",
            "--total-steps",
            str(args.p4_steps),
            "--save",
            str(p4_ckpt),
            "--export-submission",
            str(export),
        ]
    )

    # Final validation check
    _run([py, "validate.py"])


if __name__ == "__main__":
    main()
