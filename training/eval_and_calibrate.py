"""Locked 40-Map Evaluator & Inference Decoding Temperature Calibrator for AEGIS-3D v2.

Usage:
    PYTHONPATH=.:env:submission python training/eval_and_calibrate.py \
        --weights submission/weights/aegis3d.pt \
        --device cuda --calibrate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_ROOT / "env"), str(_ROOT / "submission")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from doom_env import action_space, observation_space
from agent import DoomAgent
from training.scorer_eval import evaluate_suite


CANDIDATE_TEMPS = {
    "argmax": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "sharp": (0.20, 0.30, 0.20, 0.15, 0.45, 0.15),
    "balanced": (0.35, 0.45, 0.35, 0.25, 0.55, 0.25),
    "soft": (0.50, 0.60, 0.50, 0.35, 0.70, 0.35),
    "sample": (0.75, 0.80, 0.75, 0.60, 0.85, 0.60),
}


def _summarize_tiers(report: dict) -> str:
    by_tier: dict[int, list[dict]] = {}
    for m in report["maps"]:
        by_tier.setdefault(int(m["tier"]), []).append(m)
    parts = []
    for t in sorted(by_tier):
        ms = by_tier[t]
        ex = np.mean([x["exit_reached"] for x in ms]) * 100.0
        dead = np.mean([x["dead"] for x in ms]) * 100.0
        sc = np.mean([x["total_score"] for x in ms])
        parts.append(f"T{t}: exit={ex:5.1f}% dead={dead:4.1f}% score={sc:5.1f}")
    return " | ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="submission/weights/aegis3d.pt")
    ap.add_argument("--suite", default="training/wads/eval_locked/eval_suite.json")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--calibrate", action="store_true", help="Sweep decode_temps and write best into weights file")
    ap.add_argument("--out-json", default="training/checkpoints/eval_report.json")
    args = ap.parse_args()

    dev = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else (args.device if args.device != "auto" else "cpu")
    suite = json.loads(Path(args.suite).read_text(encoding="utf-8"))

    agent = DoomAgent(observation_space(), action_space(), dev)
    wpath = Path(args.weights)
    ckpt = torch.load(wpath, map_location=dev, weights_only=True)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    agent.model.load_state_dict(sd, strict=True)
    agent.model.eval()

    if not args.calibrate:
        if isinstance(ckpt, dict) and "decode_temps" in ckpt:
            agent.decode_temps = tuple(float(x) for x in ckpt["decode_temps"])
        rep = evaluate_suite(agent, suite, out_json=args.out_json)
        print(f"Weighted Score: {rep['weighted_total_score']:.2f} | Exit: {rep['exit_rate']*100:.1f}% | Surv: {rep['survival_rate']*100:.1f}% | p50: {rep['mean_p50_ms']:.2f}ms")
        print(_summarize_tiers(rep))
        return

    best_name, best_temps, best_rep = None, None, None
    best_score = -1e9
    for name, temps in CANDIDATE_TEMPS.items():
        agent.decode_temps = temps
        rep = evaluate_suite(agent, suite)
        print(
            f"[{name:8s}] score={rep['weighted_total_score']:7.2f} exit={rep['exit_rate']*100:5.1f}% "
            f"surv={rep['survival_rate']*100:5.1f}% p50={rep['mean_p50_ms']:.2f}ms | {_summarize_tiers(rep)}",
            flush=True,
        )
        if rep["weighted_total_score"] > best_score:
            best_score = rep["weighted_total_score"]
            best_name, best_temps, best_rep = name, temps, rep

    assert best_rep is not None and best_temps is not None
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(best_rep, indent=2), encoding="utf-8")

    # Save optimal decode_temps cleanly inside the submission weights file!
    save_sd = {k: v.detach().cpu() for k, v in sd.items()}
    tmp = wpath.with_suffix(".tmp")
    torch.save({"model": save_sd, "decode_temps": list(best_temps)}, tmp)
    tmp.replace(wpath)
    print(f"\nSaved optimal decode_temps ({best_name}: {best_temps}) -> {wpath} | Best Weighted Score: {best_score:.2f}")


if __name__ == "__main__":
    main()
