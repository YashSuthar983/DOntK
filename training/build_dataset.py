"""Build training packed DoomForge WAD shards + locked 40-map Zero-Leakage Eval Suite.

Run:
    python training/build_dataset.py --train-maps-per-tier 50 --eval-maps-per-tier 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.doomforge import build_wad_pack

TIER_TICKS = {1: 2100, 2: 2800, 3: 3500, 4: 4200}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="training/wads")
    ap.add_argument("--train-maps-per-tier", type=int, default=50)
    ap.add_argument("--eval-maps-per-tier", type=int, default=10)
    args = ap.parse_args()

    out = Path(args.out_dir)
    train_dir = out / "train"
    eval_dir = out / "eval_locked"
    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    for tier in (1, 2, 3, 4):
        p = build_wad_pack(
            train_dir / f"tier{tier}_train.wad",
            tier=tier,
            num_maps=args.train_maps_per_tier,
            base_seed=10000 * tier,
        )
        print(f"Built train shard: {p} ({args.train_maps_per_tier} maps)")

    suite: list[dict] = []
    for tier in (1, 2, 3, 4):
        p = build_wad_pack(
            eval_dir / f"tier{tier}_eval.wad",
            tier=tier,
            num_maps=args.eval_maps_per_tier,
            base_seed=90000 + 1000 * tier,
        )
        for i in range(args.eval_maps_per_tier):
            suite.append(
                {
                    "wad": str(p.resolve()),
                    "map_name": f"MAP{i + 1:02d}",
                    "tier": tier,
                    "episode_ticks": TIER_TICKS[tier],
                    "seed": 42 + i,
                }
            )
        print(f"Built locked eval shard: {p} ({args.eval_maps_per_tier} maps)")

    manifest = eval_dir / "eval_suite.json"
    manifest.write_text(json.dumps(suite, indent=2), encoding="utf-8")
    print(f"Saved locked {len(suite)}-map zero-leakage suite -> {manifest}")


if __name__ == "__main__":
    main()
