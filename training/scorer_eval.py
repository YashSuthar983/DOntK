"""Zero-Leakage Scorer-Replica Evaluator for AEGIS-3D.

Mirrors the four official scoring categories in docs/EVALUATION.md using the exact
ground-truth hooks in doom_env/wrapper.py (`privileged_state()` and `sector_polygons()`):
  1. Exit reached + remaining health & time efficiency bonus
  2. Progression: keycards picked up, ordinary doors opened, locked doors unlocked
  3. Exploration: unique WAD sectors visited + unique 128x128 spatial cells visited
  4. Combat & survival: kills, damage taken, early death penalty
Plus per-step latency tracking (p50, p99, max ms).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from doom_env import make_env


def _point_in_sector(px: float, py: float, segs: np.ndarray) -> bool:
    """Ray-casting point-in-polygon test against (K, 4) [x1, y1, x2, y2] boundary segments."""
    if len(segs) == 0:
        return False
    x1, y1, x2, y2 = segs[:, 0], segs[:, 1], segs[:, 2], segs[:, 3]
    cond = (y1 > py) != (y2 > py)
    if not np.any(cond):
        return False
    x1c, y1c, x2c, y2c = x1[cond], y1[cond], x2[cond], y2[cond]
    x_int = x1c + (py - y1c) * (x2c - x1c) / (y2c - y1c)
    return bool(np.count_nonzero(px < x_int) % 2 == 1)


@dataclass
class MapEvalResult:
    map_id: str
    tier: int
    episode_ticks: int
    steps: int
    tics_used: int
    exit_reached: bool
    dead: bool
    health_end: float
    keys_collected: int
    doors_opened: int
    locked_doors_unlocked: int
    sectors_visited: int
    total_sectors: int
    cells_visited: int
    kills: int
    damage_taken: float
    score_exit: float
    score_progression: float
    score_exploration: float
    score_combat: float
    total_score: float
    p50_ms: float
    p99_ms: float
    max_ms: float


TIER_WEIGHTS = {1: 1.0, 2: 1.5, 3: 2.0, 4: 3.0}


def evaluate_agent_on_map(
    agent: Any,
    wad_path: str | Path,
    map_name: str = "MAP01",
    *,
    tier: int = 1,
    episode_ticks: int = 2100,
    seed: int = 0,
    record_path: str | Path | None = None,
) -> MapEvalResult:
    """Evaluate a DoomAgent-compatible object on a single map and return detailed metrics."""
    env = make_env(wad_path, map_name, episode_ticks=episode_ticks, render_buffers=True)
    try:
        options = {"record_path": str(record_path)} if record_path else None
        obs, _ = env.reset(seed=seed, options=options)
        agent.reset()

        sector_polys = env.sector_polygons()
        total_sectors = max(1, len(sector_polys))
        sector_bbs = [
            (
                sid,
                segs,
                float(min(segs[:, 0].min(), segs[:, 2].min())),
                float(max(segs[:, 0].max(), segs[:, 2].max())),
                float(min(segs[:, 1].min(), segs[:, 3].min())),
                float(max(segs[:, 1].max(), segs[:, 3].max())),
            )
            for sid, segs in sector_polys
            if len(segs) > 0
        ]

        visited_sectors: set[int] = set()
        visited_cells: set[tuple[int, int]] = set()
        opened_doors: set[int] = set()
        unlocked_doors: set[int] = set()
        latencies_ms: list[float] = []

        def _update_trackers(priv: dict) -> None:
            px, py = priv["x"], priv["y"]
            visited_cells.add((int(px // 128), int(py // 128)))
            for sid, segs, xmin, xmax, ymin, ymax in sector_bbs:
                if sid not in visited_sectors and xmin <= px <= xmax and ymin <= py <= ymax:
                    if _point_in_sector(px, py, segs):
                        visited_sectors.add(sid)
                        break
            floor = priv["sector_floor"]
            ceil = priv["sector_ceiling"]
            locks = priv["door_lock"]
            if floor is not None and ceil is not None and locks is not None:
                open_mask = (ceil - floor) >= 56.0
                for sid in np.flatnonzero(open_mask & (locks == 0)):
                    opened_doors.add(int(sid))
                for sid in np.flatnonzero(open_mask & (locks > 0)):
                    unlocked_doors.add(int(sid))

        _update_trackers(env.privileged_state())
        steps = 0
        while True:
            t0 = time.perf_counter()
            action = agent.act(obs)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            obs, _, term, trunc, _ = env.step(action)
            steps += 1
            priv = env.privileged_state()
            _update_trackers(priv)
            if term or trunc:
                break

        # Mirror docs/EVALUATION.md scoring buckets
        exit_reached = bool(priv["exit_reached"])
        dead = bool(priv["dead"])
        tics_used = int(priv["tic"])
        health_end = max(0.0, float(priv["health"]))
        keys_collected = int(np.sum(priv["keys"] > 0.5))
        kills = int(priv["kills"])
        damage_taken = float(priv["damage_taken"])

        score_exit = 0.0
        if exit_reached:
            time_ratio = max(0.0, 1.0 - tics_used / max(1, episode_ticks))
            hp_ratio = min(1.0, health_end / 100.0)
            score_exit = 50.0 + 10.0 * hp_ratio + 10.0 * time_ratio

        score_progression = 15.0 * keys_collected + 5.0 * len(opened_doors) + 20.0 * len(unlocked_doors)
        coverage_frac = len(visited_sectors) / total_sectors
        score_exploration = 15.0 * coverage_frac + 0.5 * len(visited_cells)
        early_death_pen = 15.0 * (1.0 - tics_used / max(1, episode_ticks)) if dead else 0.0
        score_combat = 4.0 * kills - 0.1 * damage_taken - early_death_pen
        total_score = score_exit + score_progression + score_exploration + score_combat

        lat_arr = np.asarray(latencies_ms, dtype=np.float64) if latencies_ms else np.zeros(1)
        return MapEvalResult(
            map_id=f"{Path(wad_path).name}:{map_name}",
            tier=tier,
            episode_ticks=episode_ticks,
            steps=steps,
            tics_used=tics_used,
            exit_reached=exit_reached,
            dead=dead,
            health_end=round(health_end, 1),
            keys_collected=keys_collected,
            doors_opened=len(opened_doors),
            locked_doors_unlocked=len(unlocked_doors),
            sectors_visited=len(visited_sectors),
            total_sectors=total_sectors,
            cells_visited=len(visited_cells),
            kills=kills,
            damage_taken=round(damage_taken, 1),
            score_exit=round(score_exit, 2),
            score_progression=round(score_progression, 2),
            score_exploration=round(score_exploration, 2),
            score_combat=round(score_combat, 2),
            total_score=round(total_score, 2),
            p50_ms=round(float(np.percentile(lat_arr, 50)), 2),
            p99_ms=round(float(np.percentile(lat_arr, 99)), 2),
            max_ms=round(float(np.max(lat_arr)), 2),
        )
    finally:
        env.close()


def evaluate_suite(agent: Any, suite: list[dict], out_json: str | Path | None = None) -> dict:
    """Evaluate across a list of map dicts: {wad, map_name, tier, episode_ticks, seed}."""
    results = [
        evaluate_agent_on_map(
            agent,
            entry["wad"],
            entry.get("map_name", "MAP01"),
            tier=int(entry.get("tier", 1)),
            episode_ticks=int(entry.get("episode_ticks", 2100)),
            seed=int(entry.get("seed", 0)),
        )
        for entry in suite
    ]
    weighted = sum(r.total_score * TIER_WEIGHTS.get(r.tier, 1.0) for r in results)
    summary = {
        "weighted_total_score": round(weighted, 2),
        "exit_rate": round(float(np.mean([r.exit_reached for r in results])), 4) if results else 0.0,
        "survival_rate": round(float(np.mean([not r.dead for r in results])), 4) if results else 0.0,
        "mean_p50_ms": round(float(np.mean([r.p50_ms for r in results])), 2) if results else 0.0,
        "maps": [asdict(r) for r in results],
    }
    if out_json is not None:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
