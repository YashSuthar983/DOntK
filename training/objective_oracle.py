"""Training-only DAgger teacher with reachability-based progression goals."""

from __future__ import annotations

import math

import numpy as np

from doom_env.sensors import PLAYER_HEIGHT
from training.frontier_oracle import FrontierOracle
from training.wad_oracle import FastDoomPackEnv


class ObjectiveOracle(FrontierOracle):
    """Prefer a reachable exit and fall back to exploration when the WAD goals are unreachable."""

    def _build_blocked_grid(self, env: FastDoomPackEnv, priv: dict) -> np.ndarray:
        blocked = super()._build_blocked_grid(env, priv)
        floor, ceiling = priv["sector_floor"], priv["sector_ceiling"]
        if floor is None or ceiling is None:
            return blocked
        field = env._fields[env.current_map]
        for switch in env.manifest.get(env.current_map, {}).get("switches", []):
            if switch.get("kind") != "remote_door":
                continue
            for sector in env._remote_sectors(switch):
                if 0 <= sector < len(floor):
                    closed = ceiling[sector] - floor[sector] < PLAYER_HEIGHT
                    for cell in field.door_cells.get(sector, ()):
                        blocked[cell] = closed
        return blocked

    def _select_nav_target(
        self, env: FastDoomPackEnv, priv: dict, blocked: np.ndarray,
    ) -> tuple[str, tuple[float, float] | None, tuple[int, int] | None]:
        meta = env.manifest.get(env.current_map, {})
        field = env._fields[env.current_map]
        player = field.world_to_cell(priv["x"], priv["y"])

        def distance_to(cell: tuple[int, int]) -> float:
            if not field.walkable[cell] or blocked[cell]:
                return math.inf
            return float(field._bfs_from(*cell, blocked)[player])

        exit_pos = meta.get("exit_pos")
        if exit_pos is not None:
            cell = field.world_to_cell(*exit_pos)
            if math.isfinite(distance_to(cell)):
                kind = "exit_switch" if meta.get("exit_type") == "switch" else "exit_walk"
                return kind, tuple(exit_pos), cell

        goals = []
        for switch in meta.get("switches", []):
            if switch.get("kind") != "remote_door":
                continue
            door_cells = [cell for sector in env._remote_sectors(switch) for cell in field.door_cells.get(sector, ())]
            if door_cells:
                if not any(blocked[cell] for cell in door_cells):
                    continue
            elif field.world_to_cell(*switch["pos"]) in self.triggered_switches:
                continue
            goals.append(("remote_switch", switch["pos"]))
        for key in meta.get("keys_placed", []):
            if not self._has_key_for_lock(priv["keys"], key["lock"]):
                goals.append(("key", key["pos"]))

        reachable = []
        for kind, xy in goals:
            cell = field.world_to_cell(*xy)
            distance = distance_to(cell)
            if math.isfinite(distance):
                reachable.append((distance, kind, tuple(xy), cell))
        if reachable:
            _, kind, xy, cell = min(reachable)
            return kind, xy, cell
        return super()._select_nav_target(env, priv, blocked)
