"""Training-Only WAD Geodesic Oracle, Privileged Grounding Labeler & Fast Multi-Map Env.

NEVER imported by submission/agent.py. Used exclusively during training for:
1. Fast <2 ms map switching across 100-map packed DoomForge / Freedoom2 WADs.
2. Exact shortest-path walkable geodesic distance d_geo(s_t, g_t) & corridor compass
   to the currently active progression objective (Key -> Locked Door -> Switch -> Exit).
3. Policy-invariant potential reward shaping + scorer-mirrored event rewards.
4. Privileged targets for Asymmetric Critic, Omniscient Teacher, and the 5 Grounding
   Losses (odometry, 64x64 ego-BEV occupancy, 20-col visual affordances, door belief).
"""

from __future__ import annotations

import json
import math
import struct
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from doom_env import make_env
from doom_env import spaces as S
from doom_env.mapinfo import load_mapinfo
from doom_env.sensors import LevelGeometry, Sensors, ego_rotate
from doom_env.wad import list_maps, map_lumps, read_wad

GRID_CELL = 64.0
BEV_SIZE = 64
BEV_RES = 32.0  # 64 * 32 = 2048 map units around player


class GridGeodesicField:
    """Fast 64-unit walkable grid extracted from WAD LineGeometry for exact geodesic distances."""

    def __init__(self, mapinfo, meta: dict | None = None) -> None:
        g = mapinfo.lines
        if g is None or len(g.p1) == 0:
            self.ox, self.oy, self.w, self.h = 0.0, 0.0, 1, 1
            self.walkable = np.ones((1, 1), dtype=bool)
        else:
            all_pts = np.concatenate([g.p1, g.p2], axis=0)
            xmin, ymin = all_pts.min(axis=0)
            xmax, ymax = all_pts.max(axis=0)
            self.ox = float(math.floor(xmin / GRID_CELL) * GRID_CELL)
            self.oy = float(math.floor(ymin / GRID_CELL) * GRID_CELL)
            self.w = max(1, int(math.ceil((xmax - self.ox) / GRID_CELL)))
            self.h = max(1, int(math.ceil((ymax - self.oy) / GRID_CELL)))
            # Mark cells inside any sector polygon or touched by non-blocking interior
            self.walkable = np.zeros((self.w, self.h), dtype=bool)
            for _sid, segs in mapinfo.sector_lines().items():
                if len(segs) == 0:
                    continue
                sxmin = int(max(0, math.floor((min(segs[:, 0].min(), segs[:, 2].min()) - self.ox) / GRID_CELL)))
                sxmax = int(min(self.w - 1, math.floor((max(segs[:, 0].max(), segs[:, 2].max()) - self.ox) / GRID_CELL)))
                symin = int(max(0, math.floor((min(segs[:, 1].min(), segs[:, 3].min()) - self.oy) / GRID_CELL)))
                symax = int(min(self.h - 1, math.floor((max(segs[:, 1].max(), segs[:, 3].max()) - self.oy) / GRID_CELL)))
                for cx in range(sxmin, sxmax + 1):
                    px = self.ox + (cx + 0.5) * GRID_CELL
                    for cy in range(symin, symax + 1):
                        if self.walkable[cx, cy]:
                            continue
                        py = self.oy + (cy + 0.5) * GRID_CELL
                        if _point_in_segs(px, py, segs):
                            self.walkable[cx, cy] = True
            if not self.walkable.any():
                self.walkable[:] = True

        self.meta = meta or {}
        self._dist_cache: dict[tuple[int, int], np.ndarray] = {}

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        cx = int(np.clip(math.floor((x - self.ox) / GRID_CELL), 0, self.w - 1))
        cy = int(np.clip(math.floor((y - self.oy) / GRID_CELL), 0, self.h - 1))
        return cx, cy

    def _bfs_from(self, gx: int, gy: int) -> np.ndarray:
        key = (gx, gy)
        if key in self._dist_cache:
            return self._dist_cache[key]
        dist = np.full((self.w, self.h), np.inf, dtype=np.float32)
        q: deque[tuple[int, int]] = deque([(gx, gy)])
        dist[gx, gy] = 0.0
        w, h, walk = self.w, self.h, self.walkable
        while q:
            cx, cy = q.popleft()
            d_next = dist[cx, cy] + GRID_CELL
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < w and 0 <= ny < h and walk[nx, ny] and d_next < dist[nx, ny]:
                    dist[nx, ny] = d_next
                    q.append((nx, ny))
        self._dist_cache[key] = dist
        return dist

    def geodesic_query(self, px: float, py: float, heading: float, target_xy: tuple[float, float]) -> tuple[float, float, float]:
        """Return (geodesic_dist, sin_ego_bearing, cos_ego_bearing) toward target_xy along walkable corridors."""
        tcx, tcy = self.world_to_cell(target_xy[0], target_xy[1])
        field = self._bfs_from(tcx, tcy)
        cx, cy = self.world_to_cell(px, py)
        d_here = float(field[cx, cy])
        if not math.isfinite(d_here):
            # Fallback to straight-line vector if cell not connected in coarse grid
            dx, dy = target_xy[0] - px, target_xy[1] - py
            dist = math.hypot(dx, dy)
            ang = math.atan2(dy, dx) - heading
            return float(dist), math.sin(ang), math.cos(ang)

        # Follow steepest descent neighbor in geodesic field for exact around-wall compass
        best_nx, best_ny, best_d = cx, cy, d_here
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < self.w and 0 <= ny < self.h and field[nx, ny] < best_d:
                best_d = float(field[nx, ny])
                best_nx, best_ny = nx, ny
        if (best_nx, best_ny) == (cx, cy):
            wx, wy = target_xy
            d_cont = math.hypot(wx - px, wy - py)
        else:
            wx = self.ox + (best_nx + 0.5) * GRID_CELL
            wy = self.oy + (best_ny + 0.5) * GRID_CELL
            d_cont = best_d + math.hypot(wx - px, wy - py)
        ang = math.atan2(wy - py, wx - px) - heading
        return float(d_cont), math.sin(ang), math.cos(ang)

    def ego_bev_ground_truth(self, px: float, py: float, heading: float, visited_cells: set[tuple[int, int]]) -> np.ndarray:
        """Render (4, 64, 64) ego-centric ground-truth BEV occupancy: [walkable, wall, exit_zone, visited]."""
        # Grid coords in ego frame: row r -> forward (+x_ego), col c -> right (+y_ego)
        idx = (np.arange(BEV_SIZE, dtype=np.float32) - (BEV_SIZE / 2.0 - 0.5)) * BEV_RES
        # Forward increases with row index, right increases with col index
        fwd, right = np.meshgrid(idx, idx, indexing="ij")
        c, s = math.cos(heading), math.sin(heading)
        # Inverse of ego_rotate: wx = px + fwd*c + right*s, wy = py + fwd*s - right*c
        wx = px + fwd * c + right * s
        wy = py + fwd * s - right * c
        gx = np.floor((wx - self.ox) / GRID_CELL).astype(np.int32)
        gy = np.floor((wy - self.oy) / GRID_CELL).astype(np.int32)
        inb = (gx >= 0) & (gx < self.w) & (gy >= 0) & (gy < self.h)

        bev = np.zeros((4, BEV_SIZE, BEV_SIZE), dtype=np.float32)
        walk = np.zeros((BEV_SIZE, BEV_SIZE), dtype=np.float32)
        walk[inb] = self.walkable[gx[inb], gy[inb]].astype(np.float32)
        bev[0] = walk
        bev[1] = 1.0 - walk

        ex_xy = self.meta.get("exit_pos")
        if ex_xy is not None:
            d_ex = np.hypot(wx - ex_xy[0], wy - ex_xy[1])
            bev[2] = (d_ex <= 96.0).astype(np.float32)

        if visited_cells:
            vcx = np.floor(wx / 128.0).astype(np.int32)
            vcy = np.floor(wy / 128.0).astype(np.int32)
            # Vectorized lookup via bounding mask
            for vx, vy in visited_cells:
                bev[3] = np.maximum(bev[3], ((vcx == vx) & (vcy == vy)).astype(np.float32))
        return bev


def _point_in_segs(px: float, py: float, segs: np.ndarray) -> bool:
    x1, y1, x2, y2 = segs[:, 0], segs[:, 1], segs[:, 2], segs[:, 3]
    cond = (y1 > py) != (y2 > py)
    if not np.any(cond):
        return False
    x1c, y1c, x2c, y2c = x1[cond], y1[cond], x2[cond], y2[cond]
    x_int = x1c + (py - y1c) * (x2c - x1c) / (y2c - y1c)
    return bool(np.count_nonzero(px < x_int) % 2 == 1)


def _frustum_affordance_20col(px: float, py: float, heading: float, meta: dict) -> np.ndarray:
    """Project known exit / switch / lock_color points into 20 horizontal 90-deg FOV columns (20, 5)."""
    # Channels: 0=exit, 1=switch, 2=lock_red, 3=lock_blue, 4=lock_yellow
    out = np.zeros((20, 5), dtype=np.float32)

    def _mark(tx: float, ty: float, ch: int, max_d: float = 1024.0) -> None:
        dx, dy = tx - px, ty - py
        d = math.hypot(dx, dy)
        if d < 1e-3 or d > max_d:
            return
        rel = (math.atan2(dy, dx) - heading + math.pi) % (2.0 * math.pi) - math.pi
        if abs(rel) > math.radians(45.0):
            return
        # rel = +45 deg is left (col 0), rel = -45 deg is right (col 19)
        u_frac = 0.5 - rel / math.radians(90.0)
        col = int(np.clip(round(u_frac * 19.0), 0, 19))
        out[col, ch] = 1.0

    ex = meta.get("exit_pos")
    if ex is not None:
        _mark(ex[0], ex[1], 0)
    for sw in meta.get("switches", []):
        _mark(sw["pos"][0], sw["pos"][1], 1)
    lock_to_ch = {129: 2, 130: 3, 131: 4}
    for ld in meta.get("locked_doors", []):
        ch = lock_to_ch.get(ld["lock"])
        if ch is not None:
            _mark(ld["pos"][0], ld["pos"][1], ch)
    return out


class FastDoomPackEnv:
    """Fast multi-map training wrapper over a packed WAD shard with Geodesic Oracle & Privileged Labels."""

    def __init__(
        self,
        wad_path: str | Path,
        *,
        episode_ticks: int = 2100,
        render_buffers: bool = True,
        gamma: float = 0.995,
        compute_spatial_targets: bool = True,
    ) -> None:
        self.wad_path = Path(wad_path).resolve()
        self.episode_ticks = int(episode_ticks)
        self.render_buffers = bool(render_buffers)
        self.gamma = float(gamma)
        self.compute_spatial_targets = bool(compute_spatial_targets)

        lumps = read_wad(self.wad_path)
        self.map_names = list_maps(lumps)
        if not self.map_names:
            raise ValueError(f"No maps found in {self.wad_path}")

        manifest_path = self.wad_path.with_suffix(".json")
        self.manifest: dict[str, dict] = {}
        if manifest_path.is_file():
            for entry in json.loads(manifest_path.read_text(encoding="utf-8")):
                self.manifest[entry["map_name"].upper()] = entry
        else:
            # Auto-extract keys & exit positions from binary Doom WADs (e.g. freedoom2.wad!)
            key_lock_map = {13: 129, 38: 129, 5: 130, 40: 130, 6: 131, 39: 131}
            for m in self.map_names:
                ml = map_lumps(lumps, m)
                keys_placed = []
                th = ml.get("THINGS", b"")
                for i in range(0, len(th) - 9, 10):
                    tx, ty, _ang, tid, _fl = struct.unpack_from("<hhHHH", th, i)
                    if tid in key_lock_map:
                        keys_placed.append({"lock": key_lock_map[tid], "pos": [float(tx), float(ty)]})
                ex_pos = None
                ex_type = "switch"
                ld = ml.get("LINEDEFS", b"")
                vtx = ml.get("VERTEXES", b"")
                for i in range(0, len(ld) - 13, 14):
                    v1, v2, _fl, spec, _tag, _f, _b = struct.unpack_from("<HHHHhHH", ld, i)
                    if spec in (11, 51, 52, 124) and v1 * 4 + 4 <= len(vtx) and v2 * 4 + 4 <= len(vtx):
                        x1, y1 = struct.unpack_from("<hh", vtx, v1 * 4)
                        x2, y2 = struct.unpack_from("<hh", vtx, v2 * 4)
                        ex_pos = [0.5 * (x1 + x2), 0.5 * (y1 + y2)]
                        ex_type = "walkover" if spec in (52, 124) else "switch"
                        break
                self.manifest[m.upper()] = {"map_name": m.upper(), "keys_placed": keys_placed, "exit_pos": ex_pos, "exit_type": ex_type}

        # Filter out maps >= 2048 sectors (ViZDoom sector reporting limit)
        valid_maps = []
        self._mapinfos = {}
        for m in self.map_names:
            mi = load_mapinfo(self.wad_path, m)
            if mi.num_sectors < 2048:
                valid_maps.append(m)
                self._mapinfos[m] = mi
        if not valid_maps:
            raise ValueError(f"No <2048-sector maps found in {self.wad_path}")
        self.map_names = valid_maps
        self._fields = {m: GridGeodesicField(self._mapinfos[m], self.manifest.get(m)) for m in self.map_names}

        self.current_map = self.map_names[0]
        self.env = make_env(self.wad_path, self.current_map, episode_ticks=episode_ticks, render_buffers=render_buffers)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        # Episode trackers
        self._prev_priv: dict | None = None
        self._prev_phi: float = 0.0
        self._visited_cells: set[tuple[int, int]] = set()
        self._opened_doors: set[int] = set()
        self._unlocked_doors: set[int] = set()
        self._rng = np.random.default_rng(0)

    def reset(self, *, map_name: str | None = None, seed: int | None = None) -> tuple[dict, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        target_map = (map_name or str(self._rng.choice(self.map_names))).upper()
        if target_map != self.current_map:
            self.current_map = target_map
            self.env.map_name = target_map
            self.env.mapinfo = self._mapinfos[target_map]
            self.env._game.set_doom_map(target_map)
            self.env._geometry = None  # Force fresh LevelGeometry on reset()
        obs, info = self.env.reset(seed=int(self._rng.integers(0, 2**31 - 1)))
        priv = self.env.privileged_state()
        self._prev_priv = priv
        self._visited_cells = {(int(priv["x"] // 128), int(priv["y"] // 128))}
        self._opened_doors = set()
        self._unlocked_doors = set()
        self._prev_phi = self._compute_phi(priv)
        return obs, self._build_oracle_info(priv, info, odom_delta=(0.0, 0.0, 0.0, 1.0))

    def _active_subgoal(self, priv: dict) -> tuple[float, float]:
        meta = self.manifest.get(self.current_map, {})
        keys_held = priv["keys"]  # [blue_c, yellow_c, red_c, blue_s, yellow_s, red_s]
        lock_held = {
            130: bool(keys_held[0] > 0.5 or keys_held[3] > 0.5),  # Blue
            131: bool(keys_held[1] > 0.5 or keys_held[4] > 0.5),  # Yellow
            129: bool(keys_held[2] > 0.5 or keys_held[5] > 0.5),  # Red
        }
        for k in meta.get("keys_placed", []):
            if not lock_held.get(k["lock"], False):
                return tuple(k["pos"])
        ex = meta.get("exit_pos")
        if ex is not None:
            return tuple(ex)
        return (priv["x"], priv["y"])

    def _compute_phi(self, priv: dict) -> float:
        field = self._fields[self.current_map]
        heading = math.radians(priv["angle"])
        d_geo, _, _ = field.geodesic_query(priv["x"], priv["y"], heading, self._active_subgoal(priv))
        # Non-negative potential increasing toward goal so (gamma - 1) * phi <= 0 (zero standing-still reward hack!)
        return max(0.0, (4096.0 - d_geo) / 256.0)

    def _build_oracle_info(self, priv: dict, base_info: dict, odom_delta: tuple[float, float, float, float]) -> dict:
        field = self._fields[self.current_map]
        heading = math.radians(priv["angle"])
        subgoal = self._active_subgoal(priv)
        d_geo, sin_g, cos_g = field.geodesic_query(priv["x"], priv["y"], heading, subgoal)
        meta = self.manifest.get(self.current_map, {})

        # Compact 24-dim privileged vector for Asymmetric Critic & Omniscient Teacher
        priv_vec = np.zeros(24, dtype=np.float32)
        priv_vec[0] = min(d_geo / 1024.0, 4.0)
        priv_vec[1] = sin_g
        priv_vec[2] = cos_g
        priv_vec[3] = priv["health"] / 100.0
        priv_vec[4] = float(priv["kills"]) / 10.0
        priv_vec[5] = float(len(self._visited_cells)) / 30.0
        priv_vec[6] = float(priv["tic"]) / max(1.0, float(self.episode_ticks))
        priv_vec[7:13] = priv["keys"]
        priv_vec[13:17] = np.asarray(odom_delta, dtype=np.float32)
        priv_vec[17] = 1.0 if meta.get("exit_type") == "switch" else 0.0
        priv_vec[18] = float(len(self._opened_doors)) / 5.0
        priv_vec[19] = float(len(self._unlocked_doors)) / 3.0

        out = {
            **base_info,
            "map_name": self.current_map,
            "priv_vec": priv_vec,
            "odom_target": np.asarray(odom_delta, dtype=np.float32),
            "geodesic_dist": d_geo,
        }
        if self.compute_spatial_targets:
            out["afford_target"] = _frustum_affordance_20col(priv["x"], priv["y"], heading, meta)
            out["bev_target"] = field.ego_bev_ground_truth(priv["x"], priv["y"], heading, self._visited_cells)
        return out

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        obs, _raw_r, term, trunc, info = self.env.step(action)
        priv = self.env.privileged_state()
        prev = self._prev_priv
        assert prev is not None

        # Exact ego-frame odometry delta from prev -> cur
        prev_h = math.radians(prev["angle"])
        dx_w = priv["x"] - prev["x"]
        dy_w = priv["y"] - prev["y"]
        dx_ego, dy_ego = ego_rotate(dx_w, dy_w, prev_h)
        dtheta = math.radians((priv["angle"] - prev["angle"] + 180.0) % 360.0 - 180.0)
        odom_delta = (float(dx_ego), float(dy_ego), math.sin(dtheta), math.cos(dtheta))

        # Scorer-mirrored event rewards + exact telescoping geodesic progress shaping
        reward = -0.002  # mild time pressure (-1.05 at 525 steps, strictly better than -4.0 death)
        if priv["exit_reached"]:
            hp_bonus = 2.5 * min(1.0, max(0.0, priv["health"]) / 100.0)
            time_bonus = 2.5 * max(0.0, 1.0 - priv["tic"] / max(1, self.episode_ticks))
            reward += 15.0 + hp_bonus + time_bonus
        if priv["dead"]:
            reward -= 4.0

        new_keys = int(np.sum(priv["keys"] > 0.5) - np.sum(prev["keys"] > 0.5))
        if new_keys > 0:
            reward += 3.5 * new_keys

        # Door opens / unlocks
        floor, ceil, locks = priv["sector_floor"], priv["sector_ceiling"], priv["door_lock"]
        if floor is not None and ceil is not None and locks is not None:
            open_mask = (ceil - floor) >= 56.0
            for sid in np.flatnonzero(open_mask & (locks == 0)):
                if int(sid) not in self._opened_doors:
                    self._opened_doors.add(int(sid))
                    reward += 1.0
            for sid in np.flatnonzero(open_mask & (locks > 0)):
                if int(sid) not in self._unlocked_doors:
                    self._unlocked_doors.add(int(sid))
                    reward += 4.0

        # Exploration cell coverage
        cell = (int(priv["x"] // 128), int(priv["y"] // 128))
        if cell not in self._visited_cells:
            self._visited_cells.add(cell)
            reward += 0.10

        # Combat & damage
        dkills = priv["kills"] - prev["kills"]
        if dkills > 0:
            reward += 0.8 * dkills
        ddmg = max(0.0, priv["damage_taken"] - prev["damage_taken"])
        if ddmg > 0:
            reward -= 0.02 * ddmg

        # Telescoping continuous geodesic progress (zero camping hack, zero suicide hack, zero loop farming!)
        phi = self._compute_phi(priv)
        if new_keys == 0 and not term:
            reward += 1.50 * (phi - self._prev_phi)
        self._prev_phi = phi
        self._prev_priv = priv

        return obs, float(reward), term, trunc, self._build_oracle_info(priv, info, odom_delta)

    def close(self) -> None:
        self.env.close()
