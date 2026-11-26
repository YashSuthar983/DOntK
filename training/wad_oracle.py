"""Training-Only WAD Geodesic Oracle, Fog-of-War Grounding Labeler & Fast Multi-Map Env.

NEVER imported by submission/agent.py. Used exclusively during training for:
1. Fast <2 ms map switching across packed DoomForge / Freedoom2 WADs.
2. Wall-aware 64-unit walkable grid BFS (`GridGeodesicField`) supporting dynamic
   door/lock/remote-switch passability on BOTH procedural UDMF & real Doom maps.
3. Exact ray-swept Fog-of-War tracking (`seen_cells`) so BEV grounding loss ONLY
   supervises observed geometry (zero hallucination behind walls) and the
   Fog-of-War Frontier Oracle has zero information asymmetry with the student!
4. Fast vectorized 32x32 Ego-BEV targets (6 channels) + 16-col visual affordances
   + scorer-aligned survival/progression/combat reward shaping.
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
from doom_env.mapinfo import _sector_tags, load_mapinfo, parse_udmf
from doom_env.sensors import LevelGeometry, PLAYER_HEIGHT, _ray_line_t, ego_rotate
from doom_env.wad import Lump, list_maps, map_lumps, read_wad

GRID_CELL = 64.0
BEV_SIZE = 32
BEV_RES = 64.0  # 32 * 64 = 2048 map units (-1024..+1024 around player)
AFFORD_COLS = 16


def _point_in_segs(px: float, py: float, segs: np.ndarray) -> bool:
    x1, y1, x2, y2 = segs[:, 0], segs[:, 1], segs[:, 2], segs[:, 3]
    cond = (y1 > py) != (y2 > py)
    if not np.any(cond):
        return False
    x1c, y1c, x2c, y2c = x1[cond], y1[cond], x2[cond], y2[cond]
    x_int = x1c + (py - y1c) * (x2c - x1c) / (y2c - y1c)
    return bool(np.count_nonzero(px < x_int) % 2 == 1)


def _segments_intersect_batch(
    a1: np.ndarray, a2: np.ndarray, p1: np.ndarray, p2: np.ndarray
) -> bool:
    """Does segment a1->a2 intersect any segment in (p1, p2) (M, 2)?"""
    if len(p1) == 0:
        return False
    r = a2 - a1
    d = p2 - p1
    w = p1 - a1
    denom = r[0] * d[:, 1] - r[1] * d[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (w[:, 0] * d[:, 1] - w[:, 1] * d[:, 0]) / denom
        u = (w[:, 0] * r[1] - w[:, 1] * r[0]) / denom
    hit = (np.abs(denom) > 1e-9) & (t > 1e-4) & (t < 1.0 - 1e-4) & (u >= -1e-4) & (u <= 1.0 + 1e-4)
    return bool(hit.any())


class GridGeodesicField:
    """Wall-aware 64-unit walkable grid extracted from WAD LineGeometry + Door metadata."""

    def __init__(self, mapinfo, meta: dict | None = None) -> None:
        self.meta = meta or {}
        g = mapinfo.lines
        if g is None or len(g.p1) == 0:
            self.ox, self.oy, self.w, self.h = 0.0, 0.0, 1, 1
            self.walkable = np.ones((1, 1), dtype=bool)
            self.cell_sector = np.full((1, 1), -1, dtype=np.int32)
            self.pass_e = np.zeros((1, 1), dtype=bool)
            self.pass_w = np.zeros((1, 1), dtype=bool)
            self.pass_n = np.zeros((1, 1), dtype=bool)
            self.pass_s = np.zeros((1, 1), dtype=bool)
        else:
            all_pts = np.concatenate([g.p1, g.p2], axis=0)
            xmin, ymin = all_pts.min(axis=0)
            xmax, ymax = all_pts.max(axis=0)
            self.ox = float(math.floor(xmin / GRID_CELL) * GRID_CELL)
            self.oy = float(math.floor(ymin / GRID_CELL) * GRID_CELL)
            self.w = max(1, min(160, int(math.ceil((xmax - self.ox) / GRID_CELL))))
            self.h = max(1, min(160, int(math.ceil((ymax - self.oy) / GRID_CELL))))
            self.walkable = np.zeros((self.w, self.h), dtype=bool)
            self.cell_sector = np.full((self.w, self.h), -1, dtype=np.int32)

            for sid, segs in mapinfo.sector_lines().items():
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
                            self.cell_sector[cx, cy] = sid
            if not self.walkable.any():
                self.walkable[:] = True

            # Solid wall linedefs: 1-sided or impassable (exclude door sectors so paths can cross doors)
            one_sided = (g.front < 0) | (g.back < 0)
            solid_mask = one_sided | g.impassable
            solid_p1 = g.p1[solid_mask]
            solid_p2 = g.p2[solid_mask]
            sb_min = np.minimum(solid_p1, solid_p2) if len(solid_p1) else np.zeros((0, 2))
            sb_max = np.maximum(solid_p1, solid_p2) if len(solid_p1) else np.zeros((0, 2))

            # Optional static floor heights from manifest dropoff_cells (refined dynamically with live sector_floor!)
            static_floor = np.zeros((self.w, self.h), dtype=np.float32)
            for dc in self.meta.get("dropoff_cells", []):
                dcx, dcy = self.world_to_cell((dc[0] + 0.5) * GRID_CELL, (dc[1] + 0.5) * GRID_CELL)
                static_floor[dcx, dcy] = 48.0

            self.pass_e = np.zeros((self.w, self.h), dtype=bool)
            self.pass_w = np.zeros((self.w, self.h), dtype=bool)
            self.pass_n = np.zeros((self.w, self.h), dtype=bool)
            self.pass_s = np.zeros((self.w, self.h), dtype=bool)
            for cx in range(self.w):
                px = self.ox + (cx + 0.5) * GRID_CELL
                for cy in range(self.h):
                    if not self.walkable[cx, cy]:
                        continue
                    py = self.oy + (cy + 0.5) * GRID_CELL
                    # Check East-West pair (cx, cy) <-> (cx+1, cy)
                    if cx + 1 < self.w and self.walkable[cx + 1, cy]:
                        nx_w = px + GRID_CELL
                        wall_hit = False
                        if len(solid_p1) > 0:
                            cand = (
                                (sb_min[:, 0] <= nx_w)
                                & (sb_max[:, 0] >= px)
                                & (sb_min[:, 1] <= py + 2.0)
                                & (sb_max[:, 1] >= py - 2.0)
                            )
                            wall_hit = _segments_intersect_batch(
                                np.array([px, py]), np.array([nx_w, py]), solid_p1[cand], solid_p2[cand]
                            )
                        if not wall_hit:
                            if static_floor[cx + 1, cy] - static_floor[cx, cy] <= 24.0:
                                self.pass_e[cx, cy] = True
                            if static_floor[cx, cy] - static_floor[cx + 1, cy] <= 24.0:
                                self.pass_w[cx + 1, cy] = True
                    # Check North-South pair (cx, cy) <-> (cx, cy+1)
                    if cy + 1 < self.h and self.walkable[cx, cy + 1]:
                        ny_w = py + GRID_CELL
                        wall_hit = False
                        if len(solid_p1) > 0:
                            cand = (
                                (sb_min[:, 0] <= px + 2.0)
                                & (sb_max[:, 0] >= px - 2.0)
                                & (sb_min[:, 1] <= ny_w)
                                & (sb_max[:, 1] >= py)
                            )
                            wall_hit = _segments_intersect_batch(
                                np.array([px, py]), np.array([px, ny_w]), solid_p1[cand], solid_p2[cand]
                            )
                        if not wall_hit:
                            if static_floor[cx, cy + 1] - static_floor[cx, cy] <= 24.0:
                                self.pass_n[cx, cy] = True
                            if static_floor[cx, cy] - static_floor[cx, cy + 1] <= 24.0:
                                self.pass_s[cx, cy + 1] = True

        # Map door sectors to grid cells
        self.door_sectors: dict[int, int] = dict(mapinfo.door_sectors)
        self.cell_door_sector = np.full((self.w, self.h), -1, dtype=np.int32)
        self.door_cells: dict[int, list[tuple[int, int]]] = {}
        for cx in range(self.w):
            for cy in range(self.h):
                sid = int(self.cell_sector[cx, cy])
                if sid in self.door_sectors:
                    self.cell_door_sector[cx, cy] = sid
                    self.door_cells.setdefault(sid, []).append((cx, cy))

        # Precompute static cell masks for fast BEV rendering
        self.locked_door_grid = np.zeros((self.w, self.h), dtype=np.float32)
        for sid, lock in self.door_sectors.items():
            if lock > 0:
                for cx, cy in self.door_cells.get(sid, []):
                    self.locked_door_grid[cx, cy] = 1.0
        for ld in self.meta.get("locked_doors", []):
            lcx, lcy = self.world_to_cell(ld["pos"][0], ld["pos"][1])
            self.locked_door_grid[lcx, lcy] = 1.0

        self.exit_switch_grid = np.zeros((self.w, self.h), dtype=np.float32)
        ex_xy = self.meta.get("exit_pos")
        if ex_xy is not None:
            ecx, ecy = self.world_to_cell(ex_xy[0], ex_xy[1])
            self.exit_switch_grid[ecx, ecy] = 1.0
        for sw in self.meta.get("switches", []):
            scx, scy = self.world_to_cell(sw["pos"][0], sw["pos"][1])
            self.exit_switch_grid[scx, scy] = 1.0

        self._dist_cache: dict[tuple[int, int], np.ndarray] = {}

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        cx = int(np.clip(math.floor((x - self.ox) / GRID_CELL), 0, self.w - 1))
        cy = int(np.clip(math.floor((y - self.oy) / GRID_CELL), 0, self.h - 1))
        return cx, cy

    def cell_center(self, cx: int, cy: int) -> tuple[float, float]:
        return (self.ox + (cx + 0.5) * GRID_CELL, self.oy + (cy + 0.5) * GRID_CELL)

    def neighbors(self, cx: int, cy: int) -> list[tuple[int, int]]:
        out = []
        if cx + 1 < self.w and self.pass_e[cx, cy]:
            out.append((cx + 1, cy))
        if cx - 1 >= 0 and self.pass_w[cx, cy]:
            out.append((cx - 1, cy))
        if cy + 1 < self.h and self.pass_n[cx, cy]:
            out.append((cx, cy + 1))
        if cy - 1 >= 0 and self.pass_s[cx, cy]:
            out.append((cx, cy - 1))
        return out

    def _bfs_from(self, gx: int, gy: int, blocked_grid: np.ndarray | None = None) -> np.ndarray:
        """Reverse BFS from goal (gx, gy): dist[u] = shortest directed path from u -> goal."""
        b_key = None if blocked_grid is None else (blocked_grid.tobytes() if blocked_grid.any() else b"")
        key = (gx, gy, b_key)
        cached = self._dist_cache.get(key)
        if cached is not None:
            return cached
        dist = np.full((self.w, self.h), np.inf, dtype=np.float32)
        q: deque[tuple[int, int]] = deque([(gx, gy)])
        dist[gx, gy] = 0.0
        w, h = self.w, self.h
        pe, pw, pn, ps = self.pass_e, self.pass_w, self.pass_n, self.pass_s
        while q:
            cx, cy = q.popleft()
            d_next = dist[cx, cy] + GRID_CELL
            if cx + 1 < w and pw[cx + 1, cy] and d_next < dist[cx + 1, cy]:
                if blocked_grid is None or not blocked_grid[cx + 1, cy]:
                    dist[cx + 1, cy] = d_next
                    q.append((cx + 1, cy))
            if cx - 1 >= 0 and pe[cx - 1, cy] and d_next < dist[cx - 1, cy]:
                if blocked_grid is None or not blocked_grid[cx - 1, cy]:
                    dist[cx - 1, cy] = d_next
                    q.append((cx - 1, cy))
            if cy + 1 < h and ps[cx, cy + 1] and d_next < dist[cx, cy + 1]:
                if blocked_grid is None or not blocked_grid[cx, cy + 1]:
                    dist[cx, cy + 1] = d_next
                    q.append((cx, cy + 1))
            if cy - 1 >= 0 and pn[cx, cy - 1] and d_next < dist[cx, cy - 1]:
                if blocked_grid is None or not blocked_grid[cx, cy - 1]:
                    dist[cx, cy - 1] = d_next
                    q.append((cx, cy - 1))
        if len(self._dist_cache) > 512:
            self._dist_cache.clear()
        self._dist_cache[key] = dist
        return dist

    def bfs_from_sources(
        self,
        sources: list[tuple[int, int]],
        blocked_grid: np.ndarray | None = None,
    ) -> np.ndarray:
        """Forward multi-source BFS respecting walls, one-way drop-offs & optional blocked cells."""
        dist = np.full((self.w, self.h), np.inf, dtype=np.float32)
        q: deque[tuple[int, int]] = deque()
        for sx, sy in sources:
            if 0 <= sx < self.w and 0 <= sy < self.h and self.walkable[sx, sy]:
                if blocked_grid is None or not blocked_grid[sx, sy]:
                    dist[sx, sy] = 0.0
                    q.append((sx, sy))
        w, h = self.w, self.h
        pe, pw, pn, ps = self.pass_e, self.pass_w, self.pass_n, self.pass_s
        while q:
            cx, cy = q.popleft()
            d_next = dist[cx, cy] + GRID_CELL
            if cx + 1 < w and pe[cx, cy] and d_next < dist[cx + 1, cy]:
                if blocked_grid is None or not blocked_grid[cx + 1, cy]:
                    dist[cx + 1, cy] = d_next
                    q.append((cx + 1, cy))
            if cx - 1 >= 0 and pw[cx, cy] and d_next < dist[cx - 1, cy]:
                if blocked_grid is None or not blocked_grid[cx - 1, cy]:
                    dist[cx - 1, cy] = d_next
                    q.append((cx - 1, cy))
            if cy + 1 < h and pn[cx, cy] and d_next < dist[cx, cy + 1]:
                if blocked_grid is None or not blocked_grid[cx, cy + 1]:
                    dist[cx, cy + 1] = d_next
                    q.append((cx, cy + 1))
            if cy - 1 >= 0 and ps[cx, cy] and d_next < dist[cx, cy - 1]:
                if blocked_grid is None or not blocked_grid[cx, cy - 1]:
                    dist[cx, cy - 1] = d_next
                    q.append((cx, cy - 1))
        return dist

    def geodesic_query(
        self,
        px: float,
        py: float,
        heading: float,
        target_xy: tuple[float, float],
        blocked_grid: np.ndarray | None = None,
    ) -> tuple[float, float, float]:
        """Return (geodesic_dist, sin_ego_bearing, cos_ego_bearing) toward target_xy along walkable corridors."""
        tcx, tcy = self.world_to_cell(target_xy[0], target_xy[1])
        field = self._bfs_from(tcx, tcy, blocked_grid=blocked_grid)
        cx, cy = self.world_to_cell(px, py)
        d_here = float(field[cx, cy])
        if not math.isfinite(d_here):
            # Try adjacent cell if player clipped near wall
            best_adj = (cx, cy, np.inf)
            for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                if 0 <= nx < self.w and 0 <= ny < self.h and field[nx, ny] < best_adj[2]:
                    best_adj = (nx, ny, float(field[nx, ny]))
            if math.isfinite(best_adj[2]):
                cx, cy, d_here = best_adj
            else:
                dx, dy = target_xy[0] - px, target_xy[1] - py
                dist = math.hypot(dx, dy)
                ang = math.atan2(dy, dx) - heading
                return float(dist), math.sin(ang), math.cos(ang)

        # Follow wall-respecting neighbor with smallest distance + 1-step lookahead for smooth gliding!
        best_nx, best_ny, best_d = cx, cy, d_here
        for nx, ny in self.neighbors(cx, cy):
            if field[nx, ny] < best_d:
                best_d = float(field[nx, ny])
                best_nx, best_ny = nx, ny
        if (best_nx, best_ny) == (cx, cy):
            wx, wy = target_xy
            d_cont = math.hypot(wx - px, wy - py)
        else:
            # Look 1 more step along steepest descent if collinear (straight corridor -> fast glide)
            n2x, n2y, d2 = best_nx, best_ny, best_d
            for nx, ny in self.neighbors(best_nx, best_ny):
                if field[nx, ny] < d2:
                    d2 = float(field[nx, ny])
                    n2x, n2y = nx, ny
            if (n2x - best_nx == best_nx - cx) and (n2y - best_ny == best_ny - cy):
                wx, wy = self.cell_center(n2x, n2y)
            else:
                wx, wy = self.cell_center(best_nx, best_ny)
            d_cont = best_d + math.hypot(self.cell_center(best_nx, best_ny)[0] - px, self.cell_center(best_nx, best_ny)[1] - py)
        ang = math.atan2(wy - py, wx - px) - heading
        return float(d_cont), math.sin(ang), math.cos(ang)

    def ego_bev_ground_truth(
        self,
        px: float,
        py: float,
        heading: float,
        seen_grid: np.ndarray,
        visited_grid: np.ndarray,
    ) -> np.ndarray:
        """Fast vectorized (6, 32, 32) Ego-BEV target:
        [0: observed_mask, 1: walkable, 2: wall, 3: locked_door, 4: exit_or_switch, 5: visited].
        """
        idx = (np.arange(BEV_SIZE, dtype=np.float32) - (BEV_SIZE / 2.0 - 0.5)) * BEV_RES
        fwd, right = np.meshgrid(idx, idx, indexing="ij")
        c, s = math.cos(heading), math.sin(heading)
        wx = px + fwd * c + right * s
        wy = py + fwd * s - right * c
        gx = np.floor((wx - self.ox) / GRID_CELL).astype(np.int32)
        gy = np.floor((wy - self.oy) / GRID_CELL).astype(np.int32)
        inb = (gx >= 0) & (gx < self.w) & (gy >= 0) & (gy < self.h)

        bev = np.zeros((6, BEV_SIZE, BEV_SIZE), dtype=np.float32)
        gxi, gyi = gx[inb], gy[inb]
        bev[0][inb] = seen_grid[gxi, gyi]
        walk = self.walkable[gxi, gyi].astype(np.float32)
        bev[1][inb] = walk
        bev[2][inb] = 1.0 - walk
        bev[3][inb] = self.locked_door_grid[gxi, gyi]
        bev[4][inb] = self.exit_switch_grid[gxi, gyi]
        bev[5][inb] = visited_grid[gxi, gyi]
        return bev


def _frustum_affordance(
    px: float, py: float, heading: float, meta: dict, n_cols: int = AFFORD_COLS, *,
    geometry: LevelGeometry | None = None, field: GridGeodesicField | None = None,
) -> np.ndarray:
    """Project known exit / switch / lock_color points into horizontal 90-deg FOV columns (n_cols, 5)."""
    # Channels: 0=exit, 1=switch, 2=lock_red, 3=lock_blue, 4=lock_yellow
    out = np.zeros((n_cols, 5), dtype=np.float32)
    candidates = []

    def _mark(tx: float, ty: float, ch: int, visible_door: int = -1) -> None:
        dx, dy = tx - px, ty - py
        d = math.hypot(dx, dy)
        if d < 1e-3 or d > S.RAY_RANGE:
            return
        rel = (math.atan2(dy, dx) - heading + math.pi) % (2.0 * math.pi) - math.pi
        if abs(rel) > math.radians(45.0):
            return
        u_frac = 0.5 - rel / math.radians(90.0)
        col = int(np.clip(round(u_frac * (n_cols - 1)), 0, n_cols - 1))
        candidates.append((col, ch, d, dx / d, dy / d, visible_door))

    ex = meta.get("exit_pos")
    if ex is not None:
        _mark(ex[0], ex[1], 0)
    for sw in meta.get("switches", []):
        _mark(sw["pos"][0], sw["pos"][1], 1)
    lock_to_ch = {129: 2, 1: 2, 130: 3, 2: 3, 131: 4, 3: 4}
    for ld in meta.get("locked_doors", []):
        ch = lock_to_ch.get(ld["lock"])
        if ch is not None:
            tx, ty = ld["pos"]
            sid = int(field.cell_sector[field.world_to_cell(tx, ty)]) if field is not None else -1
            target_door = sid if field is not None and field.door_sectors.get(sid, 0) > 0 else -1
            _mark(tx, ty, ch, visible_door=target_door)

    if not candidates:
        return out
    if geometry is None:
        for col, ch, *_ in candidates:
            out[col, ch] = 1.0
        return out

    lines = geometry.nearby(px, py, max(candidate[2] for candidate in candidates))
    lines = lines[geometry.sight_block[lines]]
    distances = _ray_line_t(
        np.array([px, py]), np.array([(c[3], c[4]) for c in candidates]),
        geometry.p1[lines], geometry.p2[lines],
    )
    # A generated lock is labeled at its center. Its own front face is visible.
    for (col, ch, d, _, _, target_door), hits in zip(candidates, distances):
        if not len(hits) or hits.min() >= d - 1e-4:
            out[col, ch] = 1.0
        elif target_door >= 0:
            line = lines[hits.argmin()]
            if target_door in (geometry.sec_a[line], geometry.sec_b[line]):
                out[col, ch] = 1.0
    return out


def _udmf_sector_tags(blocks: dict) -> dict[int, list[int]]:
    tags: dict[int, list[int]] = {}
    for sid, sector in enumerate(blocks.get("sector", [])):
        tag = int(sector.get("id", 0))
        if tag:
            tags.setdefault(tag, []).append(sid)
    return tags


def _auto_extract_manifest(lumps: list[Lump], map_names: list[str]) -> dict[str, dict]:
    """Auto-extract keys, switches, locked doors & exits from UDMF or binary Doom WADs."""
    key_lock_map = {13: 129, 38: 129, 5: 130, 40: 130, 6: 131, 39: 131}
    manifest: dict[str, dict] = {}
    for m in map_names:
        ml = map_lumps(lumps, m)
        keys_placed: list[dict] = []
        switches: list[dict] = []
        locked_doors: list[dict] = []
        ex_pos = None
        ex_type = "switch"

        tagged_sectors: dict[int, list[int]] = {}
        if "TEXTMAP" in ml:
            b = parse_udmf(ml["TEXTMAP"].decode("utf-8", "replace"))
            tagged_sectors = _udmf_sector_tags(b)
            verts = [(float(v.get("x", 0.0)), float(v.get("y", 0.0))) for v in b.get("vertex", [])]
            for th in b.get("thing", []):
                tid = int(th.get("type", 0) or 0)
                if tid in key_lock_map:
                    keys_placed.append({
                        "lock": key_lock_map[tid],
                        "pos": [float(th.get("x", 0.0)), float(th.get("y", 0.0))],
                    })
            for ln in b.get("linedef", []):
                spec = int(ln.get("special", 0) or 0)
                v1, v2 = int(ln.get("v1", -1)), int(ln.get("v2", -1))
                if not (0 <= v1 < len(verts) and 0 <= v2 < len(verts)):
                    continue
                mx = 0.5 * (verts[v1][0] + verts[v2][0])
                my = 0.5 * (verts[v1][1] + verts[v2][1])
                if spec in (243, 244, 11, 51, 52, 124):
                    ex_pos = [mx, my]
                    ex_type = "walkover" if (ln.get("playercross", False) or spec in (52, 124)) else "switch"
                    if ex_type == "switch":
                        switches.append({"kind": "exit_switch", "pos": [mx, my]})
                elif spec == 12 and int(ln.get("arg0", 0) or 0) > 0 and ln.get("playeruse", False):
                    switches.append({"kind": "remote_door", "tag": int(ln.get("arg0", 0)), "pos": [mx, my]})
                elif spec in (13, 14) and int(ln.get("arg3", 0) or 0) > 0:
                    locked_doors.append({"lock": int(ln.get("arg3", 0)), "pos": [mx, my]})
        else:
            _, tagged_sectors = _sector_tags(ml.get("SECTORS", b""))
            th = ml.get("THINGS", b"")
            for i in range(0, len(th) - 9, 10):
                tx, ty, _ang, tid, _fl = struct.unpack_from("<hhHHH", th, i)
                if tid in key_lock_map:
                    keys_placed.append({"lock": key_lock_map[tid], "pos": [float(tx), float(ty)]})
            ld = ml.get("LINEDEFS", b"")
            vtx = ml.get("VERTEXES", b"")
            for i in range(0, len(ld) - 13, 14):
                v1, v2, _fl, spec, tag, _f, _b = struct.unpack_from("<HHHHhHH", ld, i)
                if v1 * 4 + 4 > len(vtx) or v2 * 4 + 4 > len(vtx):
                    continue
                x1, y1 = struct.unpack_from("<hh", vtx, v1 * 4)
                x2, y2 = struct.unpack_from("<hh", vtx, v2 * 4)
                mx, my = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
                if spec in (11, 51, 52, 124) and ex_pos is None:
                    ex_pos = [mx, my]
                    ex_type = "walkover" if spec in (52, 124) else "switch"
                    if ex_type == "switch":
                        switches.append({"kind": "exit_switch", "pos": [mx, my]})
                elif spec in (26, 27, 28, 32, 33, 34):
                    lk = {26: 130, 32: 130, 27: 131, 34: 131, 28: 129, 33: 129}[spec]
                    locked_doors.append({"lock": lk, "pos": [mx, my]})
                elif spec in (29, 61, 63, 103, 111, 112, 114, 115) and tag > 0:
                    switches.append({"kind": "remote_door", "tag": int(tag), "pos": [mx, my]})

        for switch in switches:
            if switch["kind"] == "remote_door":
                switch["door_sectors"] = tagged_sectors.get(switch["tag"], [])
        manifest[m.upper()] = {
            "map_name": m.upper(),
            "keys_placed": keys_placed,
            "locked_doors": locked_doors,
            "switches": switches,
            "exit_pos": ex_pos,
            "exit_type": ex_type,
        }
    return manifest


class FastDoomPackEnv:
    """Fast multi-map training wrapper over a packed WAD shard with Fog-of-War & Geodesic Oracle."""

    def __init__(
        self,
        wad_path: str | Path,
        *,
        episode_ticks: int = 2100,
        render_buffers: bool = True,
        gamma: float = 0.995,
        compute_spatial_targets: bool = True,
        map_subset: list[str] | None = None,
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
        if map_subset is not None:
            sub = {m.upper() for m in map_subset}
            self.map_names = [m for m in self.map_names if m.upper() in sub]

        manifest_path = self.wad_path.with_suffix(".json")
        self.manifest: dict[str, dict] = {}
        if manifest_path.is_file():
            for entry in json.loads(manifest_path.read_text(encoding="utf-8")):
                name = entry["map_name"].upper()
                self.manifest[name] = entry
                # Older sidecars may contain only a remote switch tag.
                unresolved = [sw for sw in entry.get("switches", []) if sw.get("kind") == "remote_door"
                              and sw.get("tag") and not sw.get("door_sectors") and sw.get("door_pos") is None]
                if unresolved and name in self.map_names:
                    ml = map_lumps(lumps, name)
                    if "TEXTMAP" in ml:
                        blocks = parse_udmf(ml["TEXTMAP"].decode("utf-8", "replace"))
                        tags = _udmf_sector_tags(blocks)
                    else:
                        _, tags = _sector_tags(ml.get("SECTORS", b""))
                    for sw in unresolved:
                        sw["door_sectors"] = tags.get(sw["tag"], [])
        else:
            self.manifest = _auto_extract_manifest(lumps, self.map_names)

        valid_maps = []
        self._mapinfos = {}
        max_sec = 260 if "freedoom2" in self.wad_path.name.lower() else 2048
        for m in self.map_names:
            mi = load_mapinfo(self.wad_path, m)
            if mi.num_sectors < max_sec:
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
        self._prev_subgoal: tuple[float, float] | None = None
        self._visited_cells: set[tuple[int, int]] = set()
        self._visited_grid: np.ndarray | None = None
        self._seen_grid: np.ndarray | None = None
        self._opened_doors: set[int] = set()
        self._unlocked_doors: set[int] = set()
        self._rewarded_switch_use: set[tuple[int, int]] = set()
        self._use_down = False
        self._previous_clearance: np.ndarray | None = None
        self._rewarded_dropoff: bool = False
        self._rng = np.random.default_rng(0)

    def _update_fog_of_war(self, obs: dict, priv: dict) -> None:
        """Mark all grid cells traversed by the 64 ego-rays as observed (exact Fog-of-War!)."""
        field = self._fields[self.current_map]
        seen = self._seen_grid
        vis = self._visited_grid
        assert seen is not None and vis is not None

        px, py = priv["x"], priv["y"]
        heading = math.radians(priv["angle"])
        cx0, cy0 = field.world_to_cell(px, py)
        vis[cx0, cy0] = 1.0
        seen[cx0, cy0] = 1.0

        # Vectorized ray march along all 64 rays up to their hit distances
        dists = obs["rays"]["distance"] * S.RAY_RANGE  # (64,)
        ray_ang = heading + np.arange(S.NUM_RAYS, dtype=np.float32) * (2.0 * math.pi / S.NUM_RAYS)
        cos_a = np.cos(ray_ang)
        sin_a = np.sin(ray_ang)

        # Sample every 48 units along each ray up to hit distance (max 16 samples = 768u)
        steps = np.arange(1, 17, dtype=np.float32) * 48.0  # (16,)
        r_mat = np.minimum(steps[None, :], dists[:, None])  # (64, 16)
        wx = px + cos_a[:, None] * r_mat
        wy = py + sin_a[:, None] * r_mat
        gx = np.floor((wx - field.ox) / GRID_CELL).astype(np.int32)
        gy = np.floor((wy - field.oy) / GRID_CELL).astype(np.int32)
        ok = (gx >= 0) & (gx < field.w) & (gy >= 0) & (gy < field.h)
        seen[gx[ok], gy[ok]] = 1.0

        # Also mark the hit endpoint cell itself (+16u into door/wall boundary)
        r_end = np.minimum(dists + 16.0, S.RAY_RANGE)
        ex = np.floor((px + cos_a * r_end - field.ox) / GRID_CELL).astype(np.int32)
        ey = np.floor((py + sin_a * r_end - field.oy) / GRID_CELL).astype(np.int32)
        ok_e = (ex >= 0) & (ex < field.w) & (ey >= 0) & (ey < field.h)
        seen[ex[ok_e], ey[ok_e]] = 1.0

    def reset(self, *, map_name: str | None = None, seed: int | None = None) -> tuple[dict, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        target_map = (map_name or str(self._rng.choice(self.map_names))).upper()
        if target_map != self.current_map:
            self.current_map = target_map
            self.env.map_name = target_map
            self.env.mapinfo = self._mapinfos[target_map]
            self.env._game.set_doom_map(target_map)
            self.env._geometry = None
        obs, info = self.env.reset(seed=int(self._rng.integers(0, 2**31 - 1)))
        priv = self.env.privileged_state()
        field = self._fields[self.current_map]
        self._prev_priv = priv
        self._visited_cells = {(int(priv["x"] // 128), int(priv["y"] // 128))}
        self._visited_grid = np.zeros((field.w, field.h), dtype=np.float32)
        self._seen_grid = np.zeros((field.w, field.h), dtype=np.float32)
        self._opened_doors = set()
        self._unlocked_doors = set()
        self._rewarded_switch_use = set()
        self._use_down = False
        self._previous_clearance = None
        self._rewarded_dropoff = False
        self._update_fog_of_war(obs, priv)
        d_geo, sin_g, cos_g, subgoal, _ = self._query_subgoal(priv)
        self._prev_subgoal = subgoal
        self._prev_phi = self._phi_from_dist(d_geo)
        return obs, self._build_oracle_info(priv, info, odom_delta=(0.0, 0.0, 0.0, 1.0), geo_res=(d_geo, sin_g, cos_g))

    def _active_subgoal_with_kind(self, priv: dict) -> tuple[tuple[float, float], str]:
        meta = self.manifest.get(self.current_map, {})
        keys_held = priv["keys"]
        lock_held = {
            130: bool(keys_held[0] > 0.5 or keys_held[3] > 0.5),
            2: bool(keys_held[0] > 0.5 or keys_held[3] > 0.5),
            131: bool(keys_held[1] > 0.5 or keys_held[4] > 0.5),
            3: bool(keys_held[1] > 0.5 or keys_held[4] > 0.5),
            129: bool(keys_held[2] > 0.5 or keys_held[5] > 0.5),
            1: bool(keys_held[2] > 0.5 or keys_held[5] > 0.5),
        }
        # Remote door switches if any remote door hasn't been opened yet
        for sw in meta.get("switches", []):
            if sw.get("kind") == "remote_door" and not self._remote_door_is_open(priv, sw):
                return (float(sw["pos"][0]), float(sw["pos"][1])), "remote_switch"
        for k in meta.get("keys_placed", []):
            if not lock_held.get(k["lock"], False):
                return (float(k["pos"][0]), float(k["pos"][1])), "key"
        ex = meta.get("exit_pos")
        if ex is not None:
            kind = "exit_switch" if meta.get("exit_type") == "switch" else "exit_walk"
            return (float(ex[0]), float(ex[1])), kind
        return (priv["x"], priv["y"]), "none"

    def _active_subgoal(self, priv: dict) -> tuple[float, float]:
        return self._active_subgoal_with_kind(priv)[0]

    def _locked_blocked_grid(self, priv: dict) -> np.ndarray | None:
        field = self._fields[self.current_map]
        if not field.door_sectors:
            return None
        keys_held = priv["keys"]
        blue = bool(keys_held[0] > 0.5 or keys_held[3] > 0.5)
        yellow = bool(keys_held[1] > 0.5 or keys_held[4] > 0.5)
        red = bool(keys_held[2] > 0.5 or keys_held[5] > 0.5)
        floor, ceil = priv.get("sector_floor"), priv.get("sector_ceiling")
        blocked = None
        for sid, lock in field.door_sectors.items():
            if lock <= 0:
                continue
            if lock in (129, 1):
                has_k = red
            elif lock in (130, 2):
                has_k = blue
            elif lock in (131, 3):
                has_k = yellow
            elif lock in (101, 229):
                has_k = red and blue and yellow
            else:
                has_k = red or blue or yellow
            if has_k:
                continue
            if floor is not None and ceil is not None and 0 <= sid < len(floor) and (ceil[sid] - floor[sid]) >= 48.0:
                continue
            if blocked is None:
                blocked = np.zeros((field.w, field.h), dtype=bool)
            for cx, cy in field.door_cells.get(sid, []):
                blocked[cx, cy] = True
        return blocked

    def _remote_sectors(self, sw: dict) -> list[int]:
        if sw.get("door_sectors"):
            return sw["door_sectors"]
        # Generated and older manifests store the door position.
        if sw.get("door_pos") is not None:
            field = self._fields[self.current_map]
            sid = int(field.cell_sector[field.world_to_cell(*sw["door_pos"])])
            if sid >= 0:
                return [sid]
        return []

    def _remote_door_is_open(self, priv: dict, sw: dict) -> bool:
        """Whether a linked opening is traversable or has already been credited."""
        floor, ceil = priv.get("sector_floor"), priv.get("sector_ceiling")
        targets = self._remote_sectors(sw)
        if floor is None or ceil is None or not targets:
            return True
        return any(0 <= sid < len(floor) and (
            ceil[sid] - floor[sid] >= PLAYER_HEIGHT or sid in self._opened_doors
        ) for sid in targets)

    def _switch_activated(self, before: dict, after: dict) -> bool:
        """Credit a USE edge only when its switch produces an engine event."""
        meta = self.manifest.get(self.current_map, {})
        if after["exit_reached"] and meta.get("exit_type") == "switch":
            return True
        if any(state.get(key) is None for state in (before, after) for key in ("sector_floor", "sector_ceiling")):
            return False
        for switch in meta.get("switches", []):
            if switch["kind"] != "remote_door":
                continue
            x, y = switch["pos"]
            key = (int(round(x)), int(round(y)))
            if key in self._rewarded_switch_use or math.hypot(x - before["x"], y - before["y"]) >= 72.0:
                continue
            for sid in self._remote_sectors(switch):
                if 0 <= sid < len(before["sector_floor"]):
                    old = before["sector_ceiling"][sid] - before["sector_floor"][sid]
                    new = after["sector_ceiling"][sid] - after["sector_floor"][sid]
                    was_opening = self._previous_clearance is not None and old > self._previous_clearance[sid]
                    if old < PLAYER_HEIGHT and new > old and not was_opening:
                        self._rewarded_switch_use.add(key)
                        return True
        return False

    @staticmethod
    def _phi_from_dist(d_geo: float) -> float:
        return max(0.0, (4096.0 - d_geo) / 256.0) + 3.0 * math.exp(-d_geo / 180.0)

    def _query_subgoal(self, priv: dict) -> tuple[float, float, float, tuple[float, float], str]:
        field = self._fields[self.current_map]
        heading = math.radians(priv["angle"])
        subgoal, kind = self._active_subgoal_with_kind(priv)
        blk = self._locked_blocked_grid(priv)
        d_geo, sin_g, cos_g = field.geodesic_query(priv["x"], priv["y"], heading, subgoal, blocked_grid=blk)
        return d_geo, sin_g, cos_g, subgoal, kind

    def _compute_phi(self, priv: dict) -> float:
        d_geo, _, _, _, _ = self._query_subgoal(priv)
        return self._phi_from_dist(d_geo)

    def _build_oracle_info(
        self,
        priv: dict,
        base_info: dict,
        odom_delta: tuple[float, float, float, float],
        geo_res: tuple[float, float, float] | None = None,
    ) -> dict:
        field = self._fields[self.current_map]
        heading = math.radians(priv["angle"])
        if geo_res is not None:
            d_geo, sin_g, cos_g = geo_res
        else:
            d_geo, sin_g, cos_g, _, _ = self._query_subgoal(priv)
        meta = self.manifest.get(self.current_map, {})

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
        priv_vec[20] = float(self._seen_grid.sum() / max(1.0, float(field.walkable.sum()))) if self._seen_grid is not None else 0.0

        out = {
            **base_info,
            "map_name": self.current_map,
            "priv_vec": priv_vec,
            "odom_target": np.asarray(odom_delta, dtype=np.float32),
            "geodesic_dist": d_geo,
        }
        if self.compute_spatial_targets and self._seen_grid is not None and self._visited_grid is not None:
            out["afford_target"] = _frustum_affordance(
                priv["x"], priv["y"], heading, meta, n_cols=AFFORD_COLS, geometry=self.env._geometry, field=field
            )
            out["bev_target"] = field.ego_bev_ground_truth(
                priv["x"], priv["y"], heading, self._seen_grid, self._visited_grid
            )
        return out

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        obs, _raw_r, term, trunc, info = self.env.step(action)
        priv = self.env.privileged_state()
        prev = self._prev_priv
        assert prev is not None

        prev_h = math.radians(prev["angle"])
        dx_w = priv["x"] - prev["x"]
        dy_w = priv["y"] - prev["y"]
        dx_ego, dy_ego = ego_rotate(dx_w, dy_w, prev_h)
        dtheta = math.radians((priv["angle"] - prev["angle"] + 180.0) % 360.0 - 180.0)
        odom_delta = (float(dx_ego), float(dy_ego), math.sin(dtheta), math.cos(dtheta))

        if not (term or trunc):
            self._update_fog_of_war(obs, priv)

        # Scorer-aligned reward shaping (strong exit, progression, dropoff, switch & survival!)
        eff_tic = int(priv["tic"])
        if eff_tic <= 0 and prev["tic"] > 0:
            eff_tic = min(self.episode_ticks, int(prev["tic"]) + 4)
            priv["tic"] = eff_tic
        reward = -0.004
        if priv["exit_reached"]:
            hp_bonus = 6.0 * min(1.0, max(0.0, priv["health"]) / 100.0)
            time_bonus = 8.0 * max(0.0, 1.0 - eff_tic / max(1, self.episode_ticks))
            reward += 35.0 + hp_bonus + time_bonus
        if priv["dead"]:
            early_frac = max(0.0, 1.0 - eff_tic / max(1, self.episode_ticks))
            reward -= 10.0 + 5.0 * early_frac

        new_keys = int(np.sum(priv["keys"] > 0.5) - np.sum(prev["keys"] > 0.5))
        if new_keys > 0:
            reward += 7.0 * new_keys

        # One-shot drop-off traversal bonus (teaches RGB-D 3D ledge drops despite fake RAY_WALL!)
        if not self._rewarded_dropoff and (priv["z"] < prev["z"] - 24.0):
            self._rewarded_dropoff = True
            reward += 4.0

        floor, ceil, locks = priv["sector_floor"], priv["sector_ceiling"], priv["door_lock"]
        if floor is not None and ceil is not None and locks is not None:
            open_mask = (ceil - floor) >= 56.0
            for sid in np.flatnonzero(open_mask & (locks == 0)):
                if int(sid) not in self._opened_doors:
                    self._opened_doors.add(int(sid))
                    reward += 1.5
            for sid in np.flatnonzero(open_mask & (locks > 0)):
                if int(sid) not in self._unlocked_doors:
                    self._unlocked_doors.add(int(sid))
                    reward += 12.0

        cell = (int(priv["x"] // 128), int(priv["y"] // 128))
        if cell not in self._visited_cells:
            self._visited_cells.add(cell)
            reward += 0.15

        dkills = priv["kills"] - prev["kills"]
        if dkills > 0:
            reward += 2.5 * dkills
        ddmg = max(0.0, priv["damage_taken"] - prev["damage_taken"])
        if ddmg > 0:
            reward -= 0.055 * ddmg

        use_down = bool(action[7])
        if use_down and not self._use_down and self._switch_activated(prev, priv):
            reward += 1.5
        self._use_down = use_down
        d_geo, sin_g, cos_g, subgoal, _ = self._query_subgoal(priv)

        phi = self._phi_from_dist(d_geo)
        # Avoid negative/positive jump when active subgoal transitions (e.g. key collected -> exit)
        if subgoal == self._prev_subgoal and new_keys == 0 and not term:
            reward += 0.90 * (phi - self._prev_phi)
        self._prev_subgoal = subgoal
        self._prev_phi = phi
        if prev["sector_floor"] is not None and prev["sector_ceiling"] is not None:
            self._previous_clearance = prev["sector_ceiling"] - prev["sector_floor"]
        self._prev_priv = priv

        return obs, float(reward), term, trunc, self._build_oracle_info(priv, info, odom_delta, geo_res=(d_geo, sin_g, cos_g))

    def close(self) -> None:
        self.env.close()
