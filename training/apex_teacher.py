"""Apex privileged DAgger teacher combining reachability planning, LOS corner-cutting,
and Tier-4 survival combat (training-only; never imported by submission/).

Combines:
- Live reachability routing (reachable exit > nearest reachable key / remote switch > frontier)
- Live remote-door sector clearance tracking (works on generated WADs and binary WADs like Freedoom2)
- Body-radius (+-14u) line-of-sight corner-cutting along the BFS shortest path
- Hitscanner-priority combat, exact closest-approach projectile dodging while firing,
  explosive-barrel safety, 360-deg enemy-ray awareness, and near-exit rush
- Executed-action USE tracking for mixed DAgger rollouts + auxiliary waypoint/goal labels
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from doom_env.sensors import PLAYER_HEIGHT, RAY_DOOR, RAY_ENEMY, RAY_LOCKED
from training.frontier_oracle import FrontierOracle, factored_np_to_multibinary
from training.wad_oracle import FastDoomPackEnv, GRID_CELL

ENEMY_TYPES = (1, 2, 3)  # melee, hitscan, projectile
PROJECTILE_TYPE = 4
HEALTH_TYPE = 5
ARMOR_TYPE = 6
AMMO_TYPE = 7
WEAPON_TYPE = 8
BARREL_TYPE = 10

USE_COOLDOWN_DOOR = 18  # ~2s at 8.75 Hz so USE never reverses a rising door
USE_COOLDOWN_SWITCH = 4
AIM_DEADBAND_DEG = 6.0
TURN_START_DEG = 9.0
ROTATE_IN_PLACE_DEG = 45.0

GOAL_KIND_TO_ID = {
    "none": 0,
    "exit_walk": 1,
    "exit_switch": 1,
    "key": 2,
    "remote_switch": 3,
    "frontier": 4,
}


def _wrap_deg(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def dir_rel_to_bin(dir_rel_deg: float | None, num_bins: int = 16) -> int:
    """Quantize ego-centric waypoint bearing (-180..180 deg, + = left) into [0..num_bins-1], or num_bins if none."""
    if dir_rel_deg is None or not math.isfinite(dir_rel_deg):
        return num_bins
    step = 360.0 / num_bins
    return int(round((dir_rel_deg % 360.0) / step)) % num_bins


class ApexTeacher(FrontierOracle):
    """High-survival privileged teacher for DAgger + Kickstart PPO."""

    def reset(self) -> None:
        super().reset()
        self.fwd_hist: deque[bool] = deque(maxlen=12)
        self.last_goal_kind: str = "none"
        self.last_dir_rel_deg: float | None = None
        self.last_goal_dist: float = 0.0
        self.prev_wpn_press: bool = False
        self.prev_hp: float = 100.0
        self.turn_dir: int = 0  # 0=none, 1=left, 2=right

    def record_action(self, action: np.ndarray) -> None:
        """Update USE and weapon edge state from the buttons actually sent to ViZDoom."""
        use_pressed = bool(action[7])
        if use_pressed and not self.prev_use:
            if self._pending_door is not None:
                self.door_wait_timer = 8
                self.use_cooldown = USE_COOLDOWN_DOOR
                self.last_door_cell_used = self._pending_door
            elif self._pending_switch is not None:
                target_cell, sweep_length = self._pending_switch
                self.use_cooldown = USE_COOLDOWN_SWITCH
                self.switch_sweep_idx += 1
                if sweep_length is not None and self.switch_sweep_idx >= sweep_length:
                    self.triggered_switches.add(target_cell)
                    self.remote_blocked_cells.clear()
                    self.switch_sweep_idx = 0
        self.prev_use = use_pressed
        self.prev_wpn_press = bool(action[8])
        self._pending_door = None
        self._pending_switch = None

    def _build_blocked_grid(self, env: FastDoomPackEnv, priv: dict) -> np.ndarray:
        blocked = super()._build_blocked_grid(env, priv)
        floor, ceiling = priv.get("sector_floor"), priv.get("sector_ceiling")
        if floor is None or ceiling is None:
            return blocked
        field = env._fields[env.current_map]
        for switch in env.manifest.get(env.current_map, {}).get("switches", []):
            if switch.get("kind") != "remote_door":
                continue
            sectors = env._remote_sectors(switch)
            if sectors:
                for sid in sectors:
                    if 0 <= sid < len(floor):
                        closed = (ceiling[sid] - floor[sid]) < PLAYER_HEIGHT
                        cells = field.door_cells.get(sid)
                        if cells:
                            for dc in cells:
                                blocked[dc] = closed
                        elif switch.get("door_pos") is not None:
                            blocked[field.world_to_cell(*switch["door_pos"])] = closed
        return blocked

    def _select_nav_target(
        self, env: FastDoomPackEnv, priv: dict, blocked: np.ndarray,
    ) -> tuple[str, tuple[float, float] | None, tuple[int, int] | None]:
        meta = env.manifest.get(env.current_map, {})
        field = env._fields[env.current_map]
        player = field.world_to_cell(priv["x"], priv["y"])

        def distance_to(cell: tuple[int, int]) -> float:
            if not (0 <= cell[0] < field.w and 0 <= cell[1] < field.h):
                return math.inf
            if not field.walkable[cell] or blocked[cell]:
                return math.inf
            d = float(field._bfs_from(cell[0], cell[1], blocked_grid=blocked)[player])
            if math.isfinite(d):
                return d
            # Check immediate walkable neighbor if player is slightly clipped on a wall edge
            best = math.inf
            for nx, ny in field.neighbors(player[0], player[1]):
                if not blocked[nx, ny]:
                    cand = float(field._bfs_from(cell[0], cell[1], blocked_grid=blocked)[nx, ny])
                    if cand < best:
                        best = cand
            return best

        # 1. Reachable exit takes absolute priority
        exit_pos = meta.get("exit_pos")
        if exit_pos is not None:
            cell = field.world_to_cell(float(exit_pos[0]), float(exit_pos[1]))
            if math.isfinite(distance_to(cell)):
                kind = "exit_switch" if meta.get("exit_type") == "switch" else "exit_walk"
                return kind, (float(exit_pos[0]), float(exit_pos[1])), cell

        # 2. Nearest reachable uncollected key or unopened remote-door switch
        goals: list[tuple[str, tuple[float, float]]] = []
        for switch in meta.get("switches", []):
            if switch.get("kind") != "remote_door":
                continue
            if env._remote_door_is_open(priv, switch):
                continue
            sw_cell = field.world_to_cell(float(switch["pos"][0]), float(switch["pos"][1]))
            if sw_cell in self.triggered_switches and not env._remote_sectors(switch):
                continue
            goals.append(("remote_switch", (float(switch["pos"][0]), float(switch["pos"][1]))))

        for key in meta.get("keys_placed", []):
            if not self._has_key_for_lock(priv["keys"], int(key["lock"])):
                goals.append(("key", (float(key["pos"][0]), float(key["pos"][1]))))

        reachable: list[tuple[float, str, tuple[float, float], tuple[int, int]]] = []
        for kind, xy in goals:
            cell = field.world_to_cell(xy[0], xy[1])
            d = distance_to(cell)
            if math.isfinite(d):
                reachable.append((d, kind, xy, cell))
        if reachable:
            _, kind, xy, cell = min(reachable)
            return kind, xy, cell

        # 3. Fallback to FrontierOracle exploration
        return super()._select_nav_target(env, priv, blocked)

    def _trace_path(
        self, field, dist_field: np.ndarray, start: tuple[int, int], blocked: np.ndarray, max_steps: int = 7
    ) -> list[tuple[int, int]]:
        path: list[tuple[int, int]] = []
        cx, cy = start
        cur_d = float(dist_field[cx, cy])
        if not math.isfinite(cur_d):
            return path
        for _ in range(max_steps):
            if cur_d <= 0.0:
                break
            best_cell = None
            best_d = cur_d
            for nx, ny in field.neighbors(cx, cy):
                if blocked[nx, ny]:
                    continue
                nd = float(dist_field[nx, ny])
                if nd < best_d:
                    best_d = nd
                    best_cell = (nx, ny)
            if best_cell is None:
                break
            path.append(best_cell)
            cx, cy = best_cell
            cur_d = best_d
        return path

    def _clear_corridor_los(
        self,
        field,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        end_cell: tuple[int, int],
        blocked: np.ndarray,
        priv: dict,
    ) -> bool:
        """Body-radius (+-14u) straight-line check across walkable, non-blocked, non-closed-door cells."""
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length < 1e-3:
            return True
        n = max(2, int(length / 14.0))
        ux, uy = dx / length, dy / length
        px_n, py_n = -uy, ux
        floor, ceil = priv.get("sector_floor"), priv.get("sector_ceiling")

        for i in range(1, n):
            t = i / n
            bx, by = x0 + dx * t, y0 + dy * t
            for off in (-14.0, 0.0, 14.0):
                sx, sy = bx + px_n * off, by + py_n * off
                cx, cy = field.world_to_cell(sx, sy)
                if (cx, cy) == end_cell:
                    continue
                if not (0 <= cx < field.w and 0 <= cy < field.h):
                    return False
                if not field.walkable[cx, cy] or blocked[cx, cy]:
                    return False
                # Don't cut through a closed door cell (approach it head-on so USE fires)
                sid = int(field.cell_sector[cx, cy])
                if sid in field.door_sectors:
                    if floor is not None and ceil is not None and 0 <= sid < len(floor):
                        if (ceil[sid] - floor[sid]) < PLAYER_HEIGHT:
                            return False
                    else:
                        return False
        return True

    def _dodge_strafe(self, obs: dict) -> int:
        """Return strafe action (1=left, 2=right, 0=none) to sidestep an incoming projectile."""
        ents = obs["visible_entities"]
        mask = np.asarray(ents["mask"]) > 0.5
        etypes = np.asarray(ents["entity_type"])
        feats = np.asarray(ents["features"])
        best_t = math.inf
        best_strafe = 0
        for k in np.flatnonzero(mask & (etypes == PROJECTILE_TYPE)):
            d_norm, sin_a, cos_a, vf, vr = (float(v) for v in feats[k])
            dist = d_norm * 1024.0
            if dist > 560.0 or dist < 1e-3:
                continue
            # Position in ego frame: px = forward, py = left
            px, py = cos_a * dist, sin_a * dist
            # Relative velocity in ego frame: vx = forward, vy = left (-vel_right)
            vx, vy = vf, -vr
            speed = math.hypot(vx, vy)
            if speed < 1.5:
                continue
            closing = -(px * vx + py * vy) / dist
            if closing < 1.5:
                continue
            cross = px * vy - py * vx
            miss = abs(cross) / speed
            if miss > 48.0:
                continue
            t_hit = dist / max(closing, 1.0)
            if t_hit < best_t:
                best_t = t_hit
                # Step perpendicular away from the projectile's line of travel
                best_strafe = 1 if cross < 0.0 else 2
        return best_strafe

    def _propose_action(self, obs: dict, env: FastDoomPackEnv) -> np.ndarray:
        self.step_count += 1
        priv = env.env.privileged_state()
        field = env._fields[env.current_map]
        px, py = float(priv["x"]), float(priv["y"])
        ang_deg = float(priv["angle"])
        heading = math.radians(ang_deg)

        if self.use_cooldown > 0:
            self.use_cooldown -= 1
        if self.wpn_cooldown > 0:
            self.wpn_cooldown -= 1

        self.pos_hist.append((px, py))

        vit = obs["vitals"]
        hp = float(vit[0])
        cur_ammo = float(vit[3])
        other_ammo = float(vit[4] + vit[5] + vit[6] + vit[7])
        took_damage = (self.prev_hp - hp) > 0.5
        self.prev_hp = hp

        # Weapon discipline: ONLY switch when current weapon is empty and other ammo exists
        wpn = 0
        if cur_ammo <= 0.0 and other_ammo > 0.0 and not self.prev_wpn_press and self.wpn_cooldown == 0:
            wpn = 1
            self.wpn_cooldown = 4

        # Plan navigation target first so we know distance to exit (for near-exit rush & aux labels)
        blocked = self._build_blocked_grid(env, priv)
        nav_kind, target_xy, target_cell = self._select_nav_target(env, priv, blocked)
        self.last_goal_kind = nav_kind

        d_geo = math.inf
        nav_err_deg: float | None = None
        next_path_cell: tuple[int, int] | None = None
        if target_xy is not None and target_cell is not None:
            dist_field = field._bfs_from(target_cell[0], target_cell[1], blocked_grid=blocked)
            pcx, pcy = field.world_to_cell(px, py)
            d_geo = float(dist_field[pcx, pcy])
            if not math.isfinite(d_geo):
                d_geo, sin_g, cos_g = field.geodesic_query(px, py, heading, target_xy, blocked_grid=blocked)
                nav_err_deg = math.degrees(math.atan2(sin_g, cos_g))
            else:
                path = self._trace_path(field, dist_field, (pcx, pcy), blocked, max_steps=7)
                if path:
                    next_path_cell = path[0]
                    tx, ty = field.cell_center(*path[0])
                    for cell in path[1:]:
                        cx_w, cy_w = field.cell_center(*cell)
                        if self._clear_corridor_los(field, px, py, cx_w, cy_w, cell, blocked, priv):
                            tx, ty = cx_w, cy_w
                        else:
                            break
                    nav_err_deg = _wrap_deg(math.degrees(math.atan2(ty - py, tx - px)) - ang_deg)
                else:
                    nav_err_deg = _wrap_deg(math.degrees(math.atan2(target_xy[1] - py, target_xy[0] - px)) - ang_deg)

        self.last_dir_rel_deg = nav_err_deg
        self.last_goal_dist = float(d_geo) if math.isfinite(d_geo) else 0.0

        near_exit = nav_kind in ("exit_walk", "exit_switch") and math.isfinite(d_geo) and d_geo <= 192.0

        # Parse visible entities (only what student sees!)
        ve = obs["visible_entities"]
        mask = np.asarray(ve["mask"]) > 0.5
        etypes = np.asarray(ve["entity_type"])
        feats = np.asarray(ve["features"])

        enemies: list[tuple[float, float, int]] = []  # (dist, rel_deg, etype)
        barrels: list[tuple[float, float]] = []       # (dist, rel_deg)
        pickups: list[tuple[float, float, int]] = []  # (dist, rel_deg, etype)
        wpn_id = int(round(float(vit[2])))
        armor = float(vit[1])
        for k in np.flatnonzero(mask):
            et = int(etypes[k])
            d = float(feats[k, 0]) * 1024.0
            rel_deg = math.degrees(math.atan2(float(feats[k, 1]), float(feats[k, 2])))
            if et in ENEMY_TYPES:
                enemies.append((d, rel_deg, et))
            elif et == BARREL_TYPE:
                barrels.append((d, rel_deg))
            elif (
                (et == WEAPON_TYPE and wpn_id <= 2 and d < 320.0)
                or (et == HEALTH_TYPE and hp < 75.0 and d < 300.0)
                or (et == ARMOR_TYPE and armor < 50.0 and d < 260.0)
                or (et == AMMO_TYPE and cur_ammo < 25.0 and d < 260.0)
            ):
                pickups.append((d, rel_deg, et))
        enemies.sort(key=lambda e: e[0])
        pickups.sort(key=lambda p: p[0])

        # Ray clearances (ray 0=front, 16=left, 32=back, 48=right)
        rd = np.asarray(obs["rays"]["distance"], dtype=np.float32) * 1024.0
        rht = np.asarray(obs["rays"]["hit_type"], dtype=np.int64)
        front_clear = float(min(rd[0], rd[1], rd[63]))
        left_clear = float(min(rd[14], rd[16], rd[18]))
        right_clear = float(min(rd[46], rd[48], rd[50]))
        back_clear = float(min(rd[30], rd[32], rd[34]))

        dodge_lr = self._dodge_strafe(obs)

        # ------------------------------------------------------------------
        # 1. Survival & Combat (avoid suicidal pistol duels; run-and-gun past tanks!)
        # ------------------------------------------------------------------
        # If we have a stronger weapon (>= shotgun), fight more aggressively; with pistol, only
        # stop to fight close/blocking threats and otherwise run-and-gun toward the objective.
        has_power_wpn = wpn_id >= 3
        max_stop_range = 420.0 if has_power_wpn else 220.0
        if hp < 40.0:
            max_stop_range = 170.0  # low HP: prioritize escaping / grabbing health over dueling

        if enemies and not near_exit and cur_ammo > 0.0:
            pb = [e for e in enemies if e[0] < 150.0]
            close_hitscan = [e for e in enemies if e[2] == 2 and e[0] < max_stop_range]
            close_any = [e for e in enemies if e[0] < max_stop_range and abs(e[1]) < 45.0]
            target = (pb or close_hitscan or close_any)[0] if (pb or close_hitscan or close_any) else None
            if target is not None:
                edist, erel, etype = target
                tol = max(AIM_DEADBAND_DEG, math.degrees(math.atan2(18.0, max(edist, 1.0))))
                barrel_blocking = any(
                    bd < edist and bd < 220.0 and abs(brel - erel) < 8.0 for bd, brel in barrels
                )
                tn = 1 if erel > AIM_DEADBAND_DEG else (2 if erel < -AIM_DEADBAND_DEG else 0)
                atk = 1 if (abs(erel) < tol and not barrel_blocking) else 0

                # Back off from point-blank melee/close enemies; otherwise keep strafing
                fb = 2 if (edist < 150.0 and back_clear > 32.0) else 0
                lr = dodge_lr
                if lr == 0:
                    self.strafe_timer -= 1
                    if self.strafe_timer <= 0:
                        self.strafe_dir = 2 if self.strafe_dir == 1 else 1
                        self.strafe_timer = 8
                    if self.strafe_dir == 1 and left_clear < 44.0:
                        self.strafe_dir = 2
                    elif self.strafe_dir == 2 and right_clear < 44.0:
                        self.strafe_dir = 1
                    lr = self.strafe_dir

                self.turn_dir = 0
                self.fwd_hist.append(fb == 1)
                return np.array([fb, lr, tn, atk, 0, wpn], dtype=np.int64)

        # Grab nearby critical pickup (weapon / health / armor / ammo) when safe
        if pickups and not near_exit and (not enemies or enemies[0][0] > 170.0):
            pdist, prel, _ = pickups[0]
            tn = 1 if prel > 10.0 else (2 if prel < -10.0 else 0)
            fb = 1 if abs(prel) < 40.0 else 0
            self.fwd_hist.append(fb == 1)
            return np.array([fb, dodge_lr, tn, 0, 0, wpn], dtype=np.int64)

        # Opportunistic hip-fire while navigating if an enemy is right in front of our crosshair
        hip_atk = 0
        if enemies and cur_ammo > 0.0:
            for edist, erel, _ in enemies:
                if edist < 600.0 and abs(erel) < max(7.0, math.degrees(math.atan2(18.0, max(edist, 1.0)))):
                    barrel_blocking = any(
                        bd < edist and bd < 220.0 and abs(brel - erel) < 8.0 for bd, brel in barrels
                    )
                    if not barrel_blocking:
                        hip_atk = 1
                    break

        # ------------------------------------------------------------------
        # 2. Door wait / switch interaction / navigation
        # ------------------------------------------------------------------
        ahead_x = px + math.cos(heading) * 52.0
        ahead_y = py + math.sin(heading) * 52.0
        ahead_cell = field.world_to_cell(ahead_x, ahead_y)
        ahead_sid = int(field.cell_sector[ahead_cell])
        floor_arr, ceil_arr = priv.get("sector_floor"), priv.get("sector_ceiling")
        ahead_is_door_sector = ahead_sid in field.door_sectors
        ahead_is_opening = False
        if ahead_is_door_sector and floor_arr is not None and ceil_arr is not None and 0 <= ahead_sid < len(floor_arr):
            ahead_is_opening = (ceil_arr[ahead_sid] - floor_arr[ahead_sid]) > 12.0

        center_hits_door = int(rht[0]) in (RAY_DOOR, RAY_LOCKED)
        center_door_d = float(rd[0]) if center_hits_door else 9999.0
        can_open_front_door = (
            center_hits_door
            and center_door_d < 92.0
            and (int(rht[0]) == RAY_DOOR or not blocked[ahead_cell])
            and not ahead_is_opening
        )

        # Switch interaction (exit switch or remote door switch)
        if nav_kind in ("exit_switch", "remote_switch") and target_xy is not None and target_cell is not None:
            d_eu = math.hypot(target_xy[0] - px, target_xy[1] - py)
            if d_eu < 72.0:
                tx, ty = target_xy
                ang_to_sw = _wrap_deg(math.degrees(math.atan2(ty - py, tx - px)) - ang_deg)
                wall_angles = [0.0, 90.0, 180.0, -90.0]
                if d_eu < 42.0:
                    desired_deg = wall_angles[self.switch_sweep_idx % len(wall_angles)]
                    err_sw = _wrap_deg(desired_deg - ang_deg)
                else:
                    err_sw = ang_to_sw
                tn = 1 if err_sw > 12.0 else (2 if err_sw < -12.0 else 0)
                fb = 1 if (abs(err_sw) < 30.0 and front_clear > 26.0) else 0
                use = 0
                if abs(err_sw) < 28.0 and not self.prev_use and self.use_cooldown == 0:
                    use = 1
                    sweep_len = len(wall_angles) if nav_kind == "remote_switch" else None
                    self._pending_switch = (target_cell, sweep_len)
                self.fwd_hist.append(fb == 1)
                return np.array([fb, dodge_lr, tn, 0, use, wpn], dtype=np.int64)

        # Open closed door directly in front of us
        if can_open_front_door and not self.prev_use and self.use_cooldown == 0:
            self._pending_door = ahead_cell
            self.fwd_hist.append(False)
            return np.array([0, 0, 0, 0, 1, wpn], dtype=np.int64)

        # Also open door if next BFS cell is a closed door within 110u and we're facing it
        if next_path_cell is not None:
            nsid = int(field.cell_sector[next_path_cell])
            if nsid in field.door_sectors and not blocked[next_path_cell]:
                n_open = (
                    floor_arr is not None
                    and ceil_arr is not None
                    and 0 <= nsid < len(floor_arr)
                    and (ceil_arr[nsid] - floor_arr[nsid]) >= PLAYER_HEIGHT
                )
                if not n_open:
                    dcx, dcy = field.cell_center(*next_path_cell)
                    if math.hypot(dcx - px, dcy - py) < 110.0:
                        door_err = _wrap_deg(math.degrees(math.atan2(dcy - py, dcx - px)) - ang_deg)
                        tn = 1 if door_err > TURN_START_DEG else (2 if door_err < -TURN_START_DEG else 0)
                        use = 0
                        if abs(door_err) < 30.0 and not self.prev_use and self.use_cooldown == 0:
                            use = 1
                            self._pending_door = next_path_cell
                        fb = 1 if (abs(door_err) < 30.0 and front_clear > 36.0 and use == 0) else 0
                        self.fwd_hist.append(fb == 1)
                        return np.array([fb, dodge_lr, tn, 0, use, wpn], dtype=np.int64)

        # Steering along path
        err = nav_err_deg if nav_err_deg is not None else 0.0
        if self.turn_dir != 0 and (abs(err) < 1.0 or (self.turn_dir == 1) != (err > 0.0)):
            self.turn_dir = 0
        if self.turn_dir == 0 and abs(err) > TURN_START_DEG:
            self.turn_dir = 1 if err > 0.0 else 2
        tn = self.turn_dir
        fb = 1 if abs(err) < ROTATE_IN_PLACE_DEG else 0
        lr = dodge_lr
        use = 0

        # Stuck recovery (pressing forward for 12 steps without moving > 20u)
        self.fwd_hist.append(fb == 1)
        if self.stuck_timer > 0:
            self.stuck_timer -= 1
            lr = self.stuck_dir
            fb = 1
        elif len(self.pos_hist) == 8 and len(self.fwd_hist) >= 8 and all(list(self.fwd_hist)[-8:]):
            moved = math.hypot(px - self.pos_hist[0][0], py - self.pos_hist[0][1])
            if moved < 18.0:
                self.stuck_count += 1
                self.stuck_dir = 1 if (self.stuck_count % 2 == 1) else 2
                self.stuck_timer = 7
                lr = self.stuck_dir
                if center_hits_door and center_door_d < 100.0 and not self.prev_use and self.use_cooldown == 0:
                    use = 1
                    self._pending_door = ahead_cell

        return np.array([fb, lr, tn, hip_atk, use, wpn], dtype=np.int64)
