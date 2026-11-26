"""Training-Only Fog-of-War Frontier Oracle Expert (NEVER imported by submission/).

100% compliant with docs/RULES.md §1 & §2 (training-time scaffolding only).
Produces near-optimal, 100% observation-history-realizable factored actions `(6,)`
by maintaining strict Fog-of-War consistency:
  1. Spawn Weapon Acquisition & Smart Weapon Cycling (Shotgun/SSG/Chaingun > Pistol > Fist).
  2. Survival & Combat: projectile dodging, damage-turn reflex, threat-prioritized
     aiming & kiting, wall-aware strafing, close-range RL safety.
  3. Opportunistic Pickups: visible keys, weapons, medikits/stimpacks, armor, ammo.
  4. Seen Progression Objectives: seen keys -> seen remote switches -> seen exit.
  5. Consistent Frontier Exploration: wall- & dropoff-aware BFS to nearest unseen
     walkable cell with clean door-open-and-wait timing (zero door bouncing!).
"""

from __future__ import annotations

import math

import numpy as np

from doom_env import spaces as S
from doom_env.sensors import RAY_DOOR, RAY_LOCKED
from training.wad_oracle import FastDoomPackEnv

# Entity type indices from doom_env.spaces.ENTITY_TYPES:
# 0:pad, 1:enemy_melee, 2:enemy_hitscan, 3:enemy_projectile, 4:projectile,
# 5:health, 6:armor, 7:ammo, 8:weapon, 9:key, 10:barrel, 11:other
ENEMY_TYPES = {1, 2, 3}
PICKUP_TYPES = {5, 6, 7, 8, 9}


def factored_np_to_multibinary(f: np.ndarray) -> np.ndarray:
    """Convert shape (6,) factored action -> shape (9,) int8 MultiBinary."""
    a = np.zeros(9, dtype=np.int8)
    if f[0] == 1:
        a[0] = 1
    elif f[0] == 2:
        a[1] = 1
    if f[1] == 1:
        a[2] = 1
    elif f[1] == 2:
        a[3] = 1
    if f[2] == 1:
        a[4] = 1
    elif f[2] == 2:
        a[5] = 1
    a[6] = int(f[3])
    a[7] = int(f[4])
    a[8] = int(f[5])
    return a


class FrontierOracle:
    """Fog-of-War Exploration-Consistent Privileged Expert for DAgger Distillation."""

    def __init__(self, omniscient_fallback_steps: int = 999999) -> None:
        self.omniscient_fallback_steps = omniscient_fallback_steps
        self.reset()

    def reset(self) -> None:
        self.step_count = 0
        self.prev_use = False
        self.prev_health = 100.0
        self.damage_spin_timer = 0
        self.door_wait_timer = 0
        self.use_cooldown = 0
        self.wpn_cooldown = 0
        self.strafe_dir = 1  # 1=left, 2=right
        self.strafe_timer = 0
        self.pos_hist: list[tuple[float, float]] = []
        self.stuck_timer = 0
        self.stuck_dir = 1
        self.triggered_switches: set[tuple[int, int]] = set()
        self.switch_sweep_idx = 0
        self.remote_blocked_cells: set[tuple[int, int]] = set()
        self.last_door_cell_used: tuple[int, int] | None = None
        self.door_use_attempts: dict[tuple[int, int], int] = {}
        self.got_weapon = False
        self.spawn_scan_steps = 0
        self.combat_hold = 0
        self.stuck_count = 0
        self._nav_cache_key: tuple | None = None
        self._nav_cache_val: tuple[str, tuple[float, float] | None, tuple[int, int] | None] = ("none", None, None)
        self._nav_cache_step: int = -999

    @staticmethod
    def _has_key_for_lock(keys_held: np.ndarray, lock: int) -> bool:
        blue = bool(keys_held[0] > 0.5 or keys_held[3] > 0.5)
        yellow = bool(keys_held[1] > 0.5 or keys_held[4] > 0.5)
        red = bool(keys_held[2] > 0.5 or keys_held[5] > 0.5)
        if lock in (0, -1):
            return True
        if lock in (129, 1):
            return red
        if lock in (130, 2):
            return blue
        if lock in (131, 3):
            return yellow
        if lock == 100:
            return red or blue or yellow
        if lock == 101:
            return red and blue and yellow
        return red or blue or yellow

    def _build_blocked_grid(self, env: FastDoomPackEnv, priv: dict) -> np.ndarray:
        field = env._fields[env.current_map]
        blocked = np.zeros((field.w, field.h), dtype=bool)
        floor, ceil = priv.get("sector_floor"), priv.get("sector_ceiling")
        keys_held = priv["keys"]

        for sid, lock in field.door_sectors.items():
            is_open = False
            if floor is not None and ceil is not None and 0 <= sid < len(floor):
                is_open = (ceil[sid] - floor[sid]) >= 48.0
            if not is_open and lock > 0 and not self._has_key_for_lock(keys_held, lock):
                for cx, cy in field.door_cells.get(sid, []):
                    blocked[cx, cy] = True

        for cx, cy in list(self.remote_blocked_cells):
            if 0 <= cx < field.w and 0 <= cy < field.h:
                sid = int(field.cell_sector[cx, cy])
                if floor is not None and ceil is not None and 0 <= sid < len(floor) and (ceil[sid] - floor[sid]) >= 48.0:
                    self.remote_blocked_cells.discard((cx, cy))
                else:
                    blocked[cx, cy] = True
        return blocked

    def _select_nav_target(
        self, env: FastDoomPackEnv, priv: dict, blocked: np.ndarray
    ) -> tuple[str, tuple[float, float] | None, tuple[int, int] | None]:
        field = env._fields[env.current_map]
        seen = env._seen_grid
        assert seen is not None
        meta = env.manifest.get(env.current_map, {})
        px, py = priv["x"], priv["y"]
        pcx, pcy = field.world_to_cell(px, py)
        keys_held = priv["keys"]
        cache_key = (pcx, pcy, int(np.sum(keys_held > 0.5)), len(self.remote_blocked_cells), len(self.triggered_switches))
        if (
            self._nav_cache_key == cache_key
            and (self.step_count - self._nav_cache_step) < 5
            and self._nav_cache_val[0] != "none"
        ):
            return self._nav_cache_val

        dist_from_player = field.bfs_from_sources([(pcx, pcy)], blocked_grid=blocked)

        def _is_seen_near(tx: float, ty: float) -> bool:
            if self.step_count >= self.omniscient_fallback_steps:
                return True
            cx, cy = field.world_to_cell(tx, ty)
            if seen[cx, cy] > 0.5:
                return True
            for nx, ny in field.neighbors(cx, cy):
                if seen[nx, ny] > 0.5:
                    return True
            return False

        def _ret(res):
            self._nav_cache_key = cache_key
            self._nav_cache_val = res
            self._nav_cache_step = self.step_count
            return res

        # 1. Seen uncollected Key (reachable)
        best_key = None
        best_key_d = np.inf
        for k in meta.get("keys_placed", []):
            if self._has_key_for_lock(keys_held, k["lock"]):
                continue
            kx, ky = float(k["pos"][0]), float(k["pos"][1])
            kcx, kcy = field.world_to_cell(kx, ky)
            d = float(dist_from_player[kcx, kcy])
            if math.isfinite(d) and _is_seen_near(kx, ky) and d < best_key_d:
                best_key_d = d
                best_key = (kx, ky, kcx, kcy)
        if best_key is not None:
            return _ret(("key", (best_key[0], best_key[1]), (best_key[2], best_key[3])))

        # 2. Seen untriggered Remote Switch (reachable)
        for sw in meta.get("switches", []):
            if sw.get("kind") != "remote_door":
                continue
            sx, sy = float(sw["pos"][0]), float(sw["pos"][1])
            scx, scy = field.world_to_cell(sx, sy)
            if (scx, scy) in self.triggered_switches:
                continue
            d = float(dist_from_player[scx, scy])
            if math.isfinite(d) and _is_seen_near(sx, sy):
                return _ret(("remote_switch", (sx, sy), (scx, scy)))

        # 3. Seen Exit (reachable!)
        ex = meta.get("exit_pos")
        if ex is not None:
            ex_x, ex_y = float(ex[0]), float(ex[1])
            ecx, ecy = field.world_to_cell(ex_x, ex_y)
            d = float(dist_from_player[ecx, ecy])
            if math.isfinite(d) and _is_seen_near(ex_x, ex_y):
                kind = "exit_switch" if meta.get("exit_type") == "switch" else "exit_walk"
                return _ret((kind, (ex_x, ex_y), (ecx, ecy)))

        # 3.5 Seen Closed Locked Door whose key we NOW hold! (Teaches exact long-horizon backtracking!)
        floor, ceil = priv.get("sector_floor"), priv.get("sector_ceiling")
        best_ld = None
        best_ld_d = np.inf
        for sid, lock in field.door_sectors.items():
            if lock <= 0 or not self._has_key_for_lock(keys_held, lock):
                continue
            is_open = False
            if floor is not None and ceil is not None and 0 <= sid < len(floor):
                is_open = (ceil[sid] - floor[sid]) >= 48.0
            if is_open:
                continue
            for dcx, dcy in field.door_cells.get(sid, []):
                if blocked[dcx, dcy]:
                    continue
                dtx, dty = field.cell_center(dcx, dcy)
                d = float(dist_from_player[dcx, dcy])
                if math.isfinite(d) and _is_seen_near(dtx, dty) and d < best_ld_d:
                    best_ld_d = d
                    best_ld = (dtx, dty, dcx, dcy)
        if best_ld is not None:
            return _ret(("unlock_door", (best_ld[0], best_ld[1]), (best_ld[2], best_ld[3])))

        # 4. Nearest Unseen Frontier Cell (reachable through normal / unlocked doors!)
        unseen_mask = field.walkable & (seen < 0.5) & (~blocked)
        if unseen_mask.any():
            ux, uy = np.nonzero(unseen_mask)
            ds = dist_from_player[ux, uy]
            finite = np.isfinite(ds)
            if finite.any():
                idx = int(np.argmin(np.where(finite, ds, np.inf)))
                fcx, fcy = int(ux[idx]), int(uy[idx])
                return _ret(("frontier", field.cell_center(fcx, fcy), (fcx, fcy)))

        # 5. Fallback: if all reachable cells are seen, re-try exit or untriggered switch
        if ex is not None:
            ex_x, ex_y = float(ex[0]), float(ex[1])
            ecx, ecy = field.world_to_cell(ex_x, ex_y)
            if math.isfinite(float(dist_from_player[ecx, ecy])):
                kind = "exit_switch" if meta.get("exit_type") == "switch" else "exit_walk"
                return _ret((kind, (ex_x, ex_y), (ecx, ecy)))
        for sw in meta.get("switches", []):
            sx, sy = float(sw["pos"][0]), float(sw["pos"][1])
            scx, scy = field.world_to_cell(sx, sy)
            if math.isfinite(float(dist_from_player[scx, scy])):
                return _ret(("remote_switch", (sx, sy), (scx, scy)))
        return _ret(("none", None, None))

    def act(self, obs: dict, env: FastDoomPackEnv) -> tuple[np.ndarray, np.ndarray]:
        self.step_count += 1
        priv = env.env.privileged_state()
        field = env._fields[env.current_map]
        px, py = priv["x"], priv["y"]
        heading = math.radians(priv["angle"])

        vitals = obs["vitals"]
        health = float(vitals[0])
        armor = float(vitals[1])
        sel_wpn = int(round(float(vitals[2])))
        sel_ammo = float(vitals[3])
        ammo_bullets, ammo_shells, ammo_rockets, ammo_cells = (
            float(vitals[4]),
            float(vitals[5]),
            float(vitals[6]),
            float(vitals[7]),
        )
        if sel_wpn >= 3:
            self.got_weapon = True

        took_damage = health < self.prev_health - 0.5
        self.prev_health = health

        ray_d = obs["rays"]["distance"] * S.RAY_RANGE  # (64,)
        ray_t = obs["rays"]["hit_type"]                # (64,)

        ent_f = obs["visible_entities"]["features"]
        ent_t = obs["visible_entities"]["entity_type"]
        ent_m = obs["visible_entities"]["mask"]

        if self.door_wait_timer > 0:
            self.door_wait_timer -= 1
        if self.use_cooldown > 0:
            self.use_cooldown -= 1
        if self.wpn_cooldown > 0:
            self.wpn_cooldown -= 1
        if self.damage_spin_timer > 0:
            self.damage_spin_timer -= 1
        if self.strafe_timer > 0:
            self.strafe_timer -= 1
        else:
            self.strafe_dir = 2 if self.strafe_dir == 1 else 1
            self.strafe_timer = 7

        self.pos_hist.append((px, py))
        if len(self.pos_hist) > 8:
            self.pos_hist.pop(0)

        fb = 0
        lr = 0
        tn = 0
        atk = 0
        use = 0
        wpn = 0

        # ------------------------------------------------------------------
        # 1. Parse visible enemies, projectiles, and pickups
        # ------------------------------------------------------------------
        enemies = []
        projectiles = []
        pickups = []
        barrels = []
        for i in range(S.MAX_ENTITIES):
            if ent_m[i] <= 0:
                continue
            et = int(ent_t[i])
            dist = float(ent_f[i, 0]) * S.RAY_RANGE
            sin_a = float(ent_f[i, 1])
            cos_a = float(ent_f[i, 2])
            ang = math.atan2(sin_a, cos_a)
            vf, vr = float(ent_f[i, 3]), float(ent_f[i, 4])
            if et in ENEMY_TYPES:
                # Hitscanners (ShotgunGuy/ChaingunGuy) and close Melee are top lethal priority!
                type_w = 1.8 if et == 2 else (1.5 if (et == 1 and dist < 240) else 1.0)
                score = type_w / max(dist, 32.0)
                enemies.append((score, dist, ang, et))
            elif et == 4:
                v_close = -(vf * cos_a - vr * sin_a)
                if v_close > 0.8 and dist < 520.0:
                    ttc = dist / max(v_close, 1.0)
                    lat_left = dist * sin_a - vr * ttc
                    if abs(lat_left) < 72.0:
                        projectiles.append((dist, lat_left, ang))
            elif et == 10 and dist < 80.0 and abs(ang) < 0.7:
                barrels.append((dist, ang))
            elif et in PICKUP_TYPES:
                want = False
                if et == 9:  # key
                    want = dist < 480.0
                elif et == 8:  # weapon (HUGE priority if we still only have Pistol!)
                    want = dist < (520.0 if not self.got_weapon else 220.0)
                elif et == 5 and health < 92.0:
                    want = dist < (420.0 if health < 60.0 else 240.0)
                elif et == 6 and armor < 80.0:
                    want = dist < 220.0
                elif et == 7 and (ammo_shells < 20.0 or ammo_bullets < 50.0):
                    want = dist < 200.0
                if want:
                    prio = 0 if et == 9 else (1 if (et == 8 and not self.got_weapon) else (2 if (et == 5 and health < 55) else 3))
                    pickups.append((prio, dist, ang, et))

        enemies.sort(key=lambda x: -x[0])
        projectiles.sort(key=lambda x: x[0])
        pickups.sort(key=lambda x: (x[0], x[1]))

        # Smart weapon switching (only when holding Fist, empty gun, or close-range Rocket Launcher!)
        has_shells = ammo_shells > 0
        has_bullets = ammo_bullets > 0
        has_cells = ammo_cells > 0
        has_good_ammo = has_shells or has_bullets or has_cells
        if self.wpn_cooldown == 0:
            if sel_wpn == 1 and has_good_ammo:
                wpn = 1
                self.wpn_cooldown = 7
            elif sel_wpn >= 2 and sel_ammo <= 0 and has_good_ammo:
                wpn = 1
                self.wpn_cooldown = 7
            elif sel_wpn == 5 and enemies and enemies[0][1] < 210.0 and has_good_ammo:
                wpn = 1
                self.wpn_cooldown = 7

        left_clear = float(min(ray_d[14], ray_d[16], ray_d[18]))
        right_clear = float(min(ray_d[46], ray_d[48], ray_d[50]))
        back_clear = float(min(ray_d[30], ray_d[32], ray_d[34]))
        front_clear = float(min(ray_d[0], ray_d[1], ray_d[63]))

        # Spawn quick-scan: grab starting weapon in Room 0 before walking into Tier 3/4 enemies with a Pistol!
        if not self.got_weapon and self.step_count <= 20 and not enemies:
            wpn_pick = [p for p in pickups if p[3] == 8]
            if wpn_pick:
                _, pdist, pang, _ = wpn_pick[0]
                if pang > 0.12:
                    tn = 1
                elif pang < -0.12:
                    tn = 2
                if abs(pang) < 0.75:
                    fb = 1
                f_act = np.array([fb, lr, tn, 0, 0, wpn], dtype=np.int64)
                self.prev_use = False
                return f_act, factored_np_to_multibinary(f_act)
            elif self.spawn_scan_steps < 16:
                self.spawn_scan_steps += 1
                f_act = np.array([0, 0, 1, 0, 0, wpn], dtype=np.int64)
                self.prev_use = False
                return f_act, factored_np_to_multibinary(f_act)

        # Damage-turn reflex if shot from behind / flank with no enemy in front!
        if took_damage and not enemies:
            self.damage_spin_timer = 7
        if self.damage_spin_timer > 0 and not enemies:
            enemy_rays = np.flatnonzero(ray_t == 4)
            if len(enemy_rays) > 0:
                nearest_r = int(enemy_rays[np.argmin(ray_d[enemy_rays])])
                tn = 1 if nearest_r < 32 else 2
            else:
                tn = 1
            lr = 1 if left_clear >= right_clear else 2
            f_act = np.array([0, lr, tn, 0, 0, wpn], dtype=np.int64)
            self.prev_use = False
            return f_act, factored_np_to_multibinary(f_act)

        # ------------------------------------------------------------------
        # 2. Pro DOOM Speed-Tactical Combat:
        #    - Kill low-HP hitscanners (type=2) fast (<320u) & body-blockers (<105u)
        #    - NEVER stop to trade with 500-HP HellKnights / Cacodemons -> run-and-gun
        #      & dodge fireballs at full speed!
        # ------------------------------------------------------------------
        if self.combat_hold > 0:
            self.combat_hold -= 1
        has_loaded_gun = (sel_wpn >= 2) and (sel_ammo > 0)
        must_fight = False
        target_e = None
        if has_loaded_gun and enemies:
            # Prefer killing hitscanners or immediate face-blockers
            for sc, ed, ea, et in enemies:
                if et == 2 and ed < 360.0:
                    must_fight = True
                    target_e = (sc, ed, ea, et)
                    break
                if et == 1 and ed < 220.0:
                    must_fight = True
                    target_e = (sc, ed, ea, et)
                    break
                if ed < 160.0 and abs(ea) < 0.90:
                    must_fight = True
                    target_e = (sc, ed, ea, et)
                    break
            if not must_fight and self.combat_hold > 0:
                sc, ed, ea, et = enemies[0]
                if (et == 2 and ed < 360.0) or ed < 130.0:
                    must_fight = True
                    target_e = (sc, ed, ea, et)

        if must_fight and target_e is not None:
            self.combat_hold = 3
            _, edist, eang, etype = target_e
            if eang > 0.09:
                tn = 1
            elif eang < -0.09:
                tn = 2
            aim_tol = 0.28 if (sel_wpn in (3, 4) or edist < 240.0) else 0.18
            if abs(eang) <= aim_tol and (sel_wpn != 5 or edist > 190.0):
                atk = 1

            # Projectile dodge takes top strafe priority!
            if projectiles:
                _, lat_left, _ = projectiles[0]
                dodge = 2 if lat_left >= 0.0 else 1
                if dodge == 1 and left_clear < 40.0 and right_clear > 52.0:
                    dodge = 2
                elif dodge == 2 and right_clear < 40.0 and left_clear > 52.0:
                    dodge = 1
                lr = dodge
            else:
                if self.strafe_dir == 1 and left_clear < 48.0:
                    self.strafe_dir = 2
                elif self.strafe_dir == 2 and right_clear < 48.0:
                    self.strafe_dir = 1
                lr = self.strafe_dir

            if edist < 140.0 and back_clear > 36.0:
                fb = 2

            f_act = np.array([fb, lr, tn, atk, use, wpn], dtype=np.int64)
            self.prev_use = bool(use)
            return f_act, factored_np_to_multibinary(f_act)

        # Opportunistic hip-fire while navigating if an enemy is right in front of our crosshair!
        if has_loaded_gun and enemies:
            for _, edist, eang, _ in enemies:
                if edist < 420.0 and abs(eang) < 0.22 and (sel_wpn != 5 or edist > 190.0):
                    atk = 1
                    break

        # ------------------------------------------------------------------
        # 3. Opportunistic Visible Key / Weapon / Health Pickup Grab
        # ------------------------------------------------------------------
        if pickups and self.door_wait_timer == 0:
            prio, pdist, pang, pet = pickups[0]
            if prio <= 2 or pdist < 240.0:
                if pang > 0.14:
                    tn = 1
                elif pang < -0.14:
                    tn = 2
                if abs(pang) < 0.70:
                    fb = 1
                f_act = np.array([fb, lr, tn, atk, use, wpn], dtype=np.int64)
                self.prev_use = bool(use)
                return f_act, factored_np_to_multibinary(f_act)

        # ------------------------------------------------------------------
        # 4. Fog-of-War Navigation (Key -> Remote Switch -> Exit -> Frontier)
        # ------------------------------------------------------------------
        blocked = self._build_blocked_grid(env, priv)
        nav_kind, target_xy, target_cell = self._select_nav_target(env, priv, blocked)

        # Only consider a door directly in front if center rays (0, 1, 63) actually hit it!
        front_indices = (0, 1, 63)
        door_in_front = False
        locked_in_front = False
        min_door_d = 9999.0
        for ridx in front_indices:
            if ray_t[ridx] == RAY_DOOR and ray_d[ridx] < min_door_d:
                door_in_front = True
                min_door_d = float(ray_d[ridx])
            elif ray_t[ridx] == RAY_LOCKED and ray_d[ridx] < min_door_d:
                locked_in_front = True
                min_door_d = float(ray_d[ridx])
        center_hits_door = (ray_t[0] in (RAY_DOOR, RAY_LOCKED)) or (
            ray_t[1] in (RAY_DOOR, RAY_LOCKED) and ray_t[63] in (RAY_DOOR, RAY_LOCKED)
        )

        ahead_x = px + math.cos(heading) * min(min_door_d + 16.0, 72.0)
        ahead_y = py + math.sin(heading) * min(min_door_d + 16.0, 72.0)
        ahead_cell = field.world_to_cell(ahead_x, ahead_y)
        ahead_sid = int(field.cell_sector[ahead_cell[0], ahead_cell[1]])
        floor, ceil = priv.get("sector_floor"), priv.get("sector_ceiling")
        ahead_is_opening = False
        if floor is not None and ceil is not None and 0 <= ahead_sid < len(floor):
            opening_h = float(ceil[ahead_sid] - floor[ahead_sid])
            if 2.0 < opening_h < 56.0:
                ahead_is_opening = True

        # Fast door-wait: resume the INSTANT the door opens (>=56u clearance)!
        if self.door_wait_timer > 0:
            still_closed_ahead = center_hits_door and min_door_d < 72.0
            if ahead_is_opening or still_closed_ahead:
                if self.door_wait_timer == 1 and not ahead_is_opening and self.last_door_cell_used is not None:
                    cnt = self.door_use_attempts.get(self.last_door_cell_used, 0) + 1
                    self.door_use_attempts[self.last_door_cell_used] = cnt
                    if cnt >= 2:
                        self.remote_blocked_cells.add(self.last_door_cell_used)
                        # Remote door may have closed again after 150 tics -> allow re-triggering remote switches!
                        self.triggered_switches.clear()
                f_act = np.array([0, 0, 0, 0, 0, wpn], dtype=np.int64)
                self.prev_use = False
                return f_act, factored_np_to_multibinary(f_act)
            else:
                self.door_wait_timer = 0
                self.use_cooldown = min(self.use_cooldown, 2)

        # Handle Switch Interaction (Remote Switch or Exit Switch)
        if nav_kind in ("remote_switch", "exit_switch") and target_xy is not None and target_cell is not None:
            d_sw = math.hypot(target_xy[0] - px, target_xy[1] - py)
            if d_sw < 72.0:
                scx, scy = target_cell
                wall_angles = []
                for (dx, dy), wang in (
                    ((0, 1), math.pi / 2),
                    ((1, 0), 0.0),
                    ((0, -1), -math.pi / 2),
                    ((-1, 0), math.pi),
                ):
                    nx, ny = scx + dx, scy + dy
                    is_solid = (not (0 <= nx < field.w and 0 <= ny < field.h)) or (not field.walkable[nx, ny])
                    if is_solid:
                        wall_angles.append(wang)
                if not wall_angles:
                    wall_angles = [0.0, math.pi / 2, math.pi, -math.pi / 2]

                desired_ang = wall_angles[self.switch_sweep_idx % len(wall_angles)]
                rel_w = (desired_ang - heading + math.pi) % (2.0 * math.pi) - math.pi
                if abs(rel_w) > 0.22:
                    tn = 1 if rel_w > 0 else 2
                    if d_sw > 28.0 and front_clear > 32.0:
                        fb = 1
                else:
                    # Step right up to the switch wall if still > 36u away so DOOM's 64u USE trace hits!
                    if front_clear > 36.0:
                        fb = 1
                    if not self.prev_use and self.use_cooldown == 0:
                        use = 1
                        self.use_cooldown = 3
                        self.switch_sweep_idx += 1
                        if nav_kind == "remote_switch" and self.switch_sweep_idx >= len(wall_angles):
                            self.triggered_switches.add(target_cell)
                            self.remote_blocked_cells.clear()
                            self.switch_sweep_idx = 0
                f_act = np.array([fb, lr, tn, 0, use, wpn], dtype=np.int64)
                self.prev_use = bool(use)
                return f_act, factored_np_to_multibinary(f_act)

        if target_xy is not None:
            d_geo, sin_g, cos_g = field.geodesic_query(px, py, heading, target_xy, blocked_grid=blocked)
            rel_ang = math.atan2(sin_g, cos_g)
        else:
            rel_ang = 0.0

        can_open_front_door = (
            center_hits_door
            and (door_in_front or (locked_in_front and not blocked[ahead_cell[0], ahead_cell[1]]))
            and min_door_d < 72.0
            and abs(rel_ang) < 0.65
            and not ahead_is_opening
        )
        if can_open_front_door and not self.prev_use and self.use_cooldown == 0:
            use = 1
            self.door_wait_timer = 10
            self.use_cooldown = 10
            self.last_door_cell_used = ahead_cell
            f_act = np.array([0, 0, 0, 0, use, wpn], dtype=np.int64)
            self.prev_use = True
            return f_act, factored_np_to_multibinary(f_act)

        if self.stuck_timer > 0:
            self.stuck_timer -= 1
            fb = 2 if self.stuck_timer > 2 else 1
            lr = self.stuck_dir
            tn = self.stuck_dir
            f_act = np.array([fb, lr, tn, 0, 0, wpn], dtype=np.int64)
            self.prev_use = False
            return f_act, factored_np_to_multibinary(f_act)

        if len(self.pos_hist) == 8 and not can_open_front_door:
            moved = math.hypot(px - self.pos_hist[0][0], py - self.pos_hist[0][1])
            if moved < 10.0:
                if center_hits_door and min_door_d < 84.0 and not self.prev_use and self.use_cooldown == 0:
                    use = 1
                    self.door_wait_timer = 10
                    self.use_cooldown = 10
                    self.last_door_cell_used = ahead_cell
                    f_act = np.array([0, 0, 0, 0, use, wpn], dtype=np.int64)
                    self.prev_use = True
                    return f_act, factored_np_to_multibinary(f_act)
                self.stuck_count += 1
                self.stuck_timer = 6
                self.stuck_dir = 1 if (self.stuck_count % 2 == 1) else 2
                self.pos_hist.clear()

        if rel_ang > 0.14:
            tn = 1
        elif rel_ang < -0.14:
            tn = 2

        if abs(rel_ang) < 0.90:
            fb = 1
        elif abs(rel_ang) > 2.3 and back_clear > 48.0:
            fb = 2

        fl_d = float(min(ray_d[4], ray_d[6], ray_d[8]))
        fr_d = float(min(ray_d[56], ray_d[58], ray_d[60]))
        if fb == 1:
            if barrels:
                _, bang = barrels[0]
                lr = 2 if bang >= 0.0 else 1  # sidestep barrel!
            elif fl_d < 28.0 and fr_d > fl_d + 6.0:
                lr = 2
            elif fr_d < 28.0 and fl_d > fr_d + 6.0:
                lr = 1

        if projectiles:
            _, lat_left, _ = projectiles[0]
            lr = 2 if lat_left >= 0.0 else 1

        f_act = np.array([fb, lr, tn, atk, use, wpn], dtype=np.int64)
        self.prev_use = bool(use)
        return f_act, factored_np_to_multibinary(f_act)
