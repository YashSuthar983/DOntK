"""Deterministic, texture-agnostic Dead-Reckoning + Dual-Scale Allocentric EgoMap.

Uses ONLY official doom_env observations and the agent's own previous buttons:
- Heading: exact 3-step Doom turn-ramp table (0.00 deg residual on any map).
- Position: friction-calibrated 4-tic integration of ego velocity (`vitals[14:16]`).
- Allocentric grid: 128x128 (64-unit cells = 8192x8192u) with 10 geometric/semantic channels:
    0: free space      1: wall           2: door            3: locked door
    4: key             5: item           6: enemy (fading)  7: visit count
    8: recent trail    9: tried locked door (resets when a new key is picked up)
- Outputs:
    - Heading-aligned 32x32 fine crop (2048x2048u) + gather indices for learned Neural Map writes
    - Heading-aligned 16x16 wide crop (4096x4096u) for long-range exploration / backtracking
    - Normalized vitals + prev_action + 40-col depth summary vector, circular rays, and entities
"""
from __future__ import annotations

import math

import numpy as np

GRID_SIZE = 128
CELL_SIZE = 64.0
FINE_CROP = 32
WIDE_CROP = 16
WIDE_STEP = 256.0  # 16 * 256 = 4096u wide view
N_FIXED_CH = 10
NUM_RAYS = 64
RAY_RANGE = 1024.0
RECENT_DECAY = 0.97
PAD_IDX = GRID_SIZE * GRID_SIZE

KEY_ENTITY_TYPE = 9
ENEMY_ENTITY_TYPES = (1, 2, 3)

# Exact Doom heading delta (degrees) for (turn[t-2], turn[t-1], turn[t]), turn = LEFT - RIGHT.
_TURN_TABLE = {
    (-1, -1, -1): -14.0625, (-1, -1, 1): 14.0625,
    (-1, 0, -1): -7.03125,  (-1, 0, 1): 7.03125,
    (-1, 1, -1): -14.0625,  (-1, 1, 1): 14.0625,
    (0, -1, -1): -12.3046875, (0, -1, 1): 12.3046875,
    (0, 0, -1): -7.03125,   (0, 0, 1): 7.03125,
    (0, 1, -1): -12.3046875, (0, 1, 1): 12.3046875,
    (1, -1, -1): -14.0625,  (1, -1, 1): 14.0625,
    (1, 0, -1): -7.03125,   (1, 0, 1): 7.03125,
    (1, 1, -1): -14.0625,   (1, 1, 1): 14.0625,
}

VITALS_SCALE = np.array(
    [100.0, 100.0, 7.0, 50.0, 200.0, 50.0, 50.0, 300.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0],
    dtype=np.float32,
)
PREVIOUS_ACTION = slice(len(VITALS_SCALE), len(VITALS_SCALE) + 9)
VEC_DIM = 17 + 9 + 40 + 2  # vitals(17) + prev_action(9) + depth_profile(40) + [has_key, keys_frac]

_RAY_ANG = np.arange(NUM_RAYS, dtype=np.float64) * (2.0 * math.pi / NUM_RAYS)
_RAY_SAMPLES = np.arange(1, 33, dtype=np.float64) * 32.0

_fi, _fj = np.meshgrid(np.arange(FINE_CROP), np.arange(FINE_CROP), indexing="ij")
_FINE_FWD = ((FINE_CROP / 2.0 - 0.5) - _fi).ravel() * CELL_SIZE
_FINE_RIGHT = (_fj - (FINE_CROP / 2.0 - 0.5)).ravel() * CELL_SIZE

_wi, _wj = np.meshgrid(np.arange(WIDE_CROP), np.arange(WIDE_CROP), indexing="ij")
_WIDE_FWD = ((WIDE_CROP / 2.0 - 0.5) - _wi).ravel() * WIDE_STEP
_WIDE_RIGHT = (_wj - (WIDE_CROP / 2.0 - 0.5)).ravel() * WIDE_STEP

# Map ray hit_type (0..5) -> fixed map channel
_HIT_TO_CH = np.array([0, 1, 2, 3, 6, 5], dtype=np.int64)


class EgoMapBuilder:
    """Per-episode observation encoder + dead-reckoned allocentric map (numpy only)."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.grid = np.zeros((N_FIXED_CH, GRID_SIZE * GRID_SIZE + 1), dtype=np.float32)
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0
        self.turns = [0, 0]
        self.step_idx = 0
        self.pvx = 0.0
        self.pvy = 0.0
        self.prev_action = np.zeros(9, dtype=np.float32)
        self.keys_held = 0
        self.pending_locks: list[tuple[int, int, int]] = []  # (cx, cy, due_step)
        self.first = True

    @staticmethod
    def _cell_idx(wx, wy) -> np.ndarray:
        cx = np.floor((np.asarray(wx) + CELL_SIZE * 0.5) / CELL_SIZE).astype(np.int64) + GRID_SIZE // 2
        cy = np.floor((np.asarray(wy) + CELL_SIZE * 0.5) / CELL_SIZE).astype(np.int64) + GRID_SIZE // 2
        ok = (cx >= 0) & (cx < GRID_SIZE) & (cy >= 0) & (cy < GRID_SIZE)
        return np.where(ok, cy * GRID_SIZE + cx, PAD_IDX)

    def step(self, obs: dict, prev_action: np.ndarray | None = None) -> tuple[dict[str, np.ndarray], bool]:
        """Integrate dead-reckoned pose, update map, and return encoded features + entered_new_cell."""
        vit = np.nan_to_num(np.asarray(obs["vitals"], dtype=np.float32).reshape(17), nan=0.0, posinf=0.0, neginf=0.0)
        fwd_v = float(np.clip(vit[14], -40.0, 40.0))
        right_v = float(np.clip(vit[15], -40.0, 40.0))

        used_last = False
        if not self.first and prev_action is not None:
            a = np.asarray(prev_action, dtype=np.float32).reshape(9)
            used_last = bool(a[7] > 0.5)
            t = int(a[4] > 0.5) - int(a[5] > 0.5)
            self.step_idx += 1
            if self.step_idx == 1:
                deg = 0.0
            elif self.step_idx == 2:
                deg = t * (3.515625 if self.turns[-1] != 0 else 1.7578125)
            else:
                deg = _TURN_TABLE.get((self.turns[-2], self.turns[-1], t), 0.0)
            self.turns.append(t)
            del self.turns[:-2]
            self.heading += math.radians(deg)
            ch, sh = math.cos(self.heading), math.sin(self.heading)
            vx = fwd_v * ch + right_v * sh
            vy = fwd_v * sh - right_v * ch
            # Friction-calibrated 4-tic displacement
            self.x += 2.589 * vx + 1.801 * self.pvx
            self.y += 2.589 * vy + 1.801 * self.pvy
            self.pvx, self.pvy = vx, vy
            self.prev_action = a
        else:
            ch, sh = math.cos(self.heading), math.sin(self.heading)
            self.pvx = fwd_v * ch + right_v * sh
            self.pvy = fwd_v * sh - right_v * ch
        self.first = False

        ch, sh = math.cos(self.heading), math.sin(self.heading)
        M = self.grid
        M[6] *= 0.90           # enemy memory decays
        M[8] *= RECENT_DECAY   # recent breadcrumb trail decays

        # 1. Paint 64 rays -> free space + hits
        dist_norm = np.clip(
            np.nan_to_num(np.asarray(obs["rays"]["distance"], dtype=np.float32).reshape(NUM_RAYS), nan=1.0),
            0.0, 1.0,
        )
        dist = dist_norm.astype(np.float64) * RAY_RANGE
        ht = np.clip(np.asarray(obs["rays"]["hit_type"], dtype=np.int64).reshape(NUM_RAYS), 0, 5)

        wa = self.heading + _RAY_ANG
        dx, dy = np.cos(wa), np.sin(wa)
        free = _RAY_SAMPLES[None, :] < (dist[:, None] - 16.0)
        fx = self.x + dx[:, None] * _RAY_SAMPLES[None, :]
        fy = self.y + dy[:, None] * _RAY_SAMPLES[None, :]
        fidx = self._cell_idx(fx[free], fy[free])
        M[0, fidx] = 1.0
        M[1:3, fidx] = 0.0  # open door / cleared wall

        near = free & (_RAY_SAMPLES[None, :] < 192.0)
        nidx = self._cell_idx(fx[near], fy[near])
        M[3, nidx] = 0.0    # only clear remembered locked door from close range
        M[9, nidx] = 0.0    # clear tried-lock mark once opened
        M[4:6, nidx] = 0.0  # clear picked-up key/item

        hit = (ht > 0) & (dist < RAY_RANGE - 1.0)
        if hit.any():
            hidx = self._cell_idx(self.x + dx[hit] * (dist[hit] + 8.0), self.y + dy[hit] * (dist[hit] + 8.0))
            M[_HIT_TO_CH[ht[hit]], hidx] = 1.0

        # 2. Paint visible keys & enemies
        ent = obs["visible_entities"]
        feat = np.nan_to_num(np.asarray(ent["features"], dtype=np.float32).reshape(32, 5), nan=0.0)
        et = np.clip(np.asarray(ent["entity_type"], dtype=np.int64).reshape(32), 0, 11)
        emask = np.asarray(ent["mask"], dtype=np.float32).reshape(32) > 0.5
        ed = feat[:, 0].astype(np.float64) * RAY_RANGE
        ex = self.x + ed * (ch * feat[:, 2] - sh * feat[:, 1])
        ey = self.y + ed * (sh * feat[:, 2] + ch * feat[:, 1])
        eidx = self._cell_idx(ex, ey)
        M[4, eidx[emask & (et == KEY_ENTITY_TYPE)]] = 1.0
        M[6, eidx[emask & np.isin(et, ENEMY_ENTITY_TYPES)]] = 1.0

        # 3. Track tried locked doors & clear them whenever a new key is picked up
        pidx = int(self._cell_idx(self.x, self.y))
        kh = int(np.sum(vit[8:14] > 0.5))
        if kh > self.keys_held:
            M[9, :] = 0.0  # new key -> every previously tried locked door is worth trying again
            self.pending_locks.clear()
            if pidx != PAD_IDX:
                pcx, pcy = pidx % GRID_SIZE, pidx // GRID_SIZE
                for oy in (-1, 0, 1):
                    for ox in (-1, 0, 1):
                        nx, ny = pcx + ox, pcy + oy
                        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                            M[4, ny * GRID_SIZE + nx] = 0.0
        self.keys_held = kh

        if used_last and ht[0] == 3 and dist[0] < 110.0:
            lidx = int(self._cell_idx(self.x + dx[0] * (dist[0] + 8.0), self.y + dy[0] * (dist[0] + 8.0)))
            if lidx != PAD_IDX:
                self.pending_locks.append((lidx % GRID_SIZE, lidx // GRID_SIZE, self.step_idx + 10))
        if self.pending_locks:
            keep = []
            for lcx, lcy, due in self.pending_locks:
                if self.step_idx < due:
                    keep.append((lcx, lcy, due))
                    continue
                for oy in (-1, 0, 1):
                    for ox in (-1, 0, 1):
                        nx, ny = lcx + ox, lcy + oy
                        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                            idx = ny * GRID_SIZE + nx
                            if M[3, idx] > 0.5:
                                M[9, idx] = 1.0
            self.pending_locks = keep

        # 4. Visits & recent trail
        new_cell = pidx != PAD_IDX and M[7, pidx] == 0.0
        if pidx != PAD_IDX:
            M[7, pidx] += 1.0
            M[8, pidx] = 1.0
        M[:, PAD_IDX] = 0.0

        # 5. Heading-aligned fine 32x32 crop & wide 16x16 crop
        fine_idx = self._cell_idx(self.x + _FINE_FWD * ch + _FINE_RIGHT * sh, self.y + _FINE_FWD * sh - _FINE_RIGHT * ch)
        fine_crop = M[:, fine_idx].reshape(N_FIXED_CH, FINE_CROP, FINE_CROP).copy()
        fine_crop[7] = np.log1p(fine_crop[7]) / 3.0

        # Wide crop: sample 2x2 sub-offsets per 256u cell and take max so thin walls/doors aren't missed
        wide_crop = np.zeros((N_FIXED_CH, WIDE_CROP * WIDE_CROP), dtype=np.float32)
        for df in (-64.0, 64.0):
            for dr in (-64.0, 64.0):
                wf = _WIDE_FWD + df
                wr = _WIDE_RIGHT + dr
                widx = self._cell_idx(self.x + wf * ch + wr * sh, self.y + wf * sh - wr * ch)
                np.maximum(wide_crop, M[:, widx], out=wide_crop)
        wide_crop = wide_crop.reshape(N_FIXED_CH, WIDE_CROP, WIDE_CROP)
        wide_crop[7] = np.log1p(wide_crop[7]) / 3.0

        # 6. Compact 1D observation vectors
        # Keep depth_prof constant so render_buffers=False training matches evaluation bit-for-bit
        depth_prof = np.ones(40, dtype=np.float32)

        key_summary = np.array([float(kh > 0), kh / 6.0], dtype=np.float32)
        vec = np.concatenate([np.clip(vit / VITALS_SCALE, -5.0, 5.0), self.prev_action, depth_prof, key_summary]).astype(np.float32)
        rays = np.concatenate([dist_norm[None, :], np.power(1.0 - dist_norm, 4.0)[None, :], np.eye(6, dtype=np.float32)[ht].T], axis=0)
        ents = np.concatenate(
            [np.clip(feat, -10.0, 10.0) * np.array([1.0, 1.0, 1.0, 0.1, 0.1], dtype=np.float32), emask[:, None].astype(np.float32)],
            axis=1,
        ).astype(np.float32)

        features = {
            "vec": vec,
            "rays": rays.astype(np.float32),
            "ents": ents,
            "etype": (et * emask.astype(np.int64)).astype(np.int64),
            "fine_map": fine_crop.astype(np.float16),
            "wide_map": wide_crop.astype(np.float16),
            "crop_idx": fine_idx.astype(np.int64),
            "pos_idx": np.int64(pidx),
            # Legal action mask: never switch away from a loaded weapon
            "wmask": np.float32(0.0 if vit[3] > 0.0 else 1.0),
        }
        return features, bool(new_cell)
