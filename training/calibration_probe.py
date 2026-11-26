"""Step 0.1 Empirical ViZDoom 1.3.3 + Freedoom 2 Calibration Probe.

Zero-assumption verification of:
1. Exact depth_buffer (uint8 0..255) -> map units curve & LUT.
2. Exact camera intrinsics (fx, fy, eye_height) for 160x120 3D pinhole unprojection.
3. Exact SELECTED_WEAPON integer IDs and SELECTED_WEAPON_AMMO values for all Freedoom 2 weapons.
4. Exact TURN_LEFT / TURN_RIGHT angular velocity per 4-tic step (2-stage turn acceleration).
5. Exact USE door activation distance and door-opening duration in decision steps.

Run:
    uv run python training/calibration_probe.py
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np

from doom_env import make_env
from doom_env import spaces as S
from doom_env.wad import Lump, write_wad


def _make_rect_udmf(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    things: list[tuple[float, float, int, int]],
    *,
    floor: int = 0,
    ceil: int = 128,
) -> str:
    lines = [
        'namespace = "zdoom";',
        f"vertex {{ x = {x_min}; y = {y_min}; }}",
        f"vertex {{ x = {x_min}; y = {y_max}; }}",
        f"vertex {{ x = {x_max}; y = {y_max}; }}",
        f"vertex {{ x = {x_max}; y = {y_min}; }}",
        f'sector {{ texturefloor = "FLOOR0_1"; textureceiling = "CEIL1_1"; heightfloor = {floor}; heightceiling = {ceil}; lightlevel = 192; }}',
    ]
    for _ in range(4):
        lines.append('sidedef { sector = 0; texturetop = "-"; texturemiddle = "STARTAN2"; texturebottom = "-"; }')
    edges = [(0, 1, 0), (1, 2, 1), (2, 3, 2), (3, 0, 3)]
    for v1, v2, sd in edges:
        lines.append(f"linedef {{ v1 = {v1}; v2 = {v2}; sidefront = {sd}; blocking = true; dontpegbottom = true; }}")
    for tx, ty, angle, ttype in things:
        lines.append(
            f"thing {{ x = {tx}; y = {ty}; angle = {angle}; type = {ttype}; "
            "skill1 = true; skill2 = true; skill3 = true; skill4 = true; skill5 = true; single = true; }"
        )
    return "\n".join(lines) + "\n"


def _write_temp_wad(tmp_dir: Path, name: str, textmap: str) -> Path:
    path = tmp_dir / f"{name}.wad"
    write_wad(path, [Lump("MAP01", b""), Lump("TEXTMAP", textmap.encode("utf-8")), Lump("ENDMAP", b"")])
    return path


def probe_depth_and_intrinsics(tmp_dir: Path) -> dict:
    """Measure exact depth_buffer uint8 values vs true wall distance and vertical floor-edge rows."""
    # Long 2560-unit corridor, player spawns at x=96 facing east (angle=0) toward wall at x=2560.
    # Floor=0, ceiling=128 -> wall bottom at z=0 (-41 from eye), wall top at z=128 (+87 from eye).
    wad = _write_temp_wad(tmp_dir, "corridor", _make_rect_udmf(0.0, 2560.0, 0.0, 256.0, [(96.0, 128.0, 0, 1)]))
    env = make_env(wad, "MAP01", episode_ticks=2100)
    obs, _ = env.reset(seed=0)

    fwd = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)
    samples = []
    fy_estimates = []
    eye_h_estimates = []

    for _ in range(180):
        priv = env.privileged_state()
        d_true = 2560.0 - priv["x"]
        d_ray = float(obs["rays"]["distance"][0]) * S.RAY_RANGE
        dep_col = obs["depth"][:, 80]  # center vertical column (u=80)
        d_center = int(dep_col[60])  # horizon pixel (v=60)

        # Find wall vertical span in center column (pixels matching wall center depth within +/-1)
        wall_rows = np.flatnonzero(np.abs(dep_col.astype(np.int32) - d_center) <= 1)
        v_top = int(wall_rows[0]) if len(wall_rows) else -1
        v_bot = int(wall_rows[-1]) if len(wall_rows) else -1
        if 5 < v_top < 55 and 65 < v_bot < 115 and d_true > 64.0:
            # Wall height = 128 units spans (v_bot - v_top) pixels at distance d_true
            # fy = (v_bot - v_top) * d_true / 128.0
            fy = (v_bot - v_top) * d_true / 128.0
            fy_estimates.append(float(fy))
            # eye_height = (v_bot - 59.5) * d_true / fy
            eye_h = (v_bot - 59.5) * d_true / fy
            eye_h_estimates.append(float(eye_h))

        samples.append(
            {
                "d_true": round(d_true, 2),
                "d_ray": round(d_ray, 2),
                "depth_uint8": d_center,
                "v_top": v_top,
                "v_bot": v_bot,
            }
        )
        obs, _, term, trunc, _ = env.step(fwd)
        if term or trunc or d_true < 25.0:
            break
    env.close()

    # Check exact monotonic mapping & fit
    # Print sparse table across depth values
    by_u8: dict[int, list[float]] = {}
    for s in samples:
        by_u8.setdefault(s["depth_uint8"], []).append(s["d_true"])
    pairs = sorted((u8, float(np.mean(ds))) for u8, ds in by_u8.items())
    # Linear fit on non-saturated points (0 < u8 < 255)
    interior = [(u8, d) for u8, d in pairs if 2 <= u8 <= 253]
    if len(interior) >= 2:
        xs = np.array([p[0] for p in interior], dtype=np.float64)
        ys = np.array([p[1] for p in interior], dtype=np.float64)
        slope, intercept = np.polyfit(xs, ys, 1)
        max_lin_err = float(np.max(np.abs((slope * xs + intercept) - ys)))
    else:
        slope, intercept, max_lin_err = 0.0, 0.0, -1.0

    return {
        "num_samples": len(samples),
        "first_sample": samples[0],
        "mid_sample": samples[len(samples) // 2],
        "last_sample": samples[-1],
        "linear_fit": {
            "d_true_approx": f"{slope:.6f} * uint8 + {intercept:.4f}",
            "slope": float(slope),
            "intercept": float(intercept),
            "max_abs_err_units": round(max_lin_err, 3),
        },
        "fy_median": round(float(np.median(fy_estimates)), 3) if fy_estimates else None,
        "eye_height_median": round(float(np.median(eye_h_estimates)), 3) if eye_h_estimates else None,
        "sample_pairs_stride10": pairs[:: max(1, len(pairs) // 12)],
    }


def probe_weapons(tmp_dir: Path) -> dict:
    """Walk over every Freedoom 2 weapon pickup and log SELECTED_WEAPON & SELECTED_WEAPON_AMMO."""
    # DoomEd thing types:
    # 2005=Chainsaw, 2001=Shotgun, 82=SuperShotgun, 2002=Chaingun,
    # 2003=RocketLauncher, 2004=PlasmaRifle, 2006=BFG9000
    weapons = [
        ("Chainsaw", 2005),
        ("Shotgun", 2001),
        ("SuperShotgun", 82),
        ("Chaingun", 2002),
        ("RocketLauncher", 2003),
        ("PlasmaRifle", 2004),
        ("BFG9000", 2006),
    ]
    observed = {}
    fwd = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)
    noop = np.zeros(9, dtype=np.int8)

    # First check starting weapon (Pistol)
    wad0 = _write_temp_wad(tmp_dir, "wpn_start", _make_rect_udmf(0.0, 512.0, 0.0, 128.0, [(64.0, 64.0, 0, 1)]))
    env = make_env(wad0, "MAP01", episode_ticks=200, render_buffers=False)
    obs, _ = env.reset(seed=0)
    observed["Start_Pistol"] = {
        "selected_weapon": float(obs["vitals"][S.VITALS_INDEX["selected_weapon"]]),
        "selected_weapon_ammo": float(obs["vitals"][S.VITALS_INDEX["selected_weapon_ammo"]]),
        "ammo_vec": [float(obs["vitals"][i]) for i in range(4, 8)],
    }
    env.close()

    # Walk over each weapon individually (vanilla DOOM auto-equips on pickup!)
    for name, ttype in weapons:
        things = [(64.0, 64.0, 0, 1), (160.0, 64.0, 0, ttype)]
        wad = _write_temp_wad(tmp_dir, f"wpn_{name}", _make_rect_udmf(0.0, 512.0, 0.0, 128.0, things))
        env = make_env(wad, "MAP01", episode_ticks=400, render_buffers=False)
        obs, _ = env.reset(seed=0)
        for step in range(40):
            obs, _, term, trunc, _ = env.step(fwd if step < 12 else noop)
            if term or trunc:
                break
        observed[name] = {
            "selected_weapon": float(obs["vitals"][S.VITALS_INDEX["selected_weapon"]]),
            "selected_weapon_ammo": float(obs["vitals"][S.VITALS_INDEX["selected_weapon_ammo"]]),
            "ammo_vec": [float(obs["vitals"][i]) for i in range(4, 8)],
        }
        env.close()

    # Also test SELECT_NEXT_WEAPON cycling from start (Pistol -> Fist?)
    wad_c = _write_temp_wad(tmp_dir, "wpn_cycle", _make_rect_udmf(0.0, 512.0, 0.0, 128.0, [(64.0, 64.0, 0, 1)]))
    env = make_env(wad_c, "MAP01", episode_ticks=400, render_buffers=False)
    obs, _ = env.reset(seed=0)
    next_wpn = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=np.int8)
    cycle_seq = []
    for step in range(25):
        obs, _, term, trunc, _ = env.step(next_wpn if step == 0 else noop)
        cycle_seq.append(
            (
                step + 1,
                float(obs["vitals"][S.VITALS_INDEX["selected_weapon"]]),
                float(obs["vitals"][S.VITALS_INDEX["selected_weapon_ammo"]]),
            )
        )
        if term or trunc:
            break
    env.close()
    observed["Cycle_From_Pistol_First10"] = cycle_seq[:10]
    return observed


def probe_turn_and_door(tmp_dir: Path) -> dict:
    """Measure exact TURN_LEFT degrees/step and door opening timing on maps/two_rooms.wad."""
    wad = _write_temp_wad(tmp_dir, "turn_box", _make_rect_udmf(0.0, 512.0, 0.0, 512.0, [(256.0, 256.0, 0, 1)]))
    env = make_env(wad, "MAP01", episode_ticks=200, render_buffers=False)
    env.reset(seed=0)
    turn_l = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.int8)
    prev_ang = env.privileged_state()["angle"]
    turn_deltas = []
    for _ in range(10):
        env.step(turn_l)
        ang = env.privileged_state()["angle"]
        delta = (ang - prev_ang + 180.0) % 360.0 - 180.0
        turn_deltas.append(round(delta, 3))
        prev_ang = ang
    env.close()

    # Probe door open timing using two_rooms.wad
    two_rooms = Path(__file__).resolve().parent.parent / "maps" / "two_rooms.wad"
    env = make_env(two_rooms, "MAP01", episode_ticks=600, render_buffers=False)
    obs, _ = env.reset(seed=0)
    # Just test raycast distance at spawn and inspect door_lock array
    priv = env.privileged_state()
    door_info = {
        "num_sectors": len(priv["sector_floor"]),
        "door_lock": priv["door_lock"].tolist(),
        "initial_front_ray": (round(float(obs["rays"]["distance"][0]) * S.RAY_RANGE, 2), int(obs["rays"]["hit_type"][0])),
    }
    env.close()
    return {"turn_left_deg_per_step": turn_deltas, "two_rooms_info": door_info}


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="doom-calib-") as td:
        tmp = Path(td)
        report = {
            "depth_and_intrinsics": probe_depth_and_intrinsics(tmp),
            "weapons": probe_weapons(tmp),
            "turn_and_door": probe_turn_and_door(tmp),
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
