"""Live DOOM Gameplay Runner & Video Replay Recorder for AEGIS-3D v2.

Run headless (records MP4/GIF replays + prints live level-by-level stats):
    PYTHONPATH=.:env:submission .venv/bin/python training/watch_live.py

Run with a live real-time ViZDoom desktop window:
    PYTHONPATH=.:env:submission .venv/bin/python training/watch_live.py --window
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import vizdoom as vd

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_ROOT / "env"), str(_ROOT / "submission")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from doom_env import action_space, make_env, observation_space
from submission.agent import DoomAgent

ACT_LABELS = ("FWD", "BAK", "SL", "SR", "TL", "TR", "ATK", "USE", "WPN")


def _try_save_video(frames: list[np.ndarray], out_path: Path, fps: int = 12) -> str:
    import subprocess
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    mp4 = out_path.with_suffix(".mp4")
    try:
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{w}x{h}", "-pix_fmt", "rgb24", "-r", str(fps),
            "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "20", "-movflags", "+faststart", str(mp4),
        ]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        assert p.stdin is not None
        for f in frames:
            p.stdin.write(np.ascontiguousarray(f, dtype=np.uint8).tobytes())
        p.stdin.close()
        p.wait(timeout=30)
        if mp4.is_file() and mp4.stat().st_size > 1000:
            return str(mp4)
    except Exception:
        pass
    return "(video save failed)"


_FONT_3X5: dict[str, tuple[str, ...]] = {
    "0": ("111", "101", "101", "101", "111"), "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"), "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"), "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"), "7": ("111", "001", "010", "010", "010"),
    "8": ("111", "101", "111", "101", "111"), "9": ("111", "101", "111", "001", "111"),
    "A": ("010", "101", "111", "101", "101"), "B": ("110", "101", "110", "101", "110"),
    "C": ("111", "100", "100", "100", "111"), "D": ("110", "101", "101", "101", "110"),
    "E": ("111", "100", "110", "100", "111"), "F": ("111", "100", "110", "100", "100"),
    "G": ("111", "100", "101", "101", "111"), "H": ("101", "101", "111", "101", "101"),
    "I": ("111", "010", "010", "010", "111"), "J": ("001", "001", "001", "101", "111"),
    "K": ("101", "101", "110", "101", "101"), "L": ("100", "100", "100", "100", "111"),
    "M": ("101", "111", "101", "101", "101"), "N": ("111", "101", "101", "101", "101"),
    "O": ("111", "101", "101", "101", "111"), "P": ("111", "101", "111", "100", "100"),
    "Q": ("111", "101", "101", "111", "001"), "R": ("110", "101", "110", "101", "101"),
    "S": ("111", "100", "111", "001", "111"), "T": ("111", "010", "010", "010", "010"),
    "U": ("101", "101", "101", "101", "111"), "V": ("101", "101", "101", "101", "010"),
    "W": ("101", "101", "101", "111", "101"), "X": ("101", "101", "010", "101", "101"),
    "Y": ("101", "101", "010", "010", "010"), "Z": ("111", "001", "010", "100", "111"),
    ":": ("000", "010", "000", "010", "000"), ".": ("000", "000", "000", "000", "010"),
    "/": ("001", "001", "010", "100", "100"), "=": ("000", "111", "000", "111", "000"),
    "|": ("010", "010", "010", "010", "010"), "!": ("010", "010", "010", "000", "010"),
    "[": ("110", "100", "100", "100", "110"), "]": ("011", "001", "001", "001", "011"),
    "-": ("000", "000", "111", "000", "000"), "_": ("000", "000", "000", "000", "111"),
    " ": ("000", "000", "000", "000", "000"),
}


def _draw_bitmap_text(canvas: np.ndarray, text: str, x0: int, y0: int, color=(235, 255, 235), px: int = 2) -> None:
    h, w = canvas.shape[:2]
    x = x0
    for ch in text.upper():
        glyph = _FONT_3X5.get(ch, _FONT_3X5[" "])
        if x + 3 * px >= w:
            break
        for r, row in enumerate(glyph):
            for c, bit in enumerate(row):
                if bit == "1":
                    y1, y2 = y0 + r * px, min(h, y0 + (r + 1) * px)
                    x1, x2 = x + c * px, min(w, x + (c + 1) * px)
                    canvas[y1:y2, x1:x2] = color
        x += 4 * px


def _annotate(rgb: np.ndarray, text_lines: list[str], scale: int = 3) -> np.ndarray:
    """Upscale 120x160 RGB frame and draw a crisp HUD status banner on top (zero-dep)."""
    up = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
    bar_h = 36
    canvas = np.zeros((up.shape[0] + bar_h, up.shape[1], 3), dtype=np.uint8)
    canvas[:bar_h] = (20, 24, 28)
    canvas[bar_h:] = up
    for i, line in enumerate(text_lines[:2]):
        _draw_bitmap_text(canvas, line, 6, 5 + i * 16, color=(235, 255, 235) if i == 0 else (180, 230, 255), px=2)
    return canvas


def run_level(
    agent: DoomAgent,
    wad: str,
    map_name: str,
    episode_ticks: int,
    *,
    seed: int = 0,
    window: bool = False,
    save_video: Path | None = None,
) -> dict:
    env = make_env(wad, map_name, episode_ticks=episode_ticks, render_buffers=True)
    if window:
        # Re-open with visible window for live desktop viewing
        env._game.close()
        env._game.set_window_visible(True)
        env._game.set_screen_resolution(vd.ScreenResolution.RES_640X480)
        env._game.init()
    try:
        obs, _ = env.reset(seed=seed)
        agent.reset()
        frames: list[np.ndarray] = []
        steps = 0
        opened_doors: set[int] = set()
        unlocked_doors: set[int] = set()
        t0 = time.perf_counter()

        while True:
            action = agent.act(obs)
            obs, _, term, trunc, _ = env.step(action)
            steps += 1
            priv = env.privileged_state()

            fl, cl, lk = priv["sector_floor"], priv["sector_ceiling"], priv["door_lock"]
            if fl is not None and cl is not None and lk is not None:
                om = (cl - fl) >= 56.0
                for sid in np.flatnonzero(om & (lk == 0)):
                    opened_doors.add(int(sid))
                for sid in np.flatnonzero(om & (lk > 0)):
                    unlocked_doors.add(int(sid))

            if save_video is not None and (steps % 2 == 0 or term or trunc):
                act_str = " ".join(ACT_LABELS[i] for i in range(9) if action[i]) or "IDLE"
                status = "EXIT REACHED!" if priv["exit_reached"] else ("DEAD" if priv["dead"] else ("TIMEOUT" if trunc else "PLAYING"))
                l1 = f"{Path(wad).name}:{map_name} | st={steps} HP={priv['health']:.0f} K={priv['kills']} Keys={int(priv['keys'].sum())} Doors={len(opened_doors)}/{len(unlocked_doors)}"
                l2 = f"ACT=[{act_str}] | {status}"
                frames.append(_annotate(obs["rgb"], [l1, l2]))

            if window:
                time.sleep(0.06)

            if term or trunc:
                break

        vid_str = ""
        if save_video is not None and frames:
            # Hold final frame briefly
            frames.extend([frames[-1]] * 12)
            vid_str = _try_save_video(frames, save_video)

        return {
            "level": f"{Path(wad).name}:{map_name}",
            "exit": bool(priv["exit_reached"]),
            "dead": bool(priv["dead"]),
            "steps": steps,
            "hp": round(float(priv["health"]), 1),
            "kills": int(priv["kills"]),
            "keys": int(np.sum(priv["keys"] > 0.5)),
            "doors": len(opened_doors),
            "unlocked": len(unlocked_doors),
            "wall_s": round(time.perf_counter() - t0, 1),
            "video": vid_str,
        }
    finally:
        env.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", action="store_true", help="Open real-time ViZDoom desktop window")
    ap.add_argument("--out-dir", default="live_replays", help="Directory to save MP4 replays")
    args = ap.parse_args()

    agent = DoomAgent(observation_space(), action_space(), "cpu")
    out_dir = Path(args.out_dir)
    freedoom2 = str(Path(vd.__file__).parent / "freedoom2.wad")

    playlist = [
        ("maps/two_rooms.wad", "MAP01", 2100, 0, "01_two_rooms"),
        ("training/wads/eval_locked/tier1_eval.wad", "MAP01", 2100, 42, "02_tier1_map01"),
        ("training/wads/eval_locked/tier1_eval.wad", "MAP02", 2100, 43, "03_tier1_map02"),
        ("training/wads/eval_locked/tier1_eval.wad", "MAP03", 2100, 44, "04_tier1_map03"),
        ("training/wads/eval_locked/tier2_eval.wad", "MAP01", 2800, 42, "05_tier2_map01"),
        ("training/wads/eval_locked/tier2_eval.wad", "MAP02", 2800, 43, "06_tier2_map02"),
        ("training/wads/eval_locked/tier3_eval.wad", "MAP01", 3500, 42, "07_tier3_map01"),
        (freedoom2, "MAP01", 2800, 0, "08_freedoom2_map01"),
    ]

    print(f"=== LIVE AEGIS-3D v2 DOOM SHOWCASE (decode_temps={agent.decode_temps}) ===", flush=True)
    exits = 0
    for wad, mname, ticks, seed, tag in playlist:
        res = run_level(
            agent,
            wad,
            mname,
            ticks,
            seed=seed,
            window=args.window,
            save_video=None if args.window else (out_dir / tag),
        )
        exits += int(res["exit"])
        mark = "EXIT!" if res["exit"] else ("DEAD " if res["dead"] else "TIME ")
        print(
            f"[{mark}] {res['level']:24s} | st={res['steps']:4d} | hp={res['hp']:5.1f} | "
            f"kills={res['kills']:2d} | keys={res['keys']} | doors={res['doors']} (unlk={res['unlocked']}) | {res['video']}",
            flush=True,
        )
    print(f"\nSummary: {exits}/{len(playlist)} levels exited!", flush=True)


if __name__ == "__main__":
    main()
