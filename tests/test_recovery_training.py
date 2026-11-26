"""Recovery demonstrations must be verified in the engine before training."""

from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "env"), str(ROOT / "submission")]
torch = pytest.importorskip("torch")

from aegis3d import AEGIS3D
from training.train_recovery import collect, load_actions
from training.wad_oracle import FastDoomPackEnv
from training.apex_teacher import ApexTeacher


def test_recovery_replay_requires_an_engine_confirmed_exit(tmp_path):
    wad = ROOT / "maps" / "two_rooms.wad"
    seed = 42
    row = dict(id="two_rooms", wad=str(wad), map="MAP01", tier=1, seed=seed,
               ticks=2100, split="training")
    path = tmp_path / "two_rooms.npz"
    empty = np.zeros((0, 9), dtype=np.int8)
    np.savez(path, prefix=empty, recovery=np.zeros((1, 9), dtype=np.int8))
    model = AEGIS3D().eval()
    with pytest.raises(ValueError, match="unexpected termination"):
        collect(model, [row], tmp_path)

    env = FastDoomPackEnv(wad, episode_ticks=row["ticks"], compute_spatial_targets=False)
    teacher = ApexTeacher()
    buttons = []
    try:
        reset = env.env.reset
        with patch.object(env.env, "reset", lambda **_: reset(seed=seed)):
            obs, _ = env.reset(seed=seed)
        for _ in range(525):
            _, action = teacher.act(obs, env)
            buttons.append(action)
            obs, _, term, trunc, info = env.step(action)
            if term or trunc:
                break
        assert info["exit_reached"]
    finally:
        env.close()

    replay = np.asarray(buttons, dtype=np.int8)
    np.savez(path, prefix=replay[:8], recovery=replay[8:])
    features, targets, available, anchors, report = collect(model, [row], tmp_path)
    assert len(features) == len(targets) == len(available) == len(replay) - 8
    assert len(anchors) == 3
    assert report[0]["prefix_steps"] == 8


def test_recovery_rejects_conflicting_buttons(tmp_path):
    path = tmp_path / "invalid.npz"
    buttons = np.zeros((1, 9), dtype=np.int8)
    buttons[0, :2] = 1
    np.savez(path, prefix=np.zeros((0, 9), dtype=np.int8), recovery=buttons)
    with pytest.raises(ValueError, match="conflicting movement buttons"):
        load_actions(path)
