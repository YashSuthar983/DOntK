import sys
from pathlib import Path

import numpy as np
import pytest

from doom_env.sensors import RAY_DOOR

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
pytest.importorskip("torch")

from training.frontier_oracle import factored_np_to_multibinary
from training.train_dagger_ppo import SubprocDoomVecEnv


def test_teacher_retries_use_after_student_skips_door():
    wad = ROOT / "maps" / "two_rooms.wad"
    env = SubprocDoomVecEnv(
        [str(wad)], num_envs=1, num_workers=1, episode_ticks=2100,
        gamma=0.995, use_oracle=True, seed=42,
    )
    try:
        packet = env.reset()
        for _ in range(80):
            label = packet["oracle_act"][0]
            if label[4]:
                break
            packet = env.step(factored_np_to_multibinary(label)[None, :])
            assert not packet["done"][0]
        else:
            raise AssertionError("The teacher never reached the door")

        packet = env.step(np.zeros((1, 9), dtype=np.int8))
        assert packet["ray_hit_type"][0, 0] == RAY_DOOR
        for _ in range(4):
            retry = packet["oracle_act"][0]
            if retry[4]:
                break
            packet = env.step(factored_np_to_multibinary(retry)[None, :])
        else:
            raise AssertionError("The teacher did not retry USE after the skipped action")

        packet = env.step(factored_np_to_multibinary(retry)[None, :])
        assert packet["oracle_act"][0, 4] == 0
    finally:
        env.close()
