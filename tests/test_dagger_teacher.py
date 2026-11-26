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


def test_objective_teacher_prioritizes_reachable_exit():
    from types import SimpleNamespace

    from training.objective_oracle import ObjectiveOracle

    def distances(goal_x, goal_y, blocked):
        return np.array([[float(goal_x)], [np.inf], [0.0]])

    field = SimpleNamespace(
        walkable=np.ones((3, 1), dtype=bool),
        world_to_cell=lambda x, y: (int(x), int(y)),
        _bfs_from=distances,
    )
    env = SimpleNamespace(
        current_map="MAP01",
        _fields={"MAP01": field},
        manifest={"MAP01": {
            "exit_pos": [2, 0],
            "exit_type": "walkover",
            "keys_placed": [{"lock": 129, "pos": [1, 0]}],
        }},
    )
    teacher = ObjectiveOracle()
    state = {"x": 0, "y": 0, "keys": np.zeros(6)}
    blocked = np.zeros((3, 1), dtype=bool)
    assert teacher._select_nav_target(env, state, blocked) == ("exit_walk", (2, 0), (2, 0))

    blocked[2, 0] = True
    assert teacher._select_nav_target(env, state, blocked) == ("key", (1, 0), (1, 0))


def test_objective_teacher_uses_live_clearance_for_remote_doors():
    from types import SimpleNamespace

    from training.objective_oracle import ObjectiveOracle

    field = SimpleNamespace(
        w=3, h=1,
        cell_sector=np.array([[0], [1], [1]]),
        door_sectors={1: 0},
        door_cells={1: [(1, 0), (2, 0)]},
        world_to_cell=lambda x, y: (int(x), int(y)),
    )
    env = SimpleNamespace(
        current_map="MAP01",
        _fields={"MAP01": field},
        manifest={"MAP01": {"switches": [{"kind": "remote_door", "door_pos": [1, 0]}]}},
    )
    teacher = ObjectiveOracle()
    state = {"keys": np.zeros(6), "sector_floor": np.zeros(2), "sector_ceiling": np.array([128, 55])}
    np.testing.assert_array_equal(teacher._build_blocked_grid(env, state), [[False], [True], [True]])

    state["sector_ceiling"][1] = 56
    assert not teacher._build_blocked_grid(env, state).any()

    state["sector_ceiling"][1] = 40
    np.testing.assert_array_equal(teacher._build_blocked_grid(env, state), [[False], [True], [True]])
