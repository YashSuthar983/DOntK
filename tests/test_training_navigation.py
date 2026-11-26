"""Live switch events, dynamic visibility and training/evaluation regressions."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from doom_env.wad import Lump, write_wad
from doom_env.wrapper import IWAD
from fixtures import fixture_wad
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.doomforge import CellSpec, GridUDMFBuilder, SwitchSpec
from training.wad_oracle import FastDoomPackEnv, _frustum_affordance


def switch_wad(path, heading, gap, sidecar):
    builder = GridUDMFBuilder()
    for x in range(5):
        builder.set_cell(x, 0, CellSpec(x + 1, is_door=x == 2, door_tag=17 if x == 2 else 0))
    builder.add_switch(SwitchSpec(0, 0, "N", special=12, tag=17))
    builder.add_switch(SwitchSpec(4, 0, "E", special=243))
    builder.add_thing(0, 0, 1, angle=heading)
    text = builder.compile_textmap().replace("heightceiling = 0;", f"heightceiling = {gap};")
    write_wad(path, [Lump("MAP01", b""), Lump("TEXTMAP", text.encode()), Lump("ENDMAP", b"")])
    if sidecar:
        meta = dict(map_name="MAP01", exit_type="switch", exit_pos=[320, 32],
                    switches=[dict(kind="remote_door", tag=17, pos=[32, 32])])
        path.with_suffix(".json").write_text(json.dumps([meta]))


@pytest.mark.parametrize("heading,press,gap,sidecar", [
    (90, True, 0, False), (90, True, 12, True), (270, True, 0, False), (90, False, 0, False),
])
def test_switch_activation_and_sight(tmp_path, heading, press, gap, sidecar):
    wad = tmp_path / "switch.wad"
    switch_wad(wad, heading, gap, sidecar)
    env = FastDoomPackEnv(wad)
    try:
        meta = env.manifest["MAP01"]
        switch = next(sw for sw in meta["switches"] if sw["kind"] == "remote_door")
        sid, = switch["door_sectors"]
        activated = press and heading == 90
        for _ in range(2):
            env.reset(map_name="MAP01", seed=0)
            initial = env.env.privileged_state()
            assert not env._remote_door_is_open(initial, switch)
            def visible():
                return _frustum_affordance(32, 32, 0, meta, geometry=env.env._geometry)[:, 0].any()
            if gap == 0:
                assert not visible()
            action = np.zeros(9, np.int8)
            action[7] = press
            _, reward, *_ = env.step(action)
            assert reward == pytest.approx(1.496 if activated else -0.004)
            assert bool(env._rewarded_switch_use) == activated
            ceiling = env.env.privileged_state()["sector_ceiling"][sid]
            assert (ceiling > gap) == activated
            _, held_reward, *_ = env.step(action)
            assert held_reward == pytest.approx(-0.004)
            for _ in range(8):
                env.step(action)
            assert env._remote_door_is_open(env.env.privileged_state(), switch) == activated
            if activated:
                assert visible()
    finally:
        env.close()


def test_freedoom_tagged_doors_and_lock_visibility():
    env = FastDoomPackEnv(IWAD, map_subset=["MAP01", "MAP08"])
    try:
        for name in env.map_names:
            env.reset(map_name=name, seed=0)
            state = env.env.privileged_state()
            switches = [s for s in env.manifest[name]["switches"] if s["kind"] == "remote_door"]
            assert switches and all(s["door_sectors"] for s in switches)
            assert all(not env._remote_door_is_open(state, s) for s in switches)
    finally:
        env.close()
    env = FastDoomPackEnv(fixture_wad("#######\n#P..B.E#\n#######"))
    try:
        env.reset(seed=0)
        meta = {"locked_doors": [{"lock": 130, "pos": [288, 96]}], "exit_pos": [416, 96]}
        flags = _frustum_affordance(96, 96, 0, meta, geometry=env.env._geometry, field=env._fields["MAP01"])
        assert flags[:, 3].sum() == 1 and flags[:, 0].sum() == 0
    finally:
        env.close()


def test_legacy_checkpoint_preserves_optimizer_and_yaw_gradient():
    torch = pytest.importorskip("torch")
    from submission.aegis3d import AEGIS3D
    from training.checkpoint import restore_optimizer

    model = AEGIS3D()
    yaw = model.ray_odom.turn_yaw.weight
    name = "ray_odom.turn_yaw.weight"
    old_weights = dict(model.state_dict())
    del old_weights[name]
    with torch.no_grad():
        yaw.fill_(1)
    model.load_state_dict(old_weights, strict=True)
    assert torch.count_nonzero(yaw) == 0
    with pytest.raises(RuntimeError, match="Missing key"):
        model.load_state_dict({k: v for k, v in old_weights.items() if k != "ray_odom.hit_emb.weight"})
    legacy = torch.optim.AdamW([p for p in model.parameters() if p is not yaw])
    for i, p in enumerate(legacy.param_groups[0]["params"]):
        p.grad = torch.full_like(p, (i + 1) / 100)
    legacy.step()
    new = torch.optim.AdamW(model.parameters())
    restore_optimizer(new, model, {"model": old_weights, "opt": legacy.state_dict()})
    assert yaw not in new.state
    for p, state in legacy.state.items():
        for key, value in state.items():
            torch.testing.assert_close(new.state[p][key], value)
    history = torch.zeros(2, 4, 9)
    history[0, 0, 4] = history[1, 1, 5] = 1
    _, _, odom = model.ray_odom(torch.ones(2, 4, 64), torch.zeros(2, 64, dtype=torch.long),
                                torch.zeros(2, 3), history)
    odom[:, 2].sum().backward()
    assert yaw.grad[0, 0] != 0 and yaw.grad[0, 3] != 0


def test_per_level_seed_counts_and_calibration(tmp_path):
    pytest.importorskip("torch")
    from training.eval_and_calibrate import calibration_rank
    from training.scorer_eval import evaluate_suite

    wad = fixture_wad("#####\n#P.z#\n#####")
    class Idle:
        def reset(self):
            pass
        def act(self, obs):
            return np.zeros(9, np.int8)
    suite = [dict(wad=wad, episode_ticks=12, seed=i) for i in (17, 18)]
    report = evaluate_suite(Idle(), suite, out_json=tmp_path / "report.json")
    level, = report["levels"]
    assert level["episodes"] == 2 and level["exits"] == 0 and level["seeds"] == [17, 18]
    assert report["min_level_exit_rate"] == 0
    assert [r["seed"] for r in report["maps"]] == [17, 18]
    assert all(r["wad"] == str(Path(wad).resolve()) for r in report["maps"])
    with pytest.raises(ValueError, match="Overlapping"):
        evaluate_suite(Idle(), suite + suite)
    easy_only = dict(levels=[dict(exit_rate=0), dict(exit_rate=1)], exit_rate=.5, weighted_total_score=100)
    both = dict(levels=[dict(exit_rate=.4), dict(exit_rate=.4)], exit_rate=.4, weighted_total_score=80)
    assert calibration_rank(both, "clear") > calibration_rank(easy_only, "clear")
    assert calibration_rank(both, "score") < calibration_rank(easy_only, "score")


def test_lock_all_door_requires_all_three_keys(tmp_path):
    builder = GridUDMFBuilder()
    for x in range(4):
        builder.set_cell(x, 0, CellSpec(x + 1, is_door=x == 2, door_lock=101 if x == 2 else 0))
    builder.add_switch(SwitchSpec(3, 0, "E", special=243))
    builder.add_thing(0, 0, 1, angle=0)
    wad = tmp_path / "lock_all.wad"
    write_wad(wad, [Lump("MAP01", b""), Lump("TEXTMAP", builder.compile_textmap().encode()), Lump("ENDMAP", b"")])
    env = FastDoomPackEnv(wad)
    try:
        env.reset(map_name="MAP01", seed=0)
        base = env.env.privileged_state()
        door_cells = next(iter(env._fields["MAP01"].door_cells.values()))

        # Only blue card: LOCK_ALL (101) door must remain blocked
        one_key = {**base, "keys": np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)}
        blocked_one = env._locked_blocked_grid(one_key)
        assert blocked_one is not None and all(blocked_one[cx, cy] for cx, cy in door_cells)

        # Two colors (blue + yellow): still blocked
        two_keys = {**base, "keys": np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)}
        blocked_two = env._locked_blocked_grid(two_keys)
        assert blocked_two is not None and all(blocked_two[cx, cy] for cx, cy in door_cells)

        # All three colors (blue card + yellow skull + red card): passable
        all_keys = {**base, "keys": np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)}
        assert env._locked_blocked_grid(all_keys) is None
    finally:
        env.close()
