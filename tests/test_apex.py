"""Unit and integration tests for EgoMapBuilder, ApexNet, ApexTeacher, and DoomAgent apex mode."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "env"), str(ROOT / "submission")):
    if p not in sys.path:
        sys.path.insert(0, p)

from agent import DoomAgent
from apex_net import ApexNet, apex_factored_to_multibinary, blocked_edge_buttons
from doom_env import action_space, make_env, observation_space
from doom_env.sensors import RAY_DOOR
from egomap import EgoMapBuilder, FINE_CROP, N_FIXED_CH, PREVIOUS_ACTION, VEC_DIM, WIDE_CROP
from training.apex_teacher import ApexTeacher, USE_RANGE, dir_rel_to_bin
from training.train_apex import _legal_teacher_actions, _sample_factored
from training.wad_oracle import FastDoomPackEnv


def test_egomap_dead_reckoning_heading_and_crops():
    env = make_env(ROOT / "maps" / "two_rooms.wad", "MAP01", episode_ticks=200)
    ego = EgoMapBuilder()
    try:
        obs, _ = env.reset(seed=0)
        ego.reset()
        p0 = env.privileged_state()
        h0 = p0["angle"]
        feats, _ = ego.step(obs, None)
        assert feats["vec"].shape == (VEC_DIM,)
        assert feats["fine_map"].shape == (N_FIXED_CH, FINE_CROP, FINE_CROP)
        assert feats["wide_map"].shape == (N_FIXED_CH, WIDE_CROP, WIDE_CROP)
        assert feats["wmask"] == 0.0  # starts with loaded pistol (50 ammo)

        # Turn left 6 steps, then right 4 steps, then forward 5 steps
        act = np.zeros(9, dtype=np.int8)
        for i in range(15):
            act[:] = 0
            if i < 6:
                act[4] = 1
            elif i < 10:
                act[5] = 1
            else:
                act[0] = 1
            obs, _, term, trunc, _ = env.step(act)
            feats, _ = ego.step(obs, act)
            if term or trunc:
                break
        p1 = env.privileged_state()
        true_dh = ((p1["angle"] - h0 + 180.0) % 360.0) - 180.0
        est_dh = ((np.degrees(ego.heading) + 180.0) % 360.0) - 180.0
        assert abs(true_dh - est_dh) < 1.0, f"Heading drift too large: true={true_dh:.2f}, est={est_dh:.2f}"
    finally:
        env.close()


def test_apex_net_forward_backward_and_export(tmp_path):
    model = ApexNet(include_training_heads=True)
    state = model.init_state(2, "cpu")
    obs = {
        "vec": torch.randn(2, VEC_DIM),
        "rays": torch.randn(2, 8, 64),
        "ents": torch.randn(2, 32, 6),
        "etype": torch.zeros(2, 32, dtype=torch.long),
        "fine_map": torch.randn(2, N_FIXED_CH, FINE_CROP, FINE_CROP),
        "wide_map": torch.randn(2, N_FIXED_CH, WIDE_CROP, WIDE_CROP),
        "crop_idx": torch.full((2, FINE_CROP * FINE_CROP), 100, dtype=torch.long),
        "pos_idx": torch.tensor([100, 100], dtype=torch.long),
        "wmask": torch.tensor([0.0, 1.0]),
    }
    obs["vec"][:, PREVIOUS_ACTION] = 0
    obs["vec"][0, PREVIOUS_ACTION.start + 7] = 1
    priv = torch.randn(2, 24)
    _, state1, _ = model.forward_step(obs, state, priv_vec=priv)
    logits, next_state, extras = model.forward_step(obs, state1, priv_vec=priv)
    assert len(logits) == 6
    # Loaded weapon (batch 0, wmask=0) must suppress NextWeapon=1
    assert float(logits[5][0, 1].detach()) < -1e3
    assert float(logits[5][1, 1].detach()) > -1e2
    assert float(logits[4][0, 1].detach()) == -1e4
    assert float(logits[4][1, 1].detach()) > -1e2
    action, logp, _ = _sample_factored(logits)
    assert action[0, 4:].tolist() == [0, 0]
    assert torch.isfinite(logp).all()
    # Gradients flow through policy, VIN, 2-step Neural Map write->read, and critic
    loss = sum(lg.square().mean() for lg in logits) + extras["value"].square().mean() + extras["dir_logits"].square().mean()
    loss.backward()
    assert model.vin_conv.weight.grad is not None and model.vin_conv.weight.grad.abs().sum() > 0
    assert model.map_write.weight.grad is not None and model.map_write.weight.grad.abs().sum() > 0

    # Export stripped checkpoint and load via DoomAgent
    ckpt_path = tmp_path / "apex_test.pt"
    torch.save({"arch": "apex", "model": model.strip_for_submission()}, ckpt_path)
    agent = DoomAgent(observation_space(), action_space(), "cpu", weights_path=ckpt_path)
    assert agent.arch == "apex"

    env = make_env(ROOT / "maps" / "two_rooms.wad", "MAP01", episode_ticks=120)
    try:
        o, _ = env.reset(seed=0)
        agent.reset()
        t0 = time.perf_counter()
        for _ in range(10):
            previous = agent._prev_action.copy()
            a = agent.act(o)
            assert not (a[7] and previous[7])
            assert a.shape == (9,) and a.dtype == np.int8
            assert a[8] == 0  # weapon mask active while pistol has ammo
            o, _, term, trunc, _ = env.step(a)
            if term or trunc:
                break
        ms = (time.perf_counter() - t0) * 1000.0 / 10.0
        assert ms < 40.0, f"ApexNet CPU inference too slow: {ms:.1f} ms/step"
    finally:
        env.close()


def test_apex_teacher_clears_two_rooms_and_provides_aux_labels():
    env = FastDoomPackEnv(ROOT / "maps" / "two_rooms.wad", episode_ticks=2100, compute_spatial_targets=False)
    teacher = ApexTeacher()
    try:
        obs, _ = env.reset(seed=0)
        teacher.reset()
        exited = False
        saw_valid_dir = False
        for _ in range(400):
            _, act = teacher.act(obs, env, assume_executed=True)
            if dir_rel_to_bin(teacher.last_dir_rel_deg) < 16:
                saw_valid_dir = True
            obs, _, term, trunc, info = env.step(act)
            if info.get("exit_reached"):
                exited = True
                break
            if term or trunc:
                break
        assert exited, "ApexTeacher should clear two_rooms.wad"
        assert saw_valid_dir
    finally:
        env.close()


def test_apex_teacher_actions_follow_executed_button_availability():
    vec = torch.zeros(4, VEC_DIM)
    vec[:, PREVIOUS_ACTION.start + 7:PREVIOUS_ACTION.stop] = torch.tensor([[0, 0], [1, 0], [0, 1], [0, 0]])
    obs = {"vec": vec, "wmask": torch.tensor([1.0, 1.0, 1.0, 0.0])}
    teacher = torch.tensor([[1, 2, 0, 1, 1, 1]]).repeat(4, 1)
    expected = [[1, 1], [0, 1], [1, 0], [1, 0]]
    assert (~blocked_edge_buttons(obs)).int().tolist() == expected
    legal = _legal_teacher_actions(teacher, obs)
    assert legal[:, 4:6].tolist() == expected
    torch.testing.assert_close(legal[:, :4], teacher[:, :4])
    assert teacher[:, 4:].eq(1).all()


def test_apex_teacher_retries_skipped_weapon_press():
    env = FastDoomPackEnv(ROOT / "maps" / "two_rooms.wad", compute_spatial_targets=False)
    teacher = ApexTeacher()
    try:
        obs, _ = env.reset(seed=0)
        obs["vitals"][3] = 0
        # The student skipped the teacher's first switch. The next label can still switch.
        label, _ = teacher.act(obs, env, assume_executed=False)
        assert label[5] == 1
        teacher.record_action(np.zeros(9, dtype=np.int8))
        label, buttons = teacher.act(obs, env, assume_executed=False)
        assert label[5] == 1
        teacher.record_action(buttons)
        label, _ = teacher.act(obs, env, assume_executed=False)
        assert label[5] == 0
    finally:
        env.close()


def test_apex_teacher_approaches_door_before_pressing_use():
    env = FastDoomPackEnv(ROOT / "maps" / "two_rooms.wad", compute_spatial_targets=False)
    teacher = ApexTeacher()
    try:
        obs, _ = env.reset(seed=0)
        saw_door_outside_use_range = False
        for _ in range(150):
            _, action = teacher.act(obs, env)
            hit_type = obs["rays"]["hit_type"][0]
            distance = float(obs["rays"]["distance"][0]) * 1024
            if hit_type == RAY_DOOR and USE_RANGE < distance < 92:
                saw_door_outside_use_range = True
                assert not action[7]
                assert teacher.use_cooldown == 0
            if action[7]:
                assert saw_door_outside_use_range
                assert hit_type == RAY_DOOR
                assert distance <= USE_RANGE
                break
            obs, _, term, trunc, _ = env.step(action)
            assert not (term or trunc)
        else:
            pytest.fail("Teacher did not approach and use the door.")
    finally:
        env.close()
