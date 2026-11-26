"""Regression checks for learned landmark writes and older checkpoints."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "submission"))
from aegis3d import AEGIS3D, SlotLandmarkMemory


class LandmarkMemoryTest(unittest.TestCase):
    def test_different_scenes_can_write_different_landmarks(self):
        memory = SlotLandmarkMemory(scene_dim=2)
        with torch.no_grad():
            memory.write_routing.weight.zero_()
            memory.write_routing.weight[0, 0] = 12
            memory.write_routing.weight[1, 1] = 12
            memory.write_gate[0].weight.zero_()
            memory.write_gate[0].weight[:2] = torch.eye(2)
            memory.write_gate[0].bias.zero_()
            memory.write_gate[-1].weight.zero_()
            memory.write_gate[-1].bias.zero_()
            memory.write_gate[-1].bias[0] = 12
            memory.write_gate[-1].weight[1:3, :2] = torch.eye(2)

        state = AEGIS3D().init_state(1, "cpu")
        pose_delta = torch.tensor([[0., 0., 0., 1.]])
        keys = torch.zeros(1, 6)
        first = memory(state.slot_xy, state.slot_conf, state.slot_feat, pose_delta, torch.tensor([[1., 0.]]), keys)
        second = memory(*first[:3], pose_delta, torch.tensor([[0., 1.]]), keys)
        xy = second[0].detach()
        self.assertGreater(float(xy[0, 0, 0]), 100)
        self.assertGreater(float(xy[0, 1, 1]), 100)
        self.assertLess(float(xy[0, 0, 1]), 1)
        self.assertLess(float(xy[0, 1, 0]), 1)
        second[-1].square().mean().backward()
        self.assertGreater(float(memory.write_routing.weight.grad.abs().max()), 0)

    def test_legacy_weights_keep_inference_and_initialize_training(self):
        training = AEGIS3D(include_training_heads=True)
        legacy = training.strip_for_submission()
        routing_key = "slot_mem.write_routing.weight"
        del legacy[routing_key]

        inference = AEGIS3D()
        inference.load_state_dict(legacy, strict=True)
        self.assertEqual(torch.count_nonzero(inference.slot_mem.write_routing.weight).item(), 0)

        training.load_state_dict(legacy, strict=False)
        routing = training.slot_mem.write_routing.weight
        self.assertFalse(torch.equal(routing[0], routing[1]))

        inference.load_state_dict(training.strip_for_submission(), strict=True)
        torch.testing.assert_close(inference.slot_mem.write_routing.weight, routing)
