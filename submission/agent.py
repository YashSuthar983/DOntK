"""Evaluation entry point for DoomAgent. The scorer instantiates DoomAgent from this file.

Contract: docs/CONTRACT.md §6.
All decisions flow strictly through trained neural network parameters (AEGIS3D or ApexNet).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from aegis3d import AEGIS3D, factored_to_multibinary
from apex_net import ApexNet, apex_factored_to_multibinary
from egomap import EgoMapBuilder

SUBMISSION_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = SUBMISSION_DIR / "weights" / "aegis3d.pt"

# Default calibrated per-head sampling temperatures:
# [MoveFB, StrafeLR, TurnLR, Attack, Use, NextWeapon]
DEFAULT_DECODE_TEMPS = (0.20, 0.30, 0.20, 0.15, 0.55, 0.15)
APEX_DEFAULT_TEMPS = (0.50, 0.50, 0.50, 0.35, 0.50, 0.25)


class DoomAgent:
    def __init__(self, observation_space, action_space, device: str, weights_path: str | Path | None = None):
        """Load weights once onto `device` ("cuda" or "cpu") and warm up kernels."""
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")

        wpath = Path(weights_path) if weights_path is not None else WEIGHTS_PATH
        ckpt = None
        if wpath.is_file():
            ckpt = torch.load(wpath, map_location=self.device, weights_only=True)

        self.arch = "apex" if (isinstance(ckpt, dict) and ckpt.get("arch") == "apex") else "aegis3d"
        if self.arch == "apex":
            self.egomap: EgoMapBuilder | None = EgoMapBuilder()
            self.model = ApexNet(include_training_heads=False).to(self.device)
            self.decode_temps = APEX_DEFAULT_TEMPS
            self._prev_action = np.zeros(9, dtype=np.int8)
        else:
            self.egomap = None
            self.model = AEGIS3D(include_training_heads=False).to(self.device)
            self.decode_temps = DEFAULT_DECODE_TEMPS

        if ckpt is not None:
            if isinstance(ckpt, dict) and "model" in ckpt:
                state_dict = ckpt["model"]
                if "decode_temps" in ckpt:
                    self.decode_temps = tuple(float(x) for x in ckpt["decode_temps"])
            else:
                state_dict = ckpt
            self.model.load_state_dict(state_dict, strict=True)

        self.model.eval()
        self.state = self.model.init_state(1, self.device)
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(2026)

        # Warm up kernels inside init budget so first act() is instant
        dummy = {
            "rgb": np.zeros((120, 160, 3), dtype=np.uint8),
            "depth": np.zeros((120, 160), dtype=np.uint8),
            "vitals": np.zeros((17,), dtype=np.float32),
            "rays": {
                "distance": np.ones((64,), dtype=np.float32),
                "hit_type": np.zeros((64,), dtype=np.int64),
            },
            "visible_entities": {
                "features": np.zeros((32, 5), dtype=np.float32),
                "entity_type": np.zeros((32,), dtype=np.int64),
                "mask": np.zeros((32,), dtype=np.int8),
            },
        }
        self.act(dummy)
        self.reset()

    def reset(self) -> None:
        """Clear all recurrent and episodic map states before each map."""
        self.state = self.model.init_state(1, self.device)
        if self.egomap is not None:
            self.egomap.reset()
            self._prev_action = np.zeros(9, dtype=np.int8)
        self._rng.manual_seed(2026)

    def _sample_heads(self, logits: list[torch.Tensor]) -> torch.Tensor:
        picks = []
        for lg, tau in zip(logits, self.decode_temps):
            if tau <= 1e-4:
                picks.append(lg.argmax(dim=-1))
            else:
                probs = torch.softmax(lg.float() / tau, dim=-1).cpu()
                picks.append(torch.multinomial(probs, num_samples=1, generator=self._rng).squeeze(-1).to(self.device))
        return torch.stack(picks, dim=-1)

    @torch.inference_mode()
    def act(self, observation: dict) -> np.ndarray:
        """Return a valid MultiBinary(9) int8 action from trained neural network parameters."""
        dev = self.device
        if self.arch == "apex":
            assert self.egomap is not None
            feats, _ = self.egomap.step(observation, self._prev_action)
            obs_t = {
                k: torch.as_tensor(v, device=dev).unsqueeze(0)
                for k, v in feats.items()
            }
            logits, next_state, _ = self.model.forward_step(obs_t, self.state)
            factored = self._sample_heads(logits)
            action_mb = apex_factored_to_multibinary(factored)[0].cpu().numpy()
            self.state = next_state
            self._prev_action = action_mb
            return action_mb

        obs_t = {
            "rgb": torch.from_numpy(np.ascontiguousarray(observation["rgb"])).unsqueeze(0).to(dev, non_blocking=True),
            "depth": torch.from_numpy(np.ascontiguousarray(observation["depth"])).unsqueeze(0).to(dev, non_blocking=True),
            "vitals": torch.from_numpy(np.ascontiguousarray(observation["vitals"])).unsqueeze(0).to(dev, non_blocking=True),
            "ray_distance": torch.from_numpy(np.ascontiguousarray(observation["rays"]["distance"])).unsqueeze(0).to(dev, non_blocking=True),
            "ray_hit_type": torch.from_numpy(np.ascontiguousarray(observation["rays"]["hit_type"])).unsqueeze(0).to(dev, non_blocking=True),
            "ent_features": torch.from_numpy(np.ascontiguousarray(observation["visible_entities"]["features"])).unsqueeze(0).to(dev, non_blocking=True),
            "ent_type": torch.from_numpy(np.ascontiguousarray(observation["visible_entities"]["entity_type"])).unsqueeze(0).to(dev, non_blocking=True),
            "ent_mask": torch.from_numpy(np.ascontiguousarray(observation["visible_entities"]["mask"])).unsqueeze(0).to(dev, non_blocking=True),
        }

        logits, next_state, _ = self.model.forward_step(obs_t, self.state)
        factored = self._sample_heads(logits)
        action_mb = factored_to_multibinary(factored)
        self.state = self.model.update_state_with_action(next_state, action_mb)
        return action_mb[0].cpu().numpy()
