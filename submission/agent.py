"""Evaluation entry point for AEGIS-3D v2. The scorer instantiates DoomAgent from this file.

Contract: docs/CONTRACT.md §6.
All decisions flow strictly through trained neural network parameters in AEGIS3D.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from aegis3d import AEGIS3D, factored_to_multibinary

SUBMISSION_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = SUBMISSION_DIR / "weights" / "aegis3d.pt"

# Default calibrated per-head sampling temperatures:
# [MoveFB, StrafeLR, TurnLR, Attack, Use, NextWeapon]
# Empirically validated "sharp" profile: crisp aim/move/turn, deadlock-free USE! (0.0 = greedy argmax)
DEFAULT_DECODE_TEMPS = (0.20, 0.30, 0.20, 0.15, 0.55, 0.15)


class DoomAgent:
    def __init__(self, observation_space, action_space, device: str):
        """Load AEGIS-3D v2 weights once onto `device` ("cuda" or "cpu") and warm up kernels."""
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")

        self.model = AEGIS3D(include_training_heads=False).to(self.device)
        self.decode_temps = DEFAULT_DECODE_TEMPS
        if WEIGHTS_PATH.is_file():
            ckpt = torch.load(WEIGHTS_PATH, map_location=self.device, weights_only=True)
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

        # Warm up CUDA/CPU kernels inside the generous 60s init budget so first act() is instant
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
        """Clear all recurrent, BEV, landmark slot, and E3B episodic states before each map."""
        self.state = self.model.init_state(1, self.device)
        self._rng.manual_seed(2026)

    @torch.inference_mode()
    def act(self, observation: dict) -> np.ndarray:
        """Return a valid MultiBinary(9) int8 action from trained AEGIS-3D v2 neural parameters."""
        dev = self.device
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
        picks = []
        for lg, tau in zip(logits, self.decode_temps):
            if tau <= 1e-4:
                picks.append(lg.argmax(dim=-1))
            else:
                probs = torch.softmax(lg.float() / tau, dim=-1).cpu()
                picks.append(torch.multinomial(probs, num_samples=1, generator=self._rng).squeeze(-1).to(dev))
        factored = torch.stack(picks, dim=-1)  # (1, 6)
        action_mb = factored_to_multibinary(factored)  # (1, 9) int8
        self.state = self.model.update_state_with_action(next_state, action_mb)
        return action_mb[0].cpu().numpy()
