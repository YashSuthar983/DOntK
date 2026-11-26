"""Replay successful training trajectories and fine-tune Aegis3D action heads."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "submission"), str(ROOT / "env"), str(ROOT)]

from agent import DEFAULT_DECODE_TEMPS
from aegis3d import AEGIS3D
from doom_env import make_env
from training.train_aegis import _obs_batch_to_torch


def load_actions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as demo:
        prefix, recovery = demo["prefix"], demo["recovery"]
    for actions in (prefix, recovery):
        if actions.ndim != 2 or actions.shape[1] != 9 or not np.isin(actions, (0, 1)).all():
            raise ValueError(f"{path}: expected binary actions of shape (steps, 9)")
        if (actions[:, :6:2] + actions[:, 1:6:2] > 1).any():
            raise ValueError(f"{path}: conflicting movement buttons")
    if not len(recovery):
        raise ValueError(f"{path}: missing recovery actions")
    return prefix.astype(np.int8), recovery.astype(np.int8)


@torch.no_grad()
def collect(
    model: AEGIS3D, rows: list[dict], demos: Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
    features, targets, available, anchors = [], [], [], []
    verified = []
    for row in rows:
        if row.get("split") != "training":
            raise ValueError(f"{row['id']}: recovery manifests may only contain training maps")
        path = demos / f"{row['id']}.npz"
        if not path.is_file():
            continue
        prefix, recovery = load_actions(path)
        actions = np.concatenate((prefix, recovery))
        env = make_env(row["wad"], row["map"], episode_ticks=row["ticks"])
        try:
            obs, _ = env.reset(seed=row["seed"])
            state = model.init_state(1, "cpu")
            for i, action in enumerate(actions):
                logits, state, extras = model.forward_step(
                    _obs_batch_to_torch([obs], torch.device("cpu")), state
                )
                hidden = extras["gru_h"][0].clone()
                if i >= len(prefix):
                    label = np.concatenate((action[:6:2] + 2 * action[1:6:2], action[6:]))
                    targets.append(torch.from_numpy(label).long())
                    features.append(hidden)
                    available.append([bool(lg[0, c] > -1e5) for lg, c in zip(logits, label)])
                    if not len(prefix) and i % 8 == 0:
                        anchors.append(hidden)
                elif i % 3 == 0:
                    anchors.append(hidden)
                state = model.update_state_with_action(state, torch.from_numpy(action).unsqueeze(0))
                obs, _, terminated, truncated, info = env.step(action)
                if bool(terminated or truncated) != (i + 1 == len(actions)):
                    raise ValueError(f"{row['id']}: unexpected termination at decision {i + 1}")
            if not info["exit_reached"]:
                raise ValueError(f"{row['id']}: replay did not reach the exit")
            verified.append({
                "id": row["id"], "tier": row["tier"], "prefix_steps": len(prefix),
                "recovery_steps": len(recovery), "action_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
            })
        finally:
            env.close()
    if not features or not anchors:
        raise ValueError("Expected at least one verified trajectory and one anchor observation")
    return torch.stack(features), torch.stack(targets), torch.tensor(available), torch.stack(anchors), verified


def fit(model, data, heads, epochs, lr, anchor_weight, seed):
    features, targets, available, anchors, _ = data
    optimizer = torch.optim.AdamW(model.action_heads[:heads].parameters(), lr=lr)
    rng = torch.Generator().manual_seed(seed)
    class_weights = [torch.tensor(w) for w in ((1., 1., 1.), (1., 1.4, 1.4), (1., 1., 1.),
                                              (1., 2.), (1., 4.), (1., 2.))]
    with torch.no_grad():
        reference = [head(anchors).softmax(-1) for head in model.action_heads[:heads]]

    def imitation(indices):
        losses = []
        for i, head in enumerate(model.action_heads[:heads]):
            valid = indices[available[indices, i]]
            if len(valid):
                losses.append(F.cross_entropy(head(features[valid]), targets[valid, i], weight=class_weights[i]))
        return torch.stack(losses).mean()

    for _ in range(epochs):
        for indices in torch.randperm(len(features), generator=rng).split(256):
            subset = torch.randint(len(anchors), (len(indices),), generator=rng)
            anchor_loss = torch.stack([
                F.kl_div(head(anchors[subset]).log_softmax(-1), target[subset], reduction="batchmean")
                for head, target in zip(model.action_heads[:heads], reference)
            ]).mean()
            loss = imitation(indices) + anchor_weight * anchor_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--demos", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heads", choices=(3, 6), type=int, default=6)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--anchor-weight", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if min(args.lr, args.epochs) <= 0 or args.anchor_weight < 0:
        parser.error("Expected positive learning rate and epochs and a nonnegative anchor weight")
    report_path = args.output.with_suffix(".json")
    if args.output.suffix != ".pt" or args.weights.resolve() in (args.output.resolve(), report_path.resolve()):
        parser.error("Choose a separate .pt output checkpoint")
    if args.output.exists() or report_path.exists():
        parser.error("Choose an unused output path")

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=True)
    if checkpoint.get("arch") == "apex":
        parser.error("Recovery training expects an Aegis3D checkpoint")
    model = AEGIS3D().eval()
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    rows = json.loads(args.manifest.read_text())
    if len({row["id"] for row in rows}) != len(rows):
        parser.error("Recovery manifest IDs must be unique")
    data = collect(model, rows, args.demos)
    fit(model, data, args.heads, args.epochs, args.lr, args.anchor_weight, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=args.output.parent) as tmp:
        torch.save({"model": model.state_dict(),
                    "decode_temps": checkpoint.get("decode_temps", DEFAULT_DECODE_TEMPS)}, tmp)
        args.output.hardlink_to(tmp.name)
    report = {"trajectories": data[-1], "decisions": len(data[0]), "anchors": len(data[3]), "heads": args.heads}
    with report_path.open("x") as destination:
        json.dump(report, destination, indent=2)
        destination.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k != "trajectories"}), flush=True)


if __name__ == "__main__":
    main()
