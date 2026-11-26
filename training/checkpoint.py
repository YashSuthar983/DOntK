"""Restore AdamW moments when fine-tuning a checkpoint with new model parameters."""


def restore_optimizer(optimizer, model, checkpoint):
    saved = checkpoint["opt"]
    groups = saved["param_groups"]
    # Both trainers construct one group in model parameter order. Preserve
    # existing moments by name when new trainable parameters were inserted.
    names = [name for name, _ in model.named_parameters()]
    existing = [name for name in names if name in checkpoint["model"]]
    if len(groups) == 1 and len(existing) < len(names) and len(existing) == len(groups[0]["params"]):
        old_ids = dict(zip(existing, groups[0]["params"]))
        next_id = max(old_ids.values(), default=-1) + 1
        for name in names:
            if name not in old_ids:
                old_ids[name] = next_id
                next_id += 1
        params = [old_ids[name] for name in names]
        saved = {**saved, "param_groups": [{**groups[0], "params": params}]}
    optimizer.load_state_dict(saved)
