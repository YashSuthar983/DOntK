# Training

Put your training pipeline here: environment setup, reward shaping, curriculum, model code, logs.
Judges review this folder during manual review.

You may modify the engine, maps and game variables freely **for training**. Evaluation always uses
the vanilla engine through the official `doom_env` wrapper, so train on `doom_env` observations.

## Measuring per-level clears

From the repository root, run the locked suite with ten consecutive engine seeds per map. Each manifest entry supplies the starting seed.

```bash
PYTHONPATH=.:env:submission python training/eval_and_calibrate.py \
    --suite training/wads/eval_locked/eval_suite.json --seeds-per-map 10 \
    --out-json /tmp/doom-per-level.json
```

The report includes individual seeds and outcomes, exits and attempts per level, and the lowest level clear rate. With `--calibrate --objective clear`, the evaluator ranks decoding settings by their weakest levels first, breaking ties with pooled clears and weighted score. Use `--objective score` for score-first calibration. Calibration writes the selected temperatures to `--weights`. Measure a chosen configuration on independent maps before reporting generalization.
