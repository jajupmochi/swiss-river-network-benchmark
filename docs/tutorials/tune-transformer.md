# Tune a Transformer

Runs a small Ray Tune search for the `transformer_embedding` method
with RoPE positional encoding.

## Command

```bash
uv run srn tune -- -m transformer_embedding -g swiss-2010 -n 50 -wl 90 -pe rope
```

Flags:

- `-n 50` — 50 Ray Tune trials. Bump to `200` for paper-grade numbers.
- `-wl 90` — 90-day window (the canonical choice; the sweep varies it
  at evaluation time).
- `-pe rope` — Rotary Positional Embedding. The paper reports all three
  PE variants.

## Expected runtime

On one A100, ~2 hours for 50 trials. On an RTX 4090 expect ~2.5 hours.
Scale linearly with `-n`.

## What it writes

- `swissrivernetwork/benchmark/outputs/ray_results/<run_id>/` — one
  subdirectory per trial with the best checkpoint and Ray metadata.
- Weights & Biases run, if `WANDB_API_KEY` is set.

## After tuning

Evaluate with:

```bash
uv run srn evaluate
uv run srn sweep
```

`evaluate` writes the wl=90 tables; `sweep` writes
`dump/predictions/<path_extra_keys>-evalwl{W}/` for every eval window
length.

!!! tip "Matching the paper"
    To reproduce the paper's Transformer numbers exactly, run all three
    PE variants (`rope`, `sinusoidal`, `learnable`) with `-n 200`, then
    run the sweep.
