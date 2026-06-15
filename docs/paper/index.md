# Paper reproduction

Everything in the ICPR 2026 paper is generated from the CLI and the
notebooks under `swissrivernetwork/benchmark/visualize_results/`.

## Headline recipe

```bash
uv run srn prepare-data
for m in lstm transformer_embedding transformer_graphlet transformer_stgnn; do
  for g in swiss-1990 swiss-2010 zurich; do
    uv run srn tune -m "$m" -g "$g" -n 200 -wl 90
  done
done
uv run srn evaluate
uv run srn sweep
```

Expected wall-clock on one A100: ~8 GPU-days for the full 8-method × 3-graph matrix. See the
sections below for partial-reproduction recipes that target individual figures.

- [Window-length sweep → Fig. 4 + HLE](window-length-sweep.md)
- [Noise robustness → Fig. 5 / Fig. 6](noise-robustness.md)
- [Figures](figures.md)

## Reproducibility notes

The original numbers reported in the paper were affected by two bugs
that are fixed on `main` since `4daeff3`:

- [Eval-window leakage](../explainers/eval-window-leakage.md)
- [Graphlet NaN at W > trained_wl](../explainers/graphlet-nan-fix.md)

Re-runs for Graphlet and Transformer-Graphlet at W ≠ 90 are required;
everything at W = 90 is unaffected.
