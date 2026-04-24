# Noise robustness

Figures 5 and 6 study how each method degrades under input perturbations:
Gaussian noise on the temperature signal and impulse noise that simulates
sensor drop-outs.

## Run

```bash
uv run jupyter lab swissrivernetwork/benchmark/visualize_results/noises.ipynb
```

The notebook:

1. Loads checkpoints from `outputs/ray_results/`.
2. Applies Gaussian / impulse noise to the test splits at configurable
   strengths.
3. Computes RMSE / MAE / NSE degradation curves and writes them to
   `visualize_results/outputs/noises/`.
4. Renders the two paper figures.

## Custom noise levels

Modify the `NOISE_STDS` and `IMPULSE_PROBS` constants at the top of the
notebook. Re-run the affected cells — no retraining is required.

## Interpretation

- Graph-aware methods (Graphlet, ST-GNN) are typically more robust to
  per-station noise because neighbor predictions dampen outliers.
- The Transformer-Embedding variant has the flattest degradation curve
  under impulse noise, consistent with the paper's discussion.
