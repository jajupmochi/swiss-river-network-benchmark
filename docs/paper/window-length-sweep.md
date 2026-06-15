# Window-length sweep

The window-length sweep is the driver behind paper Figure 4 and the HLE
dimension of Figure 2.

## Run

```bash
uv run srn sweep
```

## Two-phase ordering

The sweep runs **ISOLATED** methods first, then **GRAPHLET** methods.
The ordering is load-bearing — graphlet methods read the neighbour
predictions dumped by the isolated phase.

| Phase | Methods |
| --- | --- |
| ISOLATED | `lstm`, `transformer` × `{learnable, sinusoidal, rope}` |
| GRAPHLET | `graphlet`, `transformer_graphlet` |

## Outputs

- `swissrivernetwork/benchmark/dump/predictions/<path_extra_keys>-evalwl{W}/`
  — one `wt_hat_*.csv` per station and method.
- `swissrivernetwork/benchmark/visualize_results/outputs/win_lens/<graph>_<method>_win_lens_resu.csv`
  — aggregated RMSE / MAE / NSE per `window_len`.

## Rendering Figure 4

```bash
uv run jupyter lab swissrivernetwork/benchmark/visualize_results/window_lens_resu.ipynb
```

The notebook reads the CSVs above and emits the final PDF under
`visualize_results/figures/`.

!!! warning "Eval-window leakage"
    If you are reproducing numbers from a pre-`4daeff3` run, discard
    everything at W ≠ 90 for graphlet methods and re-run — see
    [the explainer](../explainers/eval-window-leakage.md).
