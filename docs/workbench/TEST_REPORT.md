# Workbench control test report

Executed: 2026-07-05 · suite `swissrivernetwork/app/test_workbench_controls.py`
(plan: [TEST_PLAN.md](TEST_PLAN.md)).

## Result

```
85 passed in ~21s
```

- **58 / 58** live Gradio control bindings exercised (every `.change` / `.click` on the
  workbench) — verified by the completeness guard, which fails if the count drifts.
- **26** distinct handler functions, each additionally swept across the **full value
  domain** of every input control (all datasets, all 8 methods, all metrics/scenarios/
  statistics, slider min/def/max, checkbox all/first/empty, and valid/none/malformed file
  uploads).
- Bring-your-own-model inference covered against **both** sources — the trained CPU
  sandbox model and an uploaded `.joblib` estimator — plus none/malformed uploads.
- No handler raised; every output matched its bound component type (Figure / HTML /
  Markdown / DataFrame / dropdown-update); the streaming generator was consumed to its
  final frame.

## Coverage matrix (26 handlers · 58 bindings, all PASS)

| Handler | Bindings | Tab · what the control does | Status |
|---|---:|---|---|
| `station-update` (lambda) | 8 | Dataset → repopulate the Station dropdown (7 sub-tabs) | ✅ |
| `forecast_outlook` | 4 | Data · probabilistic outlook (dataset/station/threshold/horizon) | ✅ |
| `plot_noise` | 4 | Models · noise-robustness curve (dataset/models/noise-type/metric) | ✅ |
| `build_map` | 3 | Data · station map (dataset/colour-by/metric) | ✅ |
| `plot_series` | 3 | Data · raw air/water series | ✅ |
| `plot_seasonal` | 3 | Data · day-of-year climatology | ✅ |
| `threejs_html` | 3 | Data · 3D river network | ✅ |
| `plot_win_lens` | 3 | Models · window-length sweep | ✅ |
| `plot_forecast` | 3 | Models · forecasting-horizon curve | ✅ |
| `plot_per_station` | 3 | Models · per-station error bars | ✅ |
| `results_table` | 3 | Models · results table (scenario/statistic) | ✅ |
| `plot_coverage` | 2 | Data · data-coverage heatmap | ✅ |
| `plot_radar` | 2 | Models · multi-aspect radar | ✅ |
| `plot_sinusoidal` | 2 | Models · positional-encoding heatmap | ✅ |
| `overview_md` | 1 | Data · dataset overview text | ✅ |
| `radar_upload` | 1 | Models · overlay uploaded metrics on the radar | ✅ |
| `train_eval` | 1 | Train/Eval · train a CPU baseline | ✅ |
| `eval_predictions` | 1 | Train/Eval · score an uploaded prediction file | ✅ |
| `handle_upload` | 1 | Upload · auto-route an uploaded file | ✅ |
| `model_status` | 1 | Inference · report an uploaded model | ✅ |
| `run_inference` | 1 | Inference · predict (sandbox or uploaded model) | ✅ |
| `stream_inference` | 1 | Inference · live streaming prediction (generator) | ✅ |
| `residual_analysis` | 1 | Analysis · residual bias + distribution | ✅ |
| `seasonal_error` | 1 | Analysis · mean abs error by month | ✅ |
| `threshold_exceedance` | 1 | Analysis · days/year above a threshold | ✅ |
| `model_ranking` | 1 | Analysis · rank architectures | ✅ |

Tab 7 (*About & resources*) is static Markdown with no bindings; it is verified by the
live-UI screenshot pass, not by a callback test.

## Bug found and fixed during testing

**Impulse-noise robustness plot crashed for every selection.**
`plot_noise` hard-coded the x-axis column as `noise_level`, but only the `gaussian_a`
result CSVs have that column — the `impulse_a` CSVs use `probability` (with a
`scale_factor` column). So selecting **Noise type = `impulse_a`** — a valid dropdown
choice on the *Noise robustness* tab — raised `KeyError: ['noise_level'] not in index`
for all datasets and models.

Fix (`swissrivernetwork/app/workbench.py`, `plot_noise`): pick the x-axis column by noise
type (`noise_level` for gaussian, `probability` for impulse) and add a defensive
`xcol in columns` guard. Mirrored into the Hugging Face `app.py`. The sweep test now
passes for both noise types. (This is the second real workbench bug surfaced by exhaustive
control testing, after the radar nowcasting-axes bug fixed in the 2026-07-04 audit.)

## Reproduce

```bash
uv run --extra app --with pytest python -m pytest \
    swissrivernetwork/app/test_workbench_controls.py -q
```

The suite is data-driven from `app.demo.fns`, so adding a control automatically adds it to
the coverage and trips the completeness guard until it is accounted for. It runs in CI via
the `app-tests` job on every push.
