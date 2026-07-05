# Workbench control test plan

**Scope.** Every interactive control on the Swiss River Network Benchmark workbench
(the Gradio app that runs identically locally and on the Hugging Face Space), tested
end-to-end, one control at a time, with no omission.

**How "no omission" is guaranteed.** The test set is *derived from the live Gradio
event graph* — `app.demo.fns` — rather than hand-written. Every control's
`.change` / `.click` binding is exactly one entry in that dict (58 bindings today), so
iterating it exercises every control. A **completeness guard** test asserts that the set
of exercised handler ids equals `set(app.demo.fns)` and that the count is 58 — if a
control is added later without a test, the guard fails.

**Reusability.** The suite is fully parametrized/data-driven
(`swissrivernetwork/app/test_workbench_controls.py`): it reads each control's own value
domain from the component (`choices`, `minimum/maximum/value`) and file fixtures. Adding
a control needs no new test code — it is picked up automatically. This is a permanent
regression harness, not a one-off script.

## Method (per control)

For each of the 58 handlers the callback is invoked:

1. **Baseline** — every input at its default value → assert the output artifact is valid.
2. **Input-domain sweep** (once per distinct handler) — vary each input control across
   its full domain while holding the others at default:

   | Control type | Values swept |
   |---|---|
   | Dropdown | **every** choice (all datasets, all 8 methods, all metrics, all scenarios, …) |
   | CheckboxGroup | all models · first-only · empty |
   | Slider | minimum · default · maximum |
   | File | a valid fixture · none · a malformed file |

3. **Output validation** — the returned object must match the bound output component
   type: `Plot`→`plotly.Figure`, `HTML`→non-empty html string, `Markdown`→string,
   `Dataframe`→`pandas.DataFrame`, `Dropdown`→a `gr.update` dict. Two-output handlers
   must return a matching 2-tuple; the streaming generator is fully consumed and its
   final frame validated. No handler may raise.

Model-inference controls are tested against **both** model sources — the CPU sandbox
model (trained in a fixture, exercising the real train→infer path) and an **uploaded**
`.joblib` estimator — plus none/malformed uploads, so the bring-your-own-model path is
covered end-to-end.

## Complete control inventory (7 tabs · 58 bindings · 26 distinct handlers)

Handler ids `[n]` refer to `app.demo.fns`.

### Tab 1 — Data

| Sub-tab | Control (label) | Type | Handler | Ids |
|---|---|---|---|---|
| Map & overview | Dataset | Dropdown | `build_map` + `overview_md` | [0],[3] |
| Map & overview | Colour by | Dropdown | `build_map` | [1] |
| Map & overview | Colour by metric | Dropdown | `build_map` | [2] |
| Raw series & seasonality | Dataset | Dropdown | station-update + `plot_series` + `plot_seasonal` | [4],[5],[6] |
| Raw series & seasonality | Station | Dropdown | `plot_series` + `plot_seasonal` | [7],[8] |
| Raw series & seasonality | Split | Dropdown | `plot_series` + `plot_seasonal` | [9],[10] |
| Data coverage | Dataset | Dropdown | `plot_coverage` | [11] |
| Data coverage | Split | Dropdown | `plot_coverage` | [12] |
| Forecast outlook | Dataset | Dropdown | station-update + `forecast_outlook` | [13],[14] |
| Forecast outlook | Station | Dropdown | `forecast_outlook` | [15] |
| Forecast outlook | Threshold | Dropdown | `forecast_outlook` | [16] |
| Forecast outlook | horizon (days) | Slider | `forecast_outlook` | [17] |
| 3D river network | Dataset | Dropdown | `threejs_html` | [18] |
| 3D river network | Height / colour by | Dropdown | `threejs_html` | [19] |
| 3D river network | Metric | Dropdown | `threejs_html` | [20] |

### Tab 2 — Models & results

| Sub-tab | Control | Type | Handler | Ids |
|---|---|---|---|---|
| Multi-aspect radar | Dataset | Dropdown | `plot_radar` | [21] |
| Multi-aspect radar | Models | CheckboxGroup | `plot_radar` | [22] |
| Multi-aspect radar | overlay file | File | `radar_upload` | [23] |
| Window-length sweep | Dataset | Dropdown | `plot_win_lens` | [24] |
| Window-length sweep | Models | CheckboxGroup | `plot_win_lens` | [26] |
| Window-length sweep | Metric | Dropdown | `plot_win_lens` | [25] |
| Forecasting horizons | Dataset / Models / Metric | Dropdown ×2 + CheckboxGroup | `plot_forecast` | [27],[28],[29] |
| Noise robustness | Dataset / Noise type / Metric / Models | Dropdown ×3 + CheckboxGroup | `plot_noise` | [30],[31],[32],[33] |
| Per-station error | Dataset / Model / Metric | Dropdown ×3 | `plot_per_station` | [34],[35],[36] |
| Results table | Dataset / Scenario / Statistic | Dropdown ×3 | `results_table` | [37],[38],[39] |
| Positional encoding | d_model / sequence length | Slider ×2 | `plot_sinusoidal` | [40],[41] |

### Tab 3 — Train / Eval (CPU sandbox)

| Sub-tab | Control | Type | Handler | Ids |
|---|---|---|---|---|
| Train a baseline | Dataset | Dropdown | station-update | [42] |
| Train a baseline | Dataset / Station / Model / window + **Train** button | Dropdown ×3 + Slider + Button | `train_eval` | [43] |
| Evaluate predictions | Ground-truth dataset | Dropdown | station-update | [44] |
| Evaluate predictions | Predictions file / dataset / Station | File + Dropdown ×2 | `eval_predictions` | [45] |

### Tab 4 — Upload your data

| Control | Type | Handler | Ids |
|---|---|---|---|
| Upload file | File | `handle_upload` | [46] |
| Reference dataset | Dropdown | `handle_upload` | [46] |

### Tab 5 — Inference & prediction

| Control | Type | Handler | Ids |
|---|---|---|---|
| Upload model | File | `model_status` | [47] |
| Dataset | Dropdown | station-update | [48] |
| Dataset / Station / Split / Model source / model file / input window / Device + **Predict** | Dropdown ×4 + File + Slider | `run_inference` | [49] |
| … same inputs + **Predict live (streaming)** | Button | `stream_inference` (generator) | [50] |

### Tab 6 — Analysis

| Sub-tab | Control | Type | Handler | Ids |
|---|---|---|---|---|
| Residuals | Dataset | Dropdown | station-update | [51] |
| Residuals | 7 inputs + **Analyse** | Dropdown ×5 + File + Slider | `residual_analysis` | [52] |
| Seasonal error | Dataset | Dropdown | station-update | [53] |
| Seasonal error | 7 inputs + **Seasonal error** | Dropdown ×5 + File + Slider | `seasonal_error` | [54] |
| Threshold exceedance | Dataset | Dropdown | station-update | [55] |
| Threshold exceedance | Dataset / Station / Threshold + **Exceedance** | Dropdown ×3 | `threshold_exceedance` | [56] |
| Model ranking | Dataset / Metric + **Rank** | Dropdown ×2 | `model_ranking` | [57] |

### Tab 7 — About & resources

Static `Markdown` only (detected-resources summary + resource links). No interactive
bindings, so no handler; covered by the live-UI screenshot pass, not by a callback test.

## Deliverables

1. `swissrivernetwork/app/test_workbench_controls.py` — the reusable suite above.
2. `docs/workbench/TEST_REPORT.md` — executed results (pass/fail per handler + counts).
3. Live-UI verification + screenshots via Playwright (feeds the user manual).
