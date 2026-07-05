# Interactive workbench — complete user manual

The **workbench** is the point-and-click companion to the Swiss River Network Benchmark.
The *same* application runs as a zero-install **Hugging Face Space** and **locally**
(`srn app gradio`), from a single source file. This manual documents **every** control,
with a step-by-step flow, a Mermaid flow diagram, and annotated screenshots. Every feature
here is covered by the automated control test suite (see
[TEST_REPORT.md](TEST_REPORT.md)).

> Convention in the diagrams: **rounded blue** = a control you operate · **grey** = data
> read from disk · **green** = the artifact you get back.

## Launching

```bash
# local, one command (auto-detects CPU/GPU, opens a browser)
uv run srn app gradio
# or the double-click launcher for non-programmers:
scripts/launch-workbench.sh   # .command on macOS, .bat on Windows
```

Or open the hosted Space (no install): the link on the repository home page.

## The seven tabs at a glance

```mermaid
flowchart TD
  START([Open the workbench]) --> DATA[Data]
  START --> MODELS["Models & results"]
  START --> TRAIN[Train / Eval sandbox]
  START --> UP[Upload your data]
  START --> INF["Inference & prediction"]
  START --> AN[Analysis]
  START --> ABOUT["About & resources"]

  DATA --> D1[Map &amp; overview]
  DATA --> D2["Raw series & seasonality"]
  DATA --> D3[Data coverage]
  DATA --> D4[Forecast outlook]
  DATA --> D5[3D river network]
  MODELS --> M1[Radar · window · horizon · noise · per-station · table · pos-encoding]
  TRAIN --> T1[Train a baseline] --> T2[Evaluate predictions]
  INF --> I1[Predict / live-stream, sandbox or uploaded model]
  AN --> A1[Residuals · seasonal error · threshold · ranking]
```

The audience split: **Explore** (Data) and **Analyse** are aimed at hydrologists;
**Models & results** reproduces the paper's comparisons; **Train/Eval**, **Upload** and
**Inference** let you bring your own data and model.

---

# Tab 1 — Data

![Data · Map & overview](img/01-landing-data-map.png)

## 1.1 Map & overview

Places every monitoring station on a Swiss basemap with the directed river-network edges,
and colours each station by a chosen value.

**Steps:** ① pick a **Dataset** → ② pick what to **Colour by** (max/mean water temperature,
or a model's error) → ③ pick the **metric** used when colouring by a model.

```mermaid
flowchart LR
  A([1 Dataset]) --> D[stations_*.csv + edges_*.csv]
  B([2 Colour by]) --> F{water-temp or model?}
  C([3 Metric]) --> F
  D --> F
  F -->|water-temp| G[per-station max/mean °C]
  F -->|model| H[per-station error from result CSV]
  G --> MAP([Folium map · blue=low, red=high]):::out
  H --> MAP
  classDef out fill:#d7f5dd,stroke:#2b8a3e;
```

The overview text below the map states the station/edge counts, the train/test split, and
the CH1903→WGS84 coordinate conversion.

## 1.2 Raw series & seasonality

Shows a station's measured **air and water temperature** over time and its **day-of-year
climatology** (mean ± standard deviation).

**Steps:** ① Dataset → ② Station (the station list refreshes to match the dataset) → ③
Split (train / test).

```mermaid
flowchart LR
  A([1 Dataset]) --> U[refresh Station list]
  U --> B([2 Station])
  B --> P[series_/graph_split.csv]
  C([3 Split]) --> P
  P --> L([Line chart: water + air °C]):::out
  P --> S([Seasonal cycle: mean ± std by day-of-year]):::out
  classDef out fill:#d7f5dd,stroke:#2b8a3e;
```

## 1.3 Data coverage

A heatmap of which station has a water-temperature value on which day — a fast way to see
gaps before modelling.

```mermaid
flowchart LR
  A([1 Dataset]) --> P[series CSV]
  B([2 Split]) --> P
  P --> H([Coverage heatmap · stations × time, filled = has value]):::out
  classDef out fill:#d7f5dd,stroke:#2b8a3e;
```

## 1.4 Forecast outlook

A drought-portal-style **probabilistic outlook**: a day-of-year climatology band (p10–p90)
with the median and an ecological threshold line for the coming weeks. Illustrative
(climatology-based), not an operational forecast.

**Steps:** ① Dataset → ② Station → ③ Threshold (e.g. 25 °C fish-stress) → ④ horizon slider
(7–56 days).

```mermaid
flowchart LR
  A([1 Dataset]) --> C[history: train + test series]
  B([2 Station]) --> C
  C --> K[day-of-year climatology p10/p50/p90]
  D([3 Threshold]) --> V(["Outlook chart: observed + p10–p90 band + median + threshold"]):::out
  E([4 Horizon]) --> V
  K --> V
  classDef out fill:#d7f5dd,stroke:#2b8a3e;
```

## 1.5 3D river network

An interactive **three.js** view: stations placed by longitude/latitude, raised and
coloured by the selected value; river edges on the base plane. Drag to rotate, scroll to
zoom.

```mermaid
flowchart LR
  A([1 Dataset]) --> N[stations + edges]
  B([2 Height / colour by]) --> V
  C([3 Metric]) --> V
  N --> V([three.js scene in an iframe]):::out
  classDef out fill:#d7f5dd,stroke:#2b8a3e;
```

---

# Tab 2 — Models & results

![Models & results · Multi-aspect radar](img/02-models-radar.png)

Every panel here regenerates a comparison from the committed result CSVs (Tier 0 — no GPU).

## 2.1 Multi-aspect radar

Normalised multi-aspect comparison (outer = better) of the selected models across
nowcasting/forecasting RMSE·MAE·NSE, noise robustness and window-efficiency. Optionally
**overlay your own** per-station metrics file.

```mermaid
flowchart LR
  A([1 Dataset]) --> R[per-model axes from result CSVs]
  B([2 Models]) --> R
  U([Optional: upload metrics CSV]) --> R
  R --> N[normalise each axis to 0–1, higher = better]
  N --> V([Radar chart]):::out
  classDef out fill:#d7f5dd,stroke:#2b8a3e;
```

## 2.2 Window-length sweep · 2.3 Forecasting horizons · 2.4 Noise robustness

Three line-curve panels of the same shape: pick a **Dataset**, a set of **Models**, and a
**Metric**; each reads its dedicated result CSV and draws one curve per model.

```mermaid
flowchart LR
  A([Dataset]) --> P[win_lens / future_steps / noises CSV]
  B([Models]) --> P
  C([Metric]) --> P
  N([Noise type · noise panel only]) --> P
  P --> V([Curve: metric vs. window / horizon / noise level]):::out
  classDef out fill:#d7f5dd,stroke:#2b8a3e;
```

> Note: the noise panel picks its x-axis by noise type — `noise_level` for Gaussian,
> `probability` for impulse. (A bug where impulse always used `noise_level` and crashed was
> fixed during control testing — see [TEST_REPORT.md](TEST_REPORT.md).)

## 2.5 Per-station error · 2.6 Results table · 2.7 Positional encoding

- **Per-station error** — a bar chart of one model's error at each station (Dataset /
  Model / Metric).
- **Results table** — the RMSE/MAE/NSE table for a Dataset × Scenario (nowcasting /
  forecasting) × Statistic (mean/median/std/min/max).
- **Positional encoding** — a live heatmap of the sinusoidal positional encoding for a
  chosen `d_model` and sequence length (an explainer, computed on the fly).

```mermaid
flowchart LR
  subgraph Per-station
    PA([Dataset/Model/Metric]) --> PB([Per-station bars]):::out
  end
  subgraph Results table
    TA([Dataset/Scenario/Statistic]) --> TB([RMSE/MAE/NSE table]):::out
  end
  subgraph Positional encoding
    EA([d_model / seq-len sliders]) --> EB([sin/cos heatmap]):::out
  end
  classDef out fill:#d7f5dd,stroke:#2b8a3e;
```

---

# Tab 3 — Train / Eval (CPU sandbox)

![Train / Eval sandbox](img/05-train-eval.png)

## 3.1 Train a baseline

Fits a **lightweight CPU model** (ridge regression or a small MLP) on a window of past air
temperature for one station — in seconds, no GPU. The trained model is remembered and
becomes the *"Last trained sandbox model"* used by the Inference and Analysis tabs.

**Steps:** ① Dataset → ② Station → ③ Model (Ridge / MLP) → ④ window slider → ⑤ **Train &
evaluate**.

```mermaid
flowchart LR
  A([1 Dataset]) --> W[windowed train/test samples]
  B([2 Station]) --> W
  D([4 window]) --> W
  C([3 Model]) --> F[fit on train]
  W --> F
  E([5 Train button]) --> F
  F --> S[(remember model in session state)]
  F --> V([Prediction-vs-truth chart + RMSE/MAE/NSE]):::out
  classDef out fill:#d7f5dd,stroke:#2b8a3e;
```

## 3.2 Evaluate predictions

Upload a CSV with a numeric prediction column; it is aligned to a station's bundled test
ground truth and scored with RMSE/MAE/NSE.

```mermaid
flowchart LR
  U([Upload predictions CSV]) --> R[read + find prediction column]
  A([Dataset]) --> G[bundled test ground truth]
  B([Station]) --> G
  R --> M[align on overlapping non-NaN points]
  G --> M
  M --> V([Chart + RMSE/MAE/NSE]):::out
  classDef out fill:#d7f5dd,stroke:#2b8a3e;
```

---

# Tab 4 — Upload your data

![Upload your data](img/06-upload.png)

Upload any table (`.csv/.tsv/.xlsx/.xls/.json/.parquet`, ≤ 50 MB, parsed in memory, never
stored). The app **auto-detects** the content and routes it to the right view; malformed,
empty, oversized or non-numeric files return a clear message rather than crashing.

```mermaid
flowchart TD
  U([Upload file]) --> P[read_any: parse by extension]
  P --> R{detect content}
  R -->|has window_len/future_step/noise_level| C([Benchmark curve]):::out
  R -->|Station + RMSE/MAE/NSE| B([Per-station bar chart]):::out
  R -->|time column + numeric| T([Time-series line chart]):::out
  R -->|none of the above| E([Clear error message]):::err
  classDef out fill:#d7f5dd,stroke:#2b8a3e;
  classDef err fill:#ffe3e3,stroke:#e03131;
```

---

# Tab 5 — Inference & prediction

![Inference & prediction — numbered steps](img/03-inference-annotated.png)

Predict a station's water temperature with **your own model** or the sandbox model. A
scikit-learn estimator (`.joblib`/`.pkl`) runs on CPU; a **TorchScript** module (`.pt`)
runs on a **GPU when one is auto-detected** (the detected compute is shown at the top),
otherwise CPU. The **Predict live** button streams the prediction point-by-point.

> Security: model files execute code when loaded, so **model upload is enabled only
> locally and disabled on the hosted Space**.

**Steps (numbered on the screenshot):** ① Dataset → ② Station → ③ Split → ④ Model source
(sandbox or uploaded) → ⑤ upload a model if chosen → ⑥ input window (must match the model's
feature count) → ⑦ **Predict** or **Predict live**.

```mermaid
flowchart LR
  S([4 Model source]) --> G{which model?}
  G -->|sandbox| SB[(last trained model)]
  G -->|uploaded| UP([5 upload .joblib/.pt]) --> LD{hosted?}
  LD -->|Space| BLK([disabled for security]):::err
  LD -->|local| M[load model · GPU if TorchScript]
  SB --> M
  A([1 Dataset]) --> X[windowed air-temp features]
  B([2 Station]) --> X
  C([3 Split]) --> X
  W([6 input window]) --> X
  X --> M
  M --> R([7 Predict → chart + RMSE/MAE/NSE]):::out
  M --> L([7 Predict live → streamed chart]):::out
  classDef out fill:#d7f5dd,stroke:#2b8a3e;
  classDef err fill:#ffe3e3,stroke:#e03131;
```

---

# Tab 6 — Analysis

![Analysis](img/04-analysis.png)

A decision-support layer over the same predictions. Each sub-tab uses the sandbox or an
uploaded model (same controls as Inference).

## 6.1 Residuals

Prediction − observation over time and its distribution; reports bias, spread, and the
fraction of days with |residual| > 2 °C.

## 6.2 Seasonal error

Mean absolute error by calendar month — is the model worst in the summer fish-stress
season?

## 6.3 Threshold exceedance

Observed days per year a station exceeds a temperature limit (e.g. 25 °C) — the most
directly useful view for a water manager. (Uses observations only; no model needed.)

## 6.4 Model ranking

Ranks all eight benchmarked architectures by their mean per-station error on a dataset.

```mermaid
flowchart LR
  P[predictions from sandbox / uploaded model] --> R1([Residual bias + distribution]):::out
  P --> R2([Mean abs error by month]):::out
  OBS[bundled observations] --> R3([Days/year above threshold]):::out
  CSV[per-station result CSVs] --> R4([Architectures ranked by mean error]):::out
  classDef out fill:#d7f5dd,stroke:#2b8a3e;
```

---

# Tab 7 — About & resources

Static page: the detected compute summary (Python, CPU cores, RAM, GPU) and links to the
GitHub repository, documentation and the ICPR 2026 paper. No interactive controls.

---

## Where the data comes from

All views read the **released benchmark artifacts** bundled with the app
(`swissrivernetwork/app/data`): station coordinates and edge lists (CH1903→WGS84), the
per-station raw series, and the aggregate/per-station result CSVs. Uploaded files are
parsed in memory and never persisted. The compute footprint is shown live on the Inference
and About tabs.
