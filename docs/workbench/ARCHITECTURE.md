# Architecture — Swiss River Network Benchmark

Every architecture and data-flow diagram for the project, in Mermaid. It opens with the
**design plan** (the catalogue of diagrams and why each exists), then implements them,
grouped into *System*, *Data flows*, *Runtime / app*, and *Deployment*.

## Design plan — the diagram catalogue

| # | Diagram | Answers | Group |
|---|---|---|---|
| 1 | System framework | What are the top-level components and how do they relate? | System |
| 2 | Package & CLI map | Which `srn` command runs which module? | System |
| 3 | Model taxonomy | How do 4 classes × 2 backbones give 8 reference models? | System |
| 4 | Reproduction tiers | What are Tier 0/1/2 and their cost? | System |
| 5 | Data preparation flow | Raw records → released graph/series/split artifacts | Data flow |
| 6 | Training flow | Dataset → model → NaN-masked loss → validation → best checkpoint | Data flow |
| 7 | Hyper-parameter search | How Ray Tune + ASHA select per (model, dataset) | Data flow |
| 8 | Evaluation & window sweep | Checkpoints → metrics → per-station / aggregate CSVs | Data flow |
| 9 | Result-artifact provenance | Which CSV backs which figure / table / workbench view | Data flow |
| 10 | Workbench runtime | CSVs + uploads → one Gradio app → six tab groups | Runtime |
| 11 | Bring-your-own-model inference | Upload → load → device → windowed predict → stream | Runtime |
| 12 | Upload auto-routing | How an arbitrary file is detected and plotted | Runtime |
| 13 | Deployment & CI/CD | Local vs Space, single source, and the pipelines | Deployment |

---

## System

### 1. System framework

```mermaid
flowchart TB
  subgraph pkg["swissrivernetwork package"]
    CLI["srn CLI (typer wrapper)"]
    BENCH["benchmark/ · pipeline + models"]
    APP["app/ · Gradio + Streamlit workbench"]
  end
  DATA[("released datasets<br/>graphs · series · result CSVs")]
  DOCS["MkDocs site (mkdocstrings)"]
  CI["CI/CD · ruff · pytest · CodeQL · Pages"]

  CLI --> BENCH
  CLI --> APP
  BENCH --> DATA
  DATA --> APP
  BENCH -. docstrings .-> DOCS
  APP -. docstrings .-> DOCS
  CI --- pkg
  CI --> DOCS
```

### 2. Package & CLI map

```mermaid
flowchart LR
  U(["user: srn ..."]) --> CLI["cli.py wrapper"]
  CLI --> P["prepare-data → data_preparation"]
  CLI --> T["tune → ray_tune"]
  CLI --> E["evaluate → ray_evaluation"]
  CLI --> S["sweep → run_win_len_sweep"]
  CLI --> TS["train-single → train_single_model"]
  CLI --> TI["train-isolated → train_isolated_station"]
  CLI --> A["app gradio/streamlit → app.workbench"]
  CLI --> V["version"]
```

### 3. Model taxonomy — 8 reference models

```mermaid
flowchart TB
  subgraph classes["4 model classes"]
    ISO["Isolated<br/>no embedding · no graph"]
    EMB["Embedded<br/>station embedding"]
    GRA["Graphlet<br/>neighbours' predictions appended"]
    STG["ST-GNN<br/>message passing (PyG)"]
  end
  subgraph back["× 2 backbones"]
    L["LSTM"]
    TR["Transformer<br/>learnable · sinusoidal · RoPE"]
  end
  ISO --> L & TR
  EMB --> L & TR
  GRA --> L & TR
  STG --> L & TR
  L --> N["= 8 reference models"]
  TR --> N
```

### 4. Reproduction tiers

```mermaid
flowchart LR
  subgraph T0["Tier 0 · CPU · seconds"]
    C[("committed result CSVs")] --> FIG["tables + figures + workbench"]
  end
  subgraph T1["Tier 1 · 1 GPU · minutes"]
    CK[("tuned checkpoints")] --> EVAL["srn evaluate → metrics"]
  end
  subgraph T2["Tier 2 · GPU ≥24GB · hours–days"]
    RAW[("raw data")] --> FULL["prepare → tune → evaluate → sweep"]
  end
  FULL --> CK
  EVAL --> C
```

---

## Data flows

### 5. Data preparation flow

```mermaid
flowchart LR
  RAW[("raw station records<br/>air + water temperature")] --> DP["data_preparation"]
  DP --> G[("directed river graph<br/>graph_*.pth")]
  DP --> SR[("aligned air/water series")]
  DP --> SP[("fixed train/val/test splits")]
  DP --> DROP["drop short / empty sequences"]
  G & SR & SP --> DS["dataset: windowed torch Datasets<br/>(NaN-masked missing ground truth)"]
```

### 6. Training flow

```mermaid
flowchart TB
  DS["windowed dataset<br/>(air-temp window → water-temp)"] --> M["model (one of 8)"]
  M --> PRED["normalized predictions"]
  PRED --> LOSS["loss with mask = not isnan(y)"]
  LOSS --> OPT["optimizer step"]
  OPT --> M
  M --> VAL["validation each epoch"]
  VAL --> MSE["validation_mse<br/>over all masked samples"]
  VAL --> ARMSE["validation_ave_rmse<br/>aggregated per day (nowcasting)"]
  MSE --> BEST{"best by validation_mse<br/>(ASHA scope)"}
  BEST --> CKPT[("best checkpoint")]
```

> The selection metric is `validation_mse` (never aggregated). `validation_ave_rmse` is a
> logged diagnostic that aggregates overlapping windows to one prediction/day for windowed
> nowcasting — see the 2026-07-04 audit report.

### 7. Hyper-parameter search (Ray Tune + ASHA)

```mermaid
flowchart LR
  CFG["search space per (model, dataset)"] --> RT["Ray Tune"]
  RT --> ASHA["ASHA scheduler<br/>early-stop weak trials"]
  ASHA --> TR1["trial 1"] & TR2["trial 2"] & TRN["trial N (num_samples)"]
  TR1 & TR2 & TRN --> SEL{"get_best_trial<br/>metric=validation_mse"}
  SEL --> CK[("best checkpoint + config")]
```

### 8. Evaluation & window-length sweep

```mermaid
flowchart LR
  CK[("best checkpoint")] --> EV["ray_evaluation<br/>reload + recompute metrics"]
  EV --> PS[("per-station result CSVs")]
  EV --> AG[("aggregate result CSVs")]
  SW["run_win_len_sweep<br/>vary eval window W"] --> DUMP[("per-W prediction dumps")]
  DUMP --> MG["merge_graphlet_dfs (inner join)"]
  MG --> WL[("window-length CSVs")]
```

> The sweep uses provenance-keyed dump paths (`-evalwl{W}`) and an inner join so a shorter
> window can never read a longer window's dump — the two reproduction bugs fixed and
> regression-tested in the library.

### 9. Result-artifact provenance

```mermaid
flowchart LR
  AG[("aggregate CSVs")] --> TAB["nowcast/forecast tables (143/144 cells)"]
  PS[("per-station CSVs")] --> BARS["per-station error · radar · ranking"]
  WL[("window-length CSVs")] --> WLFIG["window-length curves"]
  FS[("future-step CSVs")] --> HOR["forecasting-horizon curves"]
  NO[("noise CSVs")] --> NOISE["noise-robustness curves"]
  TAB & BARS & WLFIG & HOR & NOISE --> WB["workbench + paper figures"]
```

---

## Runtime / app

### 10. Workbench runtime architecture

```mermaid
flowchart TB
  CSV[("released CSVs")] --> WB
  UP(["upload: data / model"]) --> WB
  WB["Workbench (single Gradio source)<br/>CPU / GPU auto-detected"]
  WB --> DATA["Data · map · series · coverage · outlook · 3D"]
  WB --> MODELS["Models · radar · window · horizon · noise · table · pos-enc"]
  WB --> TRAIN["Train/Eval · CPU sandbox"]
  WB --> INF["Inference · sandbox or uploaded model + live stream"]
  WB --> AN["Analysis · residuals · seasonal · threshold · ranking"]
  WB --> ABOUT["About · detected compute"]
```

### 11. Bring-your-own-model inference

```mermaid
flowchart TB
  SRC{"model source"} -->|sandbox| SB[("last trained model")]
  SRC -->|upload| F(["file .joblib/.pkl/.pt"])
  F --> H{"hosted Space?"}
  H -->|yes| BLK["disabled (code-exec risk)"]
  H -->|no| LOAD{"file type"}
  LOAD -->|sklearn| CPU["load → CPU"]
  LOAD -->|TorchScript| DEV{"CUDA available?"}
  DEV -->|yes| GPU["load → GPU"]
  DEV -->|no| CPU2["load → CPU"]
  SB & CPU & GPU & CPU2 --> X["windowed air-temp features"]
  X --> PR["predict"]
  PR --> R["chart + RMSE/MAE/NSE"]
  PR --> ST["live-streamed chart"]
```

### 12. Upload auto-routing

```mermaid
flowchart TB
  U(["uploaded file"]) --> RA["read_any (csv/tsv/xlsx/json/parquet, ≤50MB)"]
  RA --> D{"detect content"}
  D -->|window_len / future_step / noise_level| CURVE["benchmark curve"]
  D -->|Station + RMSE/MAE/NSE| BAR["per-station bars"]
  D -->|time column + numeric| TS["time-series lines"]
  D -->|else| ERR["clear error message"]
```

---

## Deployment

### 13. Deployment & CI/CD topology

```mermaid
flowchart TB
  SRC["single workbench source"] --> LOCAL["local: srn app gradio<br/>(model upload enabled)"]
  SRC --> SPACE["HF Space: app.py<br/>(SPACE_ID gate: upload disabled)"]

  subgraph GH["GitHub Actions"]
    CI["ci.yml · ruff · pytest · app-tests · CodeQL"]
    DOCS["docs.yml · mkdocs + mike"]
    REL["release.yml"]
  end
  REPO["push / PR"] --> CI
  REPO --> DOCS
  DOCS --> PAGES["GitHub Pages (versioned docs)"]
  REL --> PYPI["PyPI / release artifacts"]
  SRC -. huggingface_hub upload .-> SPACE
```

---

## Notes

- All diagrams are validated to parse with Mermaid 11.
- The pipeline stages (1, 5–8) map one-to-one onto the four `srn` commands
  (`prepare-data → tune → evaluate → sweep`); everything downstream (Tier 0) is CPU-only.
- The workbench (10–12) is intentionally a *single* source file so the local app and the
  Hugging Face Space cannot drift; the only runtime difference is the `SPACE_ID` upload
  gate (11) and CPU/GPU auto-detection.
