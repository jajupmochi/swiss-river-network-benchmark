# Swiss River Network Benchmark

!!! info "ICPR 2026 submission"
    Open-source reference code, datasets, and figures for
    **Benchmarking Transformers on Spatio-Temporal River Water
    Temperature Modeling**.

The Swiss River Network Benchmark is a reproducible benchmark for river
water-temperature forecasting. It ships three real-world graph datasets,
eight reference methods, and the exact training / evaluation / sweep
pipeline used in the paper.

<div class="grid cards" markdown>

-   :material-rocket-launch-outline: __Install in 30 s__

    ---

    ```bash
    git clone https://github.com/jajupmochi/swiss-river-network-benchmark.git
    cd swiss-river-network-benchmark
    uv sync --no-cache
    uv run srn --help
    ```

    [:octicons-arrow-right-24: Getting started](getting-started/index.md)

-   :material-flask-outline: __Reproduce the paper__

    ---

    Run the full training / evaluation / sweep pipeline with one CLI.
    Every figure in the paper has a matching notebook.

    [:octicons-arrow-right-24: Paper reproduction](paper/index.md)

-   :material-application-braces: __Live demo__

    ---

    Interactive UIs on Hugging Face Space, local Streamlit, and a
    double-click desktop installer — all sharing one visualisation layer.

    [:octicons-arrow-right-24: Desktop app](getting-started/desktop.md)

-   :material-book-open-variant: __API reference__

    ---

    Generated from the source code via `mkdocstrings`. Useful when you
    want to plug a new method into the benchmark harness.

    [:octicons-arrow-right-24: API reference](api/index.md)

</div>

## What's in the box

- **3 datasets** — `swiss-1990`, `swiss-2010`, `zurich`.
- **8 methods** — LSTM, Graphlet, LSTM + station embedding, ST-GNN,
  Transformer, Transformer + Graphlet, Transformer + Embedding,
  Transformer + ST-GNN.
- **5 installation paths** — `uv`, `pip`, Docker, desktop installer,
  LLM-agent paste-and-run.
- **Window-length sweep** — the exact configuration behind paper Fig. 4
  and the HLE dimension of Fig. 2.

## Why open this benchmark

- **Reproducibility first.** Bugs that produced the original paper
  numbers are fixed on `main` since `4daeff3` — see
  [explainers](explainers/index.md) for the details.
- **Model-agnostic.** New methods plug in through one `config` dict and
  one dataset adapter; there is no framework lock-in.
- **Operator-friendly.** A hydrologist who never touches Python can
  still run the Streamlit demo from a release installer.

[:material-github: View on GitHub](https://github.com/jajupmochi/swiss-river-network-benchmark){ .md-button .md-button--primary }
[:material-emoticon: Hugging Face Space](https://huggingface.co/spaces/jajupmochi/swiss-river-network-benchmark){ .md-button }
