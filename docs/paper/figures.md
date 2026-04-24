# Figures

Every paper figure has a dedicated notebook or CSV chain.

| Paper | Source |
| --- | --- |
| Fig. 1 — network map | `visualize_results/datasets_overview.ipynb` |
| Fig. 2 — HLE + robustness radar | `visualize_results/results_in_polar.ipynb` + the win-len and noise CSVs |
| Fig. 3 — time-series sample | `visualize_results/visualize_results.ipynb` |
| Fig. 4 — window-length sweep | `visualize_results/window_lens_resu.ipynb` |
| Fig. 5 — Gaussian noise robustness | `visualize_results/noises.ipynb` |
| Fig. 6 — impulse noise robustness | `visualize_results/noises.ipynb` |
| Sankey — method / graph choices | `visualize_results/sankey.ipynb` |
| Yearly trends | `visualize_results/yearly_trends.ipynb` |

## Rasterise for the README / docs

After you've re-generated PDFs, export PNGs at a publication-friendly
DPI:

```bash
uv run python scripts/export_assets.py --only figures --dpi 200
```

Outputs land in `assets/export/figures/`.
