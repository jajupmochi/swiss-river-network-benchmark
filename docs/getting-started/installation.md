# Installation

## Prerequisites

| Requirement | Why |
| --- | --- |
| Python 3.12+ | PEP-695 generics used internally |
| `uv` ≥ 0.5 | The only supported dependency manager |
| NVIDIA GPU + CUDA 12.x | Required for training / evaluation |
| Git | to clone the repository |

Verify your GPU once with `nvidia-smi`. If it fails, training will crash
at the first `.cuda()` call.

## A. `uv` (recommended)

```bash
git clone https://github.com/jajupmochi/swiss-river-network-benchmark.git
cd swiss-river-network-benchmark
uv sync --no-cache
uv run srn --help
```

Optional extras:

```bash
uv sync --all-extras              # everything
uv pip install -e '.[app]'        # demo apps only
uv pip install -e '.[docs]'       # mkdocs + i18n + mike
uv pip install -e '.[dev]'        # ruff + pytest + nbmake
```

## B. `pip`

```bash
python -m pip install 'swissrivernetwork[app]'        # once on PyPI
pip install -e '.[app]'                               # from a clone
```

## C. Docker

```bash
docker compose --profile ui up -d app     # Streamlit on :8501
docker compose --profile train run --rm train srn sweep
```

See [`docker-compose.yml`](https://github.com/jajupmochi/swiss-river-network-benchmark/blob/main/docker-compose.yml)
for all profiles (`train`, `ui`, `notebook`).

## Sanity check

```bash
uv run srn version
uv run python -c "import torch; assert torch.cuda.is_available()"
```

Both lines should exit cleanly.
