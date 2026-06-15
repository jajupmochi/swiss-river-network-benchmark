# Swiss River Network Benchmark — CUDA-enabled reference image.
#
# The image is self-contained: it carries the full dependency tree from
# `uv.lock` and can run every entry point (`srn prepare-data`, `srn tune`,
# `srn evaluate`, `srn sweep`, `srn app streamlit`, `srn app gradio`) as
# long as the host exposes a CUDA GPU via `--gpus all`.

ARG CUDA_VERSION=12.1.0
ARG UBUNTU_VERSION=22.04
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu${UBUNTU_VERSION} AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_SYSTEM_PYTHON=1 \
    RAY_CHDIR_TO_TRIAL_DIR=0

# System packages: Python 3.12, git (for editable installs), build-essential
# for PyTorch Geometric source wheels when needed, and curl for the uv installer.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
        ca-certificates curl git build-essential \
        libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Symlink `python` to Python 3.12 so `srn` + scripts run without friction.
RUN ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3

# Install uv (official installer).
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    install /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /app

# Install the locked dependency tree first for good caching.
COPY pyproject.toml uv.lock README.md ./
COPY swissrivernetwork /app/swissrivernetwork
RUN uv sync --no-cache --all-extras --frozen

# Bring in the rest of the repo AFTER sync so code changes don't bust the
# dependency layer.
COPY . /app

EXPOSE 8501 7860 8888

# Default: drop into a shell with the venv activated. Override via compose.
ENV PATH="/app/.venv/bin:${PATH}"
CMD ["bash"]
