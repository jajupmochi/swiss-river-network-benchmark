"""Top-level command-line interface for the Swiss River Network Benchmark.

This module is a thin *wrapper* around the existing drivers in
``swissrivernetwork.benchmark.*``. It does not re-implement any training or
evaluation logic — every subcommand forwards its argv (and its return code)
straight to the underlying module via ``runpy``, so the installed ``srn``
console script stays in lock-step with the canonical ``uv run python -m …``
invocations documented in ``README.md`` and ``.claude/skills/run-benchmark``.

Usage::

    srn prepare-data               # data preparation (no flags)
    srn tune -m lstm -g swiss-2010 -n 50    # Ray Tune search (flags forward as-is)
    srn evaluate                   # ray_evaluation (config in __main__)
    srn sweep                      # window-length sweep
    srn train-single               # train a single unified model
    srn train-isolated             # train per-station models
    srn app gradio                 # launch the Gradio demo
    srn app streamlit              # launch the Streamlit local UI
    srn version                    # print package version

Design notes
------------
* The CLI is intentionally argv-forwarding: it does not re-parse the existing
  argparse flags declared inside each driver. If you run ``srn tune --help``
  you get this wrapper's help; ``srn tune -- --help`` forwards ``--help`` to
  ``ray_tune`` and shows the driver's full flag list.
* Non-zero exit codes from a driver propagate through ``typer.Exit``.
* No existing file under ``swissrivernetwork.benchmark.*`` is imported at the
  top — imports happen inside each command so that ``srn --help`` works even
  when heavy dependencies (torch, ray) are missing at import time.
"""

from __future__ import annotations

import runpy
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import List, Optional

try:
    import typer
except ImportError as exc:  # pragma: no cover - typer is a required runtime dep
    raise SystemExit(
        "typer is required for the `srn` CLI. Install it via `uv sync` or `pip install 'swissrivernetwork[cli]'`."
    ) from exc


app = typer.Typer(
    name="srn",
    help=(
        "Swiss River Network Benchmark — unified CLI wrapper around the "
        "underlying `swissrivernetwork.benchmark.*` drivers."
    ),
    no_args_is_help=True,
    add_completion=False,
)
app_cmd = typer.Typer(name="app", help="Launch interactive demo / local UI.", no_args_is_help=True)
app.add_typer(app_cmd, name="app")


def _forward(module: str, argv: Optional[List[str]] = None) -> None:
    """Run a ``swissrivernetwork.benchmark.*`` module with forwarded argv.

    Uses :func:`runpy.run_module` so that the target module's
    ``if __name__ == "__main__":`` block executes exactly as it would under
    ``python -m``. We temporarily patch ``sys.argv`` so argparse inside the
    module sees the user's flags.
    """
    old_argv = sys.argv
    sys.argv = [module, *(argv or [])]
    try:
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        if code != 0:
            raise typer.Exit(code=code) from exc
    finally:
        sys.argv = old_argv


# ---------------------------------------------------------------------------
# Core benchmark subcommands
# ---------------------------------------------------------------------------
@app.command("prepare-data", help="Build the three dataset splits (swiss-1990, swiss-2010, zurich).")
def prepare_data() -> None:
    _forward("swissrivernetwork.benchmark.data_preparation")


@app.command(
    "tune",
    help="Hyperparameter search via Ray Tune. Flags after `--` are forwarded to ray_tune.py.",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def tune(ctx: typer.Context) -> None:
    _forward("swissrivernetwork.benchmark.ray_tune", ctx.args)


@app.command(
    "evaluate",
    help="Evaluate Ray-tuned checkpoints. Edit ray_evaluation.py `__main__` block to configure.",
)
def evaluate() -> None:
    _forward("swissrivernetwork.benchmark.ray_evaluation")


@app.command(
    "sweep",
    help="Window-length sweep producing the Fig. 4 / HLE CSVs.",
)
def sweep() -> None:
    _forward("swissrivernetwork.benchmark.run_win_len_sweep")


@app.command(
    "train-single",
    help="Train a single unified model without Ray Tune.",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def train_single(ctx: typer.Context) -> None:
    _forward("swissrivernetwork.benchmark.train_single_model", ctx.args)


@app.command(
    "train-isolated",
    help="Train per-station models without Ray Tune.",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def train_isolated(ctx: typer.Context) -> None:
    _forward("swissrivernetwork.benchmark.train_isolated_station", ctx.args)


# ---------------------------------------------------------------------------
# Demo apps
# ---------------------------------------------------------------------------
@app_cmd.command("gradio", help="Launch the interactive workbench (Gradio; same app as the Hugging Face Space).")
def app_gradio() -> None:
    _forward("swissrivernetwork.app.gradio_app")


@app_cmd.command(
    "workbench",
    help="Alias for `srn app gradio` — the full local workbench (explore / train / infer / analyse).",
)
def app_workbench() -> None:
    _forward("swissrivernetwork.app.workbench")


@app_cmd.command("streamlit", help="Launch the Streamlit local UI (Explore / Predict / Compare).")
def app_streamlit() -> None:
    # Streamlit expects `streamlit run <script>` rather than `python -m <module>`. We
    # emulate that entry via the streamlit.web.cli module.
    import importlib.resources as _res

    with _res.as_file(_res.files("swissrivernetwork.app").joinpath("streamlit_app.py")) as script:
        _forward("streamlit.web.cli", ["run", str(script)])


# ---------------------------------------------------------------------------
# Meta
# ---------------------------------------------------------------------------
@app.command("version", help="Print package version.")
def version_cmd() -> None:
    try:
        typer.echo(version("swissrivernetwork"))
    except PackageNotFoundError:
        typer.echo("swissrivernetwork (not installed)")


def main() -> None:
    """Console-script entry point declared in ``pyproject.toml``."""
    app()


if __name__ == "__main__":
    main()
