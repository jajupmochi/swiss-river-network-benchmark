"""Rasterise the hand-authored SVG assets and paper figure PDFs to PNGs.

This script is a convenience for producing pixel versions of assets for
channels that do not render SVG directly (PyPI long description, Hugging
Face card, some Markdown pipelines) and for embedding paper figures in
`README.md`. It reads from ``assets/`` and
``swissrivernetwork/benchmark/visualize_results/figures/`` (if present) and
writes to ``assets/export/`` which is git-ignored.

It has **no side effects on the training pipeline** and can be run on CPU
without GPU / Ray / Torch installed.

Usage::

    uv run python scripts/export_assets.py                     # defaults: PNG at 2x
    uv run python scripts/export_assets.py --dpi 3
    uv run python scripts/export_assets.py --only logo         # subset

The script prefers the `cairosvg` / `pdf2image` Python packages and falls
back to the `rsvg-convert` / `pdftoppm` CLIs if those are on PATH.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO / "assets"
FIGURES = REPO / "swissrivernetwork" / "benchmark" / "visualize_results" / "figures"
EXPORT = ASSETS / "export"


def _have(pkg: str) -> bool:
    try:
        __import__(pkg)
    except ImportError:
        return False
    return True


def _run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def svg_to_png(svg: Path, out: Path, scale: int) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if _have("cairosvg"):
        import cairosvg

        cairosvg.svg2png(url=str(svg), write_to=str(out), scale=float(scale))
        return
    rsvg = shutil.which("rsvg-convert")
    if rsvg:
        _run([rsvg, "-z", str(scale), "-o", str(out), str(svg)])
        return
    inkscape = shutil.which("inkscape")
    if inkscape:
        _run([inkscape, "--export-type=png", f"--export-dpi={96 * scale}", f"--export-filename={out}", str(svg)])
        return
    raise SystemExit(
        "Could not rasterise SVG — install `cairosvg` (uv pip install cairosvg) or "
        "one of `rsvg-convert` / `inkscape` on PATH."
    )


def pdf_to_png(pdf: Path, out_dir: Path, dpi: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = pdf.stem
    if _have("pdf2image"):
        from pdf2image import convert_from_path

        pages = convert_from_path(str(pdf), dpi=dpi)
        for i, page in enumerate(pages, start=1):
            suffix = "" if len(pages) == 1 else f"-p{i}"
            page.save(out_dir / f"{stem}{suffix}.png")
        return
    pdftoppm = shutil.which("pdftoppm")
    if pdftoppm:
        _run([pdftoppm, "-png", "-r", str(dpi), str(pdf), str(out_dir / stem)])
        return
    print(f"  (skipped {pdf.name}: install `pdf2image` or `pdftoppm`)", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scale", type=int, default=2, help="Raster scale multiplier for SVGs (default: 2).")
    parser.add_argument("--dpi", type=int, default=200, help="Raster DPI for PDFs (default: 200).")
    parser.add_argument(
        "--only",
        choices=["logo", "social", "diagrams", "figures", "all"],
        default="all",
        help="Subset of assets to export.",
    )
    args = parser.parse_args()

    EXPORT.mkdir(exist_ok=True)

    def want(name: str) -> bool:
        return args.only in (name, "all")

    svg_targets = []
    if want("logo"):
        svg_targets += sorted((ASSETS / "logo").glob("*.svg"))
    if want("social"):
        svg_targets += sorted((ASSETS / "social").glob("*.svg"))
    if want("diagrams"):
        svg_targets += sorted((ASSETS / "diagrams").glob("*.svg"))

    for svg in svg_targets:
        out = EXPORT / svg.parent.name / (svg.stem + ".png")
        svg_to_png(svg, out, args.scale)
        print(f"  {svg.relative_to(REPO)}  →  {out.relative_to(REPO)}")

    if want("figures") and FIGURES.exists():
        for pdf in sorted(FIGURES.glob("*.pdf")):
            pdf_to_png(pdf, EXPORT / "figures", args.dpi)
            print(f"  {pdf.relative_to(REPO)}  →  {(EXPORT / 'figures').relative_to(REPO)}")


if __name__ == "__main__":
    main()
