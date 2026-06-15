<div align="center">

<img src="assets/social/banner.svg" alt="Swiss River Network Benchmark" width="100%"/>

# Swiss River Network Benchmark

<strong>Benchmark für Transformer zur räumlich-zeitlichen Modellierung von Flusswassertemperaturen</strong><br/>
<em>ICPR-2026-Einreichung · quelloffener Referenzcode, Datensätze und Abbildungen</em>

[![CI](https://github.com/jajupmochi/swiss-river-network-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/jajupmochi/swiss-river-network-benchmark/actions/workflows/ci.yml)
[![Docs](https://github.com/jajupmochi/swiss-river-network-benchmark/actions/workflows/docs.yml/badge.svg)](https://jajupmochi.github.io/swiss-river-network-benchmark/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![Code style: ruff](https://img.shields.io/badge/style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Hugging Face Space](https://img.shields.io/badge/🤗%20Space-live%20demo-blue)](https://huggingface.co/spaces/jajupmochi/swiss-river-network-benchmark)
[![Paper](https://img.shields.io/badge/ICPR-2026-8A2BE2)](#zitation)

**Sprache:** [English](README.md) · [简体中文](README.zh.md) · **Deutsch** · [Français](README.fr.md)

</div>

---

## Kurzfassung

Der Swiss River Network Benchmark ist ein **reproduzierbarer** offener Benchmark für die
raum-zeitliche Vorhersage der Wassertemperatur in Flüssen. Enthalten sind:

- **Drei Graph-Datensätze** — `swiss-1990`, `swiss-2010`, `zurich` — abgeleitet aus
  schweizerischen hydrologischen Messstationen.
- **Acht Referenzmethoden** — LSTM, Graphlet, LSTM + Stations-Embedding, ST-GNN,
  Transformer (lernbare / sinusoidale / RoPE-Positionskodierungen) sowie graphenbewusste
  Transformer-Varianten (Transformer-Graphlet, Transformer-Embedding, Transformer-ST-GNN).
- **Die vollständige Paper-Pipeline** — Hyperparametersuche mit Ray Tune,
  Testzeit-Evaluation, Fenstergrößen-Sweep (Abb. 4 + HLE), Robustheitsstudie unter
  Gauß-/Impulsrauschen (Abb. 5–6) und Visualisierungs-Notebooks, die jede Paper-Abbildung
  aus CSVs reproduzieren.
- **Fünf Installationswege** — `uv`, `pip`, Docker, Desktop-Installer (Klick & Start)
  sowie ein einmalig einzufügender Prompt für Claude Code / Codex / Gemini / Copilot.
- **Eine interaktive Demo** auf Hugging Face Spaces und eine reichere **lokale
  Streamlit-UI**, die beide den echten Visualisierungscode des Projekts einbetten.

> ⚠️ **Eine GPU mit CUDA ist für Training und Evaluation Pflicht.** Die Demos und
> reine Dokumentations-Workflows laufen auf CPU.

## Inhaltsverzeichnis

1. [Galerie](#galerie)
2. [Schnellstart (≤ 30 s)](#schnellstart--30-s)
3. [Installation — fünf Wege](#installation--fünf-wege)
   - [A. Entwickler-Installation mit `uv`](#a-entwickler-installation-mit-uv-empfohlen)
   - [B. `pip`-Installation](#b-pip-installation)
   - [C. Docker](#c-docker)
   - [D. Desktop-Installer (Windows / macOS / Linux)](#d-desktop-installer-windows--macos--linux)
   - [E. LLM-Agent „copy & run"](#e-llm-agent-copy--run)
4. [Das Paper reproduzieren](#das-paper-reproduzieren)
5. [Live-Demo & lokale UI](#live-demo--lokale-ui)
6. [Projektstruktur](#projektstruktur)
7. [CLI-Referenz](#cli-referenz)
8. [Dokumentation](#dokumentation)
9. [Mitwirken](#mitwirken)
10. [Zitation](#zitation)
11. [Danksagungen](#danksagungen)
12. [Lizenz](#lizenz)

## Galerie

<table>
  <tr>
    <td align="center" width="33%"><strong>Abb. 2 — HLE / Robustheits-Radar</strong><br/>
      <img src="assets/diagrams/architecture.svg" width="100%"/><br/>
      <sub>(Platzhalter — siehe <code>visualize_results/figures/all_resu_radar_grid_plot.pdf</code>.)</sub>
    </td>
    <td align="center" width="33%"><strong>Abb. 4 — Fenstergrößen-Sweep</strong><br/>
      <img src="assets/logo/logo.svg" width="100%"/><br/>
      <sub>(Platzhalter — erzeugt von <code>window_lens_resu.ipynb</code>.)</sub>
    </td>
    <td align="center" width="33%"><strong>Sankey — Methoden / Graph-Wahl</strong><br/>
      <img src="assets/social/social-card.svg" width="100%"/><br/>
      <sub>(Platzhalter — erzeugt von <code>sankey.ipynb</code>.)</sub>
    </td>
  </tr>
</table>

> Nach einer vollständigen Reproduktion
> `uv run python scripts/export_assets.py --only figures --dpi 200` ausführen, um die
> echten Paper-Abbildungen als PNGs in `assets/export/figures/` zu rendern.

## Schnellstart (≤ 30 s)

```bash
git clone https://github.com/jajupmochi/swiss-river-network-benchmark.git
cd swiss-river-network-benchmark
uv sync --no-cache

# Installation prüfen.
uv run srn --help
uv run srn version
```

Interaktive Demo lokal starten:

```bash
uv run srn app streamlit          # volle lokale UI
# oder
uv run srn app gradio             # Gradio (gleicher Code wie der HF Space)
```

## Installation — fünf Wege

> Der Benchmark ist so ausgelegt, dass Sie den Installationspfad passend zu Ihrer Rolle
> wählen können.

### A. Entwickler-Installation mit `uv` (empfohlen)

```bash
git clone https://github.com/jajupmochi/swiss-river-network-benchmark.git
cd swiss-river-network-benchmark
uv sync --no-cache                     # reproduzierbare Umgebung aus uv.lock
uv run srn --help                      # Konsolen-Einstiegspunkt
```

Optionale Extras:

```bash
uv sync --all-extras                   # alles
uv pip install -e '.[app]'             # nur Demo-Apps
uv pip install -e '.[docs]'            # mkdocs + i18n + mike
uv pip install -e '.[dev]'             # ruff, pytest, nbmake, pre-commit
```

### B. `pip`-Installation

```bash
python -m pip install 'swissrivernetwork[app]'          # sobald auf PyPI veröffentlicht
# oder aus dem Clone:
pip install -e '.[app]'
```

Mindestens Python 3.12. Für Training ist eine GPU mit CUDA 12.1+ erforderlich; die Demos
laufen CPU-only.

### C. Docker

> Für Trainings-Workloads werden eine NVIDIA-GPU und `nvidia-container-toolkit` benötigt.

```bash
docker compose up app                  # Streamlit-UI unter http://localhost:8501
docker compose run --rm train srn sweep
```

### D. Desktop-Installer (Windows / macOS / Linux)

Wenn Sie Hydrolog:in oder Praktiker:in sind und nicht in die Kommandozeile wollen, laden
Sie ein fertiges Installationspaket von der
[Releases-Seite](https://github.com/jajupmochi/swiss-river-network-benchmark/releases)
und starten es per Doppelklick:

| Plattform | Artefakt |
| --- | --- |
| Windows 10/11 x64 | `SwissRiverNetworkBenchmark-<ver>-win64.exe` |
| macOS (Apple Silicon) | `SwissRiverNetworkBenchmark-<ver>.dmg` |
| Linux x64 | `SwissRiverNetworkBenchmark-<ver>-x86_64.AppImage` |

Das Desktop-Bundle startet die Streamlit-UI lokal, lädt einen mitgelieferten Checkpoint
und visualisiert Vorhersagen an beliebigen Stationen — ohne Python- oder CUDA-Installation.
Trainings-Workloads erfordern weiterhin eine GPU — für Forschungsarbeit bevorzugen Sie
**A / B / C**.

Lokaler Build (fortgeschritten):

```bash
uv sync --all-extras
uv run pyinstaller packaging/swissrivernetwork.spec
```

### E. LLM-Agent „copy & run"

Öffnen Sie Ihren bevorzugten Coding-Agenten (Claude Code, Codex, Gemini CLI oder GitHub
Copilot CLI) und fügen Sie den Prompt unten ein. Der Agent klont, installiert, bereitet
die Daten vor, führt die Smoke-Tests aus und startet die UI — alles in einem Durchgang.

> 📎 Das vollständige, kopierbare Playbook liegt unter
> [`.claude/skills/install/SKILL.md`](.claude/skills/install/SKILL.md).

```text
Install the Swiss River Network Benchmark by cloning
https://github.com/jajupmochi/swiss-river-network-benchmark.git into the current directory,
running `uv sync --no-cache --all-extras`, smoke-checking with `uv run pytest -q`, and
then starting the Streamlit UI via `uv run srn app streamlit`. Read
.claude/skills/install/SKILL.md for the complete playbook before starting.
```

## Das Paper reproduzieren

```bash
# 0. Die drei Datensatz-Splits vorbereiten.
uv run srn prepare-data

# 1. Hyperparametersuche (pro Methode, pro Graph). Beispiel: LSTM auf swiss-2010.
uv run srn tune -m lstm -g swiss-2010 -n 200 -wl 90

# 2. Getunte Checkpoints evaluieren und wl=90-Tabellen schreiben.
uv run srn evaluate

# 3. Fenstergrößen-Sweep → Paper-Abb. 4 + HLE-Dimension von Abb. 2.
uv run srn sweep

# 4. Abbildungen aus CSVs rendern.
uv run jupyter lab swissrivernetwork/benchmark/visualize_results/
```

Der Fenster-Sweep läuft in zwei strikten Phasen:

1. **ISOLATED** — `lstm`, `transformer` (× PE in `{learnable, sinusoidal, rope}`). Schreibt
   `wt_hat`-Vorhersagen unter `dump/predictions/<path_extra_keys>-evalwl{W}/`.
2. **GRAPHLET** — `graphlet`, `transformer_graphlet`. Liest Phase-1-Dumps als
   Nachbar-Features.

> 🐛 **Reproduzierbarkeitshinweis.** Ältere Sweep-Läufe waren von zwei Bugs betroffen, die
> inzwischen behoben sind:
>
> 1. *Eval-Fenster-Leakage* im Isolated-Dump-Pfad (alle Sweep-Zeilen bei W ≠ 90 für
>    Graphlet lasen still die Vorhersage mit dem größten W).
> 2. *Outer-Join-NaN* in `merge_graphlet_dfs` bei W > trained_wl.
>
> Beide Fixes sind seit `4daeff3` auf `main`. Graphlet- und Transformer-Graphlet-Sweep-
> Zeilen bei W ≠ 90 müssen neu erzeugt werden; alles bei W = 90 ist nicht betroffen.
> Details siehe [`CHANGELOG.md`](CHANGELOG.md).

## Live-Demo & lokale UI

| Ziel | Befehl | Hinweis |
| --- | --- | --- |
| Hugging Face Space | [🤗 Demo öffnen](https://huggingface.co/spaces/jajupmochi/swiss-river-network-benchmark) | Gradio-Frontend, Live-Vorhersagen auf mitgelieferten Checkpoints. |
| Lokales Gradio | `uv run srn app gradio` | Gleiche App wie der HF Space. |
| Lokales Streamlit | `uv run srn app streamlit` | Tabs „Explore / Predict / Compare", nutzt den bestehenden Visualisierungscode des Projekts. |
| Desktop-Installer | `.exe` / `.dmg` / `.AppImage` doppelklicken | Dieselbe Streamlit-UI, gepackt. |

Beide Demo-Apps enthalten **Echtzeit-Visualisierung**, direkt aus den Notebooks unter
`swissrivernetwork/benchmark/visualize_results/` — keine Fake-Daten.

## Projektstruktur

```
swiss-river-network-benchmark/
├── swissrivernetwork/
│   ├── cli.py                              # `srn`-Einstieg (typer, leitet an Driver weiter)
│   ├── benchmark/
│   │   ├── data_preparation.py             # Datensatz-Splits erstellen
│   │   ├── ray_tune.py                     # Hyperparametersuche
│   │   ├── ray_evaluation.py               # Testzeit-Evaluation
│   │   ├── run_win_len_sweep.py            # Fenster-Sweep (Abb. 4 / HLE)
│   │   ├── train_single_model.py
│   │   ├── train_isolated_station.py
│   │   ├── util.py                         # merge_graphlet_dfs, get_evaluation_path_keys…
│   │   ├── dataset.py                      # Reader + SequenceDataset(Windowed)
│   │   └── visualize_results/              # Notebooks, die jede Paper-Abbildung erzeugen
│   ├── app/
│   │   ├── gradio_app.py                   # HF Space + lokales Gradio
│   │   └── streamlit_app.py                # lokale UI mit Live-Visualisierung
│   └── …                                   # Experiment-Helfer, NN-Module, Utilities
├── assets/                                  # Logo, Social Card, Architekturdiagramm
├── docs/                                    # MkDocs-Material-Site (en / zh / de / fr)
├── packaging/                               # PyInstaller-Spec + Plattform-Entry-Skripte
├── scripts/                                 # export_assets.py, Smoke-Helper
├── tests/
├── .claude/skills/                          # Claude-Code-Skills (install, run-benchmark)
├── pyproject.toml                           # PEP-621-Metadaten, Extras, Konsolen-Scripts
├── CITATION.cff
├── CHANGELOG.md
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── SECURITY.md
└── LICENSE
```

## CLI-Referenz

Alle Subkommandos von `srn` leiten direkt an die kanonischen Driver weiter — sie
existieren, damit Sie nach `pip install` einen installierbaren `srn`-Befehl haben:

| Befehl | Zugrundeliegender Driver |
| --- | --- |
| `srn prepare-data` | `python -m swissrivernetwork.benchmark.data_preparation` |
| `srn tune -m <method> -g <graph> …` | `python -m swissrivernetwork.benchmark.ray_tune …` |
| `srn evaluate` | `python -m swissrivernetwork.benchmark.ray_evaluation` |
| `srn sweep` | `python -m swissrivernetwork.benchmark.run_win_len_sweep` |
| `srn train-single` | `python -m swissrivernetwork.benchmark.train_single_model` |
| `srn train-isolated` | `python -m swissrivernetwork.benchmark.train_isolated_station` |
| `srn app gradio` | Gradio-Demo starten. |
| `srn app streamlit` | Streamlit-UI lokal starten. |
| `srn version` | Installierte Paket-Version ausgeben. |

Driver-spezifische Flags nach `--` anhängen:

```bash
uv run srn tune -m transformer_embedding -g swiss-2010 -n 200 -wl 90 -pe rope
```

Die vollständige Flag-Liste finden Sie in
[`.claude/skills/run-benchmark/SKILL.md`](.claude/skills/run-benchmark/SKILL.md)
oder durch `--help` bei jedem Driver.

## Dokumentation

Die vollständige Dokumentation wird mit MkDocs Material gebaut und in vier Sprachen
ausgeliefert.

| URL | Sprache |
| --- | --- |
| <https://jajupmochi.github.io/swiss-river-network-benchmark/> | English (Standard) |
| <https://jajupmochi.github.io/swiss-river-network-benchmark/zh/> | 简体中文 |
| <https://jajupmochi.github.io/swiss-river-network-benchmark/de/> | Deutsch |
| <https://jajupmochi.github.io/swiss-river-network-benchmark/fr/> | Français |

Kapitel: **Einstieg**, **Anwenderleitfaden (für Hydrolog:innen)**, **Tutorials**,
**Paper-Reproduktion**, **API-Referenz**, **Hintergründe**, **Entwicklerleitfaden**,
**Zitation**, **FAQ**.

Lokal bauen:

```bash
uv pip install -e '.[docs]'
uv run mkdocs serve
```

## Mitwirken

- Vor Issues / PRs bitte [`CONTRIBUTING.md`](CONTRIBUTING.md) lesen.
- Bug-Reports, Feature-Wünsche und Paper-Reproduktionsfragen haben je ein eigenes
  Issue-Template.
- Teilnahme unterliegt dem [Verhaltenskodex](CODE_OF_CONDUCT.md).
- Sicherheitsprobleme werden vertraulich gemeldet — siehe [`SECURITY.md`](SECURITY.md).

Offene Aufgaben und kommende Meilensteine werden im
[GitHub-Issue-Tracker](https://github.com/jajupmochi/swiss-river-network-benchmark/issues)
verfolgt.

## Zitation

Wenn Sie diesen Benchmark in wissenschaftlicher Arbeit verwenden, zitieren Sie bitte
sowohl Software als auch Paper.

**Software (Zenodo-DOI folgt nach dem ersten Release):**

```bibtex
@software{jia_swissrivernetwork_2026,
  author    = {Linlin Jia and Benjamin Fankhauser},
  title     = {Swiss River Network Benchmark: Spatio-Temporal River Water Temperature Modeling},
  year      = {2026},
  version   = {0.1.0},
  url       = {https://github.com/jajupmochi/swiss-river-network-benchmark},
  license   = {MIT}
}
```

**Paper (ICPR-2026-Einreichung — Platzhalter, nach Annahme aktualisieren):**

```bibtex
@inproceedings{jia_transformers_rivertemp_2026,
  author    = {Linlin Jia and Benjamin Fankhauser},
  title     = {Benchmarking Transformers on Spatio-Temporal River Water Temperature Modeling},
  booktitle = {International Conference on Pattern Recognition (ICPR)},
  year      = {2026},
  note      = {Under review.}
}
```

Das Repo enthält zusätzlich eine maschinenlesbare [`CITATION.cff`](CITATION.cff) — der
„Cite this repository"-Knopf auf der Repo-Seite liefert beide Einträge automatisch.

## Danksagungen

Dieser Benchmark baut auf Vorarbeiten von **Benjamin Fankhauser** und der
Hydrologiegruppe der **Universität Bern** auf. Wir danken dem Bundesamt für Umwelt (BAFU),
dem Zürcher **Amt für Abfall, Wasser, Energie und Luft (AWEL)** sowie den
kooperierenden öffentlichen Observatorien für die Stationsmessungen, die diesen
Datensatz erst möglich machen.

Infrastruktur und Tooling: [PyTorch](https://pytorch.org/),
[PyTorch Geometric](https://pyg.org/), [Ray Tune](https://www.ray.io/ray-tune),
[Hugging Face](https://huggingface.co/), [Gradio](https://www.gradio.app/),
[Streamlit](https://streamlit.io/), [MkDocs Material](https://squidfunk.github.io/mkdocs-material/),
[uv](https://github.com/astral-sh/uv).

## Lizenz

MIT — siehe [`LICENSE`](LICENSE).
