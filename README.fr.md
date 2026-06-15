<div align="center">

<img src="assets/social/banner.svg" alt="Swiss River Network Benchmark" width="100%"/>

# Swiss River Network Benchmark

<strong>Benchmark de Transformers pour la modélisation spatio-temporelle de la température des rivières</strong><br/>
<em>Soumission ICPR 2026 · code de référence, jeux de données et figures en open source</em>

[![CI](https://github.com/jajupmochi/swiss-river-network-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/jajupmochi/swiss-river-network-benchmark/actions/workflows/ci.yml)
[![Docs](https://github.com/jajupmochi/swiss-river-network-benchmark/actions/workflows/docs.yml/badge.svg)](https://jajupmochi.github.io/swiss-river-network-benchmark/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![Code style: ruff](https://img.shields.io/badge/style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Hugging Face Space](https://img.shields.io/badge/🤗%20Space-live%20demo-blue)](https://huggingface.co/spaces/jajupmochi/swiss-river-network-benchmark)
[![Paper](https://img.shields.io/badge/ICPR-2026-8A2BE2)](#citation)

**Langue :** [English](README.md) · [简体中文](README.zh.md) · [Deutsch](README.de.md) · **Français**

</div>

---

## En bref

Swiss River Network Benchmark est un benchmark ouvert et **reproductible** pour la
prévision spatio-temporelle de la température de l'eau en rivière. Il comprend :

- **Trois jeux de données graphes** — `swiss-1990`, `swiss-2010`, `zurich` — dérivés
  de stations hydrologiques suisses.
- **Huit méthodes de référence** — LSTM, Graphlet, LSTM + embedding de station, ST-GNN,
  Transformer (encodages positionnels appris / sinusoïdaux / RoPE), ainsi que les
  variantes Transformer sensibles au graphe (Transformer-Graphlet, Transformer-Embedding,
  Transformer-ST-GNN).
- **Le pipeline complet du papier** — recherche d'hyperparamètres via Ray Tune,
  évaluation au test, balayage de la longueur de fenêtre (Fig. 4 + HLE), étude de
  robustesse au bruit gaussien / impulsionnel (Fig. 5–6), et notebooks de visualisation
  qui reproduisent chaque figure du papier à partir de CSV.
- **Cinq chemins d'installation** — `uv`, `pip`, Docker, installeur desktop
  « double-clic », et un prompt à coller pour Claude Code / Codex / Gemini / Copilot.
- **Une démo interactive** sur Hugging Face Spaces et une **UI Streamlit locale** plus
  riche, toutes deux embarquant le véritable code de visualisation du projet.

> ⚠️ **Un GPU avec CUDA est indispensable** pour l'entraînement et l'évaluation. Les
> applis démo et les flux purement documentaires fonctionnent sur CPU.

## Table des matières

1. [Galerie](#galerie)
2. [Démarrage rapide (≤ 30 s)](#démarrage-rapide--30-s)
3. [Installation — cinq chemins](#installation--cinq-chemins)
   - [A. Installation développeur via `uv`](#a-installation-développeur-via-uv-recommandée)
   - [B. Installation `pip`](#b-installation-pip)
   - [C. Docker](#c-docker)
   - [D. Installeur desktop (Windows / macOS / Linux)](#d-installeur-desktop-windows--macos--linux)
   - [E. Agent LLM « coller et lancer »](#e-agent-llm--coller-et-lancer-)
4. [Reproduire le papier](#reproduire-le-papier)
5. [Démo en ligne & UI locale](#démo-en-ligne--ui-locale)
6. [Arborescence du projet](#arborescence-du-projet)
7. [Référence CLI](#référence-cli)
8. [Documentation](#documentation)
9. [Contribuer](#contribuer)
10. [Citation](#citation)
11. [Remerciements](#remerciements)
12. [Licence](#licence)

## Galerie

<table>
  <tr>
    <td align="center" width="33%"><strong>Fig. 2 — radar HLE / robustesse</strong><br/>
      <img src="assets/diagrams/architecture.svg" width="100%"/><br/>
      <sub>(Placeholder — cf. <code>visualize_results/figures/all_resu_radar_grid_plot.pdf</code>.)</sub>
    </td>
    <td align="center" width="33%"><strong>Fig. 4 — balayage de fenêtre</strong><br/>
      <img src="assets/logo/logo.svg" width="100%"/><br/>
      <sub>(Placeholder — produit par <code>window_lens_resu.ipynb</code>.)</sub>
    </td>
    <td align="center" width="33%"><strong>Sankey — choix de méthode / graphe</strong><br/>
      <img src="assets/social/social-card.svg" width="100%"/><br/>
      <sub>(Placeholder — produit par <code>sankey.ipynb</code>.)</sub>
    </td>
  </tr>
</table>

> Après une reproduction complète, exécutez
> `uv run python scripts/export_assets.py --only figures --dpi 200` pour rastériser
> les vraies figures du papier dans `assets/export/figures/`.

## Démarrage rapide (≤ 30 s)

```bash
git clone https://github.com/jajupmochi/swiss-river-network-benchmark.git
cd swiss-river-network-benchmark
uv sync --no-cache

# Vérification rapide de l'installation.
uv run srn --help
uv run srn version
```

Lancer la démo interactive en local :

```bash
uv run srn app streamlit          # UI locale complète
# ou
uv run srn app gradio             # Gradio (la même que le HF Space)
```

## Installation — cinq chemins

> Le benchmark est conçu pour que chacun puisse choisir l'installation adaptée à son rôle.

### A. Installation développeur via `uv` (recommandée)

```bash
git clone https://github.com/jajupmochi/swiss-river-network-benchmark.git
cd swiss-river-network-benchmark
uv sync --no-cache                     # environnement reproductible depuis uv.lock
uv run srn --help                      # point d'entrée console
```

Extras optionnels :

```bash
uv sync --all-extras                   # tout
uv pip install -e '.[app]'             # applis démo uniquement
uv pip install -e '.[docs]'            # mkdocs + i18n + mike
uv pip install -e '.[dev]'             # ruff, pytest, nbmake, pre-commit
```

### B. Installation `pip`

```bash
python -m pip install 'swissrivernetwork[app]'          # une fois publié sur PyPI
# ou depuis un clone :
pip install -e '.[app]'
```

Python 3.12 minimum. Un GPU avec CUDA 12.1+ est requis pour l'entraînement ; les démos
fonctionnent en CPU.

### C. Docker

> Les charges d'entraînement nécessitent un GPU NVIDIA + `nvidia-container-toolkit`.

```bash
docker compose up app                  # UI Streamlit sur http://localhost:8501
docker compose run --rm train srn sweep
```

### D. Installeur desktop (Windows / macOS / Linux)

Si vous êtes hydrologue ou praticien et que vous ne voulez pas toucher au terminal,
téléchargez un installeur prêt à l'emploi depuis la
[page Releases](https://github.com/jajupmochi/swiss-river-network-benchmark/releases)
et double-cliquez :

| Plateforme | Artefact |
| --- | --- |
| Windows 10/11 x64 | `SwissRiverNetworkBenchmark-<ver>-win64.exe` |
| macOS (Apple Silicon) | `SwissRiverNetworkBenchmark-<ver>.dmg` |
| Linux x64 | `SwissRiverNetworkBenchmark-<ver>-x86_64.AppImage` |

Le bundle desktop lance l'UI Streamlit localement, charge un checkpoint inclus, et
visualise les prédictions sur n'importe quelle station sans Python ni CUDA. Les
entraînements demandent toujours un GPU — pour la recherche, préférez **A / B / C**.

Construction locale (avancée) :

```bash
uv sync --all-extras
uv run pyinstaller packaging/swissrivernetwork.spec
```

### E. Agent LLM « coller et lancer »

Ouvrez votre agent de codage préféré (Claude Code, Codex, Gemini CLI ou GitHub Copilot
CLI) et collez le prompt ci-dessous. L'agent clonera, installera, préparera les
données, exécutera les tests de fumée et lancera l'UI — en un seul tour.

> 📎 Le playbook complet, copiable tel quel, est à
> [`.claude/skills/install/SKILL.md`](.claude/skills/install/SKILL.md).

```text
Install the Swiss River Network Benchmark by cloning
https://github.com/jajupmochi/swiss-river-network-benchmark.git into the current directory,
running `uv sync --no-cache --all-extras`, smoke-checking with `uv run pytest -q`, and
then starting the Streamlit UI via `uv run srn app streamlit`. Read
.claude/skills/install/SKILL.md for the complete playbook before starting.
```

## Reproduire le papier

```bash
# 0. Préparer les trois splits du jeu de données.
uv run srn prepare-data

# 1. Recherche d'hyperparamètres (par méthode, par graphe). Exemple : LSTM sur swiss-2010.
uv run srn tune -m lstm -g swiss-2010 -n 200 -wl 90

# 2. Évaluer les checkpoints tunés et écrire les tables wl=90.
uv run srn evaluate

# 3. Balayage de la longueur de fenêtre → Fig. 4 + dimension HLE de la Fig. 2.
uv run srn sweep

# 4. Rendre les figures depuis les CSV.
uv run jupyter lab swissrivernetwork/benchmark/visualize_results/
```

Le balayage se déroule strictement en deux phases :

1. **ISOLATED** — `lstm`, `transformer` (× PE dans `{learnable, sinusoidal, rope}`).
   Écrit les prédictions `wt_hat` dans `dump/predictions/<path_extra_keys>-evalwl{W}/`.
2. **GRAPHLET** — `graphlet`, `transformer_graphlet`. Lit les dumps de la phase 1 comme
   caractéristiques des voisins.

> 🐛 **Note de reproductibilité.** Les anciens balayages étaient affectés par deux bugs
> désormais corrigés :
>
> 1. *Fuite de fenêtre d'évaluation* dans le chemin de dump isolated (toutes les lignes
>    à W ≠ 90 pour Graphlet lisaient silencieusement la prédiction à plus grand W).
> 2. *NaN d'outer-join* dans `merge_graphlet_dfs` à W > trained_wl.
>
> Les deux correctifs sont sur `main` depuis `4daeff3`. Les lignes Graphlet et
> Transformer-Graphlet à W ≠ 90 doivent être regénérées ; W = 90 n'est pas affecté.
> Détails dans [`CHANGELOG.md`](CHANGELOG.md).

## Démo en ligne & UI locale

| Cible | Commande | Notes |
| --- | --- | --- |
| Hugging Face Space | [🤗 ouvrir la démo](https://huggingface.co/spaces/jajupmochi/swiss-river-network-benchmark) | Front-end Gradio, prédictions en direct sur des checkpoints embarqués. |
| Gradio local | `uv run srn app gradio` | Même appli que le HF Space. |
| Streamlit local | `uv run srn app streamlit` | Onglets Explore / Predict / Compare, réutilise le code de visualisation du projet. |
| Installeur desktop | double-cliquer sur `.exe` / `.dmg` / `.AppImage` | Même UI Streamlit, empaquetée. |

Les deux démos incluent une **visualisation en temps réel** tirée directement des
notebooks sous `swissrivernetwork/benchmark/visualize_results/` — aucune donnée
simulée.

## Arborescence du projet

```
swiss-river-network-benchmark/
├── swissrivernetwork/
│   ├── cli.py                              # entrée `srn` (typer, relaie vers les drivers)
│   ├── benchmark/
│   │   ├── data_preparation.py             # construction des splits
│   │   ├── ray_tune.py                     # recherche d'hyperparamètres
│   │   ├── ray_evaluation.py               # évaluation au test
│   │   ├── run_win_len_sweep.py            # balayage de fenêtre (Fig. 4 / HLE)
│   │   ├── train_single_model.py
│   │   ├── train_isolated_station.py
│   │   ├── util.py                         # merge_graphlet_dfs, get_evaluation_path_keys…
│   │   ├── dataset.py                      # lecteurs + SequenceDataset(Windowed)
│   │   └── visualize_results/              # notebooks reproduisant chaque figure
│   ├── app/
│   │   ├── gradio_app.py                   # HF Space + Gradio local
│   │   └── streamlit_app.py                # UI locale avec visualisation temps réel
│   └── …                                   # helpers expérimentaux, modules NN, utils
├── assets/                                  # logo, social card, diagramme d'architecture
├── docs/                                    # site MkDocs Material (en / zh / de / fr)
├── packaging/                               # spec PyInstaller + scripts de plateforme
├── scripts/                                 # export_assets.py, helpers de fumée
├── tests/
├── .claude/skills/                          # skills Claude Code (install, run-benchmark)
├── pyproject.toml                           # métadonnées PEP 621, extras, console scripts
├── CITATION.cff
├── CHANGELOG.md
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── SECURITY.md
└── LICENSE
```

## Référence CLI

Toutes les sous-commandes de `srn` relaient vers les drivers canoniques — elles
existent pour qu'après `pip install` vous disposiez d'une commande `srn` installable :

| Commande | Driver sous-jacent |
| --- | --- |
| `srn prepare-data` | `python -m swissrivernetwork.benchmark.data_preparation` |
| `srn tune -m <method> -g <graph> …` | `python -m swissrivernetwork.benchmark.ray_tune …` |
| `srn evaluate` | `python -m swissrivernetwork.benchmark.ray_evaluation` |
| `srn sweep` | `python -m swissrivernetwork.benchmark.run_win_len_sweep` |
| `srn train-single` | `python -m swissrivernetwork.benchmark.train_single_model` |
| `srn train-isolated` | `python -m swissrivernetwork.benchmark.train_isolated_station` |
| `srn app gradio` | Lancer la démo Gradio. |
| `srn app streamlit` | Lancer l'UI Streamlit locale. |
| `srn version` | Afficher la version installée du paquet. |

Les flags spécifiques aux drivers se mettent après `--` :

```bash
uv run srn tune -m transformer_embedding -g swiss-2010 -n 200 -wl 90 -pe rope
```

Liste complète des flags dans
[`.claude/skills/run-benchmark/SKILL.md`](.claude/skills/run-benchmark/SKILL.md) ou via
`--help` sur chaque driver.

## Documentation

Le site de documentation est construit avec MkDocs Material et publié dans quatre
langues.

| URL | Langue |
| --- | --- |
| <https://jajupmochi.github.io/swiss-river-network-benchmark/> | English (par défaut) |
| <https://jajupmochi.github.io/swiss-river-network-benchmark/zh/> | 简体中文 |
| <https://jajupmochi.github.io/swiss-river-network-benchmark/de/> | Deutsch |
| <https://jajupmochi.github.io/swiss-river-network-benchmark/fr/> | Français |

Sections : **Prise en main**, **Guide utilisateur (pour hydrologues)**, **Tutoriels**,
**Reproduction du papier**, **Référence API**, **Explications**, **Guide
développeur**, **Citation**, **FAQ**.

Build local :

```bash
uv pip install -e '.[docs]'
uv run mkdocs serve
```

## Contribuer

- Lisez [`CONTRIBUTING.md`](CONTRIBUTING.md) avant d'ouvrir une issue / PR.
- Les rapports de bug, demandes de fonctionnalité et questions de reproduction du
  papier ont chacun leur template.
- La participation est régie par le [Code de conduite](CODE_OF_CONDUCT.md).
- Les questions de sécurité suivent un processus de divulgation privée — voir
  [`SECURITY.md`](SECURITY.md).

Les tâches ouvertes et les jalons à venir sont suivis dans le
[tracker d'issues GitHub](https://github.com/jajupmochi/swiss-river-network-benchmark/issues).

## Citation

Si vous utilisez ce benchmark dans un travail académique, citez le logiciel et le
papier.

**Logiciel (DOI Zenodo à compléter après la première release) :**

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

**Papier (soumission ICPR 2026 — placeholder, mettre à jour après acceptation) :**

```bibtex
@inproceedings{jia_transformers_rivertemp_2026,
  author    = {Linlin Jia and Benjamin Fankhauser},
  title     = {Benchmarking Transformers on Spatio-Temporal River Water Temperature Modeling},
  booktitle = {International Conference on Pattern Recognition (ICPR)},
  year      = {2026},
  note      = {Under review.}
}
```

GitHub embarque aussi un [`CITATION.cff`](CITATION.cff) lisible par machine — le
bouton « Cite this repository » sur la page du repo résout automatiquement les deux
entrées.

## Remerciements

Ce benchmark s'appuie sur les travaux antérieurs de **Benjamin Fankhauser** et du
groupe hydrologie de l'**Université de Berne**. Nous remercions l'Office fédéral de
l'environnement (OFEV), l'**Office des déchets, de l'eau, de l'énergie et de l'air
(AWEL)** de Zurich, ainsi que les observatoires publics partenaires, pour les mesures
de station qui rendent ce jeu de données possible.

Infrastructure et outillage : [PyTorch](https://pytorch.org/),
[PyTorch Geometric](https://pyg.org/), [Ray Tune](https://www.ray.io/ray-tune),
[Hugging Face](https://huggingface.co/), [Gradio](https://www.gradio.app/),
[Streamlit](https://streamlit.io/), [MkDocs Material](https://squidfunk.github.io/mkdocs-material/),
[uv](https://github.com/astral-sh/uv).

## Licence

MIT — voir [`LICENSE`](LICENSE).
