<div align="center">

<img src="assets/social/banner.svg" alt="Swiss River Network Benchmark" width="100%"/>

# 瑞士河网基准（Swiss River Network Benchmark）

<strong>面向河流水温时空建模的 Transformer 基准测试</strong><br/>
<em>ICPR 2026 投稿 · 开源参考代码、数据集与论文图表</em>

[![CI](https://github.com/jajupmochi/swiss-river-network-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/jajupmochi/swiss-river-network-benchmark/actions/workflows/ci.yml)
[![Docs](https://github.com/jajupmochi/swiss-river-network-benchmark/actions/workflows/docs.yml/badge.svg)](https://jajupmochi.github.io/swiss-river-network-benchmark/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![Code style: ruff](https://img.shields.io/badge/style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Hugging Face Space](https://img.shields.io/badge/🤗%20Space-live%20demo-blue)](https://huggingface.co/spaces/jajupmochi/swiss-river-network-benchmark)
[![Paper](https://img.shields.io/badge/ICPR-2026-8A2BE2)](#引用)

**语言：** [English](README.md) · **简体中文** · [Deutsch](README.de.md) · [Français](README.fr.md)

</div>

---

## 一句话介绍

瑞士河网基准是一个**可复现**的河流水温时空预测开源基准，提供：

- **三套图数据集** —— `swiss-1990`、`swiss-2010`、`zurich` —— 基于瑞士水文监测站。
- **八种参考方法** —— LSTM、Graphlet、LSTM + 站点 embedding、ST-GNN、
  Transformer（可学习 / 正弦 / RoPE 位置编码），以及图感知的
  Transformer 变体（Transformer-Graphlet、Transformer-Embedding、Transformer-ST-GNN）。
- **完整的论文流水线** —— Ray Tune 超参搜索、测试评估、
  窗口长度扫描（图 4 + HLE）、高斯 / 脉冲噪声鲁棒性实验（图 5–6），
  以及从 CSV 还原所有论文图表的可视化 notebook。
- **五种安装方式** —— `uv`、`pip`、Docker、桌面一键安装器，以及
  面向 Claude Code / Codex / Gemini / Copilot 的"粘贴即运行"提示。
- **可交互 demo**，在 Hugging Face Spaces 上线，本地还有更完整的 **Streamlit 界面**，
  两者都直接嵌入了项目真正的可视化代码。

> ⚠️ 训练与评估**必须使用带 CUDA 的 GPU**。Demo 应用和只看文档的工作流可在 CPU 上运行。

## 目录

1. [图库](#图库)
2. [30 秒快速上手](#30-秒快速上手)
3. [安装 —— 五种方式](#安装--五种方式)
   - [A. 通过 `uv` 的开发者安装](#a-通过-uv-的开发者安装推荐)
   - [B. `pip` 安装](#b-pip-安装)
   - [C. Docker](#c-docker)
   - [D. 桌面安装器（Windows / macOS / Linux）](#d-桌面安装器windows--macos--linux)
   - [E. LLM 智能体"粘贴即运行"](#e-llm-智能体粘贴即运行)
4. [复现论文](#复现论文)
5. [在线 demo 与本地 UI](#在线-demo-与本地-ui)
6. [项目结构](#项目结构)
7. [CLI 参考](#cli-参考)
8. [文档](#文档)
9. [贡献](#贡献)
10. [引用](#引用)
11. [致谢](#致谢)
12. [许可证](#许可证)

## 图库

<table>
  <tr>
    <td align="center" width="33%"><strong>图 2 —— HLE / 鲁棒性雷达图</strong><br/>
      <img src="assets/diagrams/architecture.svg" width="100%"/><br/>
      <sub>（占位图 —— 见 <code>visualize_results/figures/all_resu_radar_grid_plot.pdf</code>。）</sub>
    </td>
    <td align="center" width="33%"><strong>图 4 —— 窗口长度扫描</strong><br/>
      <img src="assets/logo/logo.svg" width="100%"/><br/>
      <sub>（占位图 —— 由 <code>window_lens_resu.ipynb</code> 生成。）</sub>
    </td>
    <td align="center" width="33%"><strong>Sankey —— 方法 / 图的选择</strong><br/>
      <img src="assets/social/social-card.svg" width="100%"/><br/>
      <sub>（占位图 —— 由 <code>sankey.ipynb</code> 生成。）</sub>
    </td>
  </tr>
</table>

> 完整复现论文后，运行 `uv run python scripts/export_assets.py --only figures --dpi 200`
> 可将真正的论文图表导出为 PNG，放到 `assets/export/figures/` 下用于嵌入。

## 30 秒快速上手

```bash
git clone https://github.com/jajupmochi/swiss-river-network-benchmark.git
cd swiss-river-network-benchmark
uv sync --no-cache

# 冒烟测试
uv run srn --help
uv run srn version
```

本地启动交互 demo：

```bash
uv run srn app streamlit          # 完整本地 UI
# 或
uv run srn app gradio             # Gradio（也是 HF Space 使用的那个）
```

## 安装 —— 五种方式

> 基准的设计让你可以根据自己的身份选择合适的安装路径。

### A. 通过 `uv` 的开发者安装（推荐）

```bash
git clone https://github.com/jajupmochi/swiss-river-network-benchmark.git
cd swiss-river-network-benchmark
uv sync --no-cache                     # 基于 uv.lock 的可复现环境
uv run srn --help                      # 控制台入口
```

可选 extras：

```bash
uv sync --all-extras                   # 全部
uv pip install -e '.[app]'             # 仅 demo 应用
uv pip install -e '.[docs]'            # mkdocs + i18n + mike
uv pip install -e '.[dev]'             # ruff、pytest、nbmake、pre-commit
```

### B. `pip` 安装

```bash
python -m pip install 'swissrivernetwork[app]'          # 上 PyPI 之后
# 或从 clone 安装：
pip install -e '.[app]'
```

最低 Python 3.12。训练需要 CUDA 12.1+ 的 GPU；demo 可纯 CPU 运行。

### C. Docker

> 训练工作负载需要 NVIDIA GPU + nvidia-container-toolkit。

```bash
docker compose up app                  # Streamlit UI，访问 http://localhost:8501
docker compose run --rm train srn sweep
```

### D. 桌面安装器（Windows / macOS / Linux）

如果你是水文学家或一线从业者，不想碰命令行，可以在
[Releases 页面](https://github.com/jajupmochi/swiss-river-network-benchmark/releases)
下载现成的安装包，双击即可运行：

| 平台 | 安装包 |
| --- | --- |
| Windows 10/11 x64 | `SwissRiverNetworkBenchmark-<ver>-win64.exe` |
| macOS（Apple Silicon） | `SwissRiverNetworkBenchmark-<ver>.dmg` |
| Linux x64 | `SwissRiverNetworkBenchmark-<ver>-x86_64.AppImage` |

桌面包会在本地启动 Streamlit UI，加载自带的 checkpoint，对任一监测站进行预测和可视化，
无需安装 Python 或 CUDA。如果要跑训练仍然需要 GPU —— 研究用途请选 **A / B / C**。

本地构建（进阶）：

```bash
uv sync --all-extras
uv run pyinstaller packaging/swissrivernetwork.spec
```

### E. LLM 智能体"粘贴即运行"

打开你喜欢的编程智能体（Claude Code、Codex、Gemini CLI 或 GitHub Copilot CLI），
把下面这段提示词粘贴过去。智能体会在一个回合内完成：clone、安装、准备数据、
跑冒烟测试、启动 UI。

> 📎 完整可复制的 playbook 位于
> [`.claude/skills/install/SKILL.md`](.claude/skills/install/SKILL.md)。

```text
Install the Swiss River Network Benchmark by cloning
https://github.com/jajupmochi/swiss-river-network-benchmark.git into the current directory,
running `uv sync --no-cache --all-extras`, smoke-checking with `uv run pytest -q`, and
then starting the Streamlit UI via `uv run srn app streamlit`. Read
.claude/skills/install/SKILL.md for the complete playbook before starting.
```

## 复现论文

```bash
# 0. 准备三套数据划分
uv run srn prepare-data

# 1. 超参搜索（按方法、按图）。示例：swiss-2010 上的 LSTM
uv run srn tune -m lstm -g swiss-2010 -n 200 -wl 90

# 2. 评估调好的 checkpoint，输出 wl=90 的表格
uv run srn evaluate

# 3. 窗口长度扫描 → 论文图 4 + 图 2 的 HLE 维度
uv run srn sweep

# 4. 从 CSV 生成图表
uv run jupyter lab swissrivernetwork/benchmark/visualize_results/
```

窗口扫描严格分两步：

1. **ISOLATED** —— `lstm`、`transformer`（PE 取 `{learnable, sinusoidal, rope}`）。将
   `wt_hat` 预测写入 `dump/predictions/<path_extra_keys>-evalwl{W}/`。
2. **GRAPHLET** —— `graphlet`、`transformer_graphlet`。读取第一阶段的 dump 作为邻居特征。

> 🐛 **复现注意事项** 早期扫描受到两个已修复 bug 的影响：
>
> 1. *评估窗口泄漏* —— isolated dump 路径在 W ≠ 90 时被 graphlet 误读为最长窗口的预测。
> 2. *Outer-join NaN* —— `merge_graphlet_dfs` 在 W > trained_wl 时会出现 NaN。
>
> 两个修复自 `4daeff3` 起已合入 `main`。需要重新跑 Graphlet 和 Transformer-Graphlet
> 在 W ≠ 90 的扫描行；W = 90 的结果不受影响。详见
> [`CHANGELOG.md`](CHANGELOG.md)。

## 在线 demo 与本地 UI

| 目标 | 命令 | 说明 |
| --- | --- | --- |
| Hugging Face Space | [🤗 打开 demo](https://huggingface.co/spaces/jajupmochi/swiss-river-network-benchmark) | Gradio 前端，用自带 checkpoint 做实时预测 |
| 本地 Gradio | `uv run srn app gradio` | 和 HF Space 同一个应用 |
| 本地 Streamlit | `uv run srn app streamlit` | Explore / Predict / Compare 三个 tab，复用项目已有可视化代码 |
| 桌面安装器 | 双击 `.exe` / `.dmg` / `.AppImage` | 打包后的 Streamlit UI |

两个 demo 都带**实时可视化**，直接复用 `swissrivernetwork/benchmark/visualize_results/`
下 notebook 里的图表代码 —— 没有假数据。

## 项目结构

```
swiss-river-network-benchmark/
├── swissrivernetwork/
│   ├── cli.py                              # `srn` 入口（typer，转发到驱动脚本）
│   ├── benchmark/
│   │   ├── data_preparation.py             # 构建数据划分
│   │   ├── ray_tune.py                     # 超参搜索
│   │   ├── ray_evaluation.py               # 测试评估
│   │   ├── run_win_len_sweep.py            # 窗口扫描（图 4 / HLE）
│   │   ├── train_single_model.py
│   │   ├── train_isolated_station.py
│   │   ├── util.py                         # merge_graphlet_dfs, get_evaluation_path_keys…
│   │   ├── dataset.py                      # 数据读取 + SequenceDataset(Windowed)
│   │   └── visualize_results/              # 生成论文所有图表的 notebook
│   ├── app/
│   │   ├── gradio_app.py                   # HF Space + 本地 Gradio
│   │   └── streamlit_app.py                # 带实时可视化的本地 UI
│   └── …                                   # 实验辅助、NN 模块、工具函数
├── assets/                                  # logo、社交卡片、架构图
├── docs/                                    # MkDocs Material 站点（en / zh / de / fr）
├── packaging/                               # PyInstaller spec + 各平台入口脚本
├── scripts/                                 # export_assets.py、冒烟测试
├── tests/
├── .claude/skills/                          # Claude Code skills（install、run-benchmark）
├── pyproject.toml                           # PEP 621 元数据、extras、控制台脚本
├── CITATION.cff
├── CHANGELOG.md
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── SECURITY.md
└── LICENSE
```

## CLI 参考

`srn` 的所有子命令都直接转发到对应的驱动脚本 —— 它们存在是为了让你在 `pip install`
之后就有一个可执行的 `srn` 命令：

| 命令 | 底层驱动 |
| --- | --- |
| `srn prepare-data` | `python -m swissrivernetwork.benchmark.data_preparation` |
| `srn tune -m <method> -g <graph> …` | `python -m swissrivernetwork.benchmark.ray_tune …` |
| `srn evaluate` | `python -m swissrivernetwork.benchmark.ray_evaluation` |
| `srn sweep` | `python -m swissrivernetwork.benchmark.run_win_len_sweep` |
| `srn train-single` | `python -m swissrivernetwork.benchmark.train_single_model` |
| `srn train-isolated` | `python -m swissrivernetwork.benchmark.train_isolated_station` |
| `srn app gradio` | 启动 Gradio demo |
| `srn app streamlit` | 启动 Streamlit 本地 UI |
| `srn version` | 输出已安装的包版本 |

驱动特定参数放在 `--` 之后：

```bash
uv run srn tune -m transformer_embedding -g swiss-2010 -n 200 -wl 90 -pe rope
```

完整参数列表见 [`.claude/skills/run-benchmark/SKILL.md`](.claude/skills/run-benchmark/SKILL.md)，
或对任一驱动加 `--help`。

## 文档

完整文档站点用 MkDocs Material 构建，四语同步发布。

| URL | 语言 |
| --- | --- |
| <https://jajupmochi.github.io/swiss-river-network-benchmark/> | English（默认） |
| <https://jajupmochi.github.io/swiss-river-network-benchmark/zh/> | 简体中文 |
| <https://jajupmochi.github.io/swiss-river-network-benchmark/de/> | Deutsch |
| <https://jajupmochi.github.io/swiss-river-network-benchmark/fr/> | Français |

章节包括：**入门指南**、**水文学家使用指南**、**教程**、**论文复现**、
**API 参考**、**原理讲解**、**开发者指南**、**引用**、**FAQ**。

本地预览：

```bash
uv pip install -e '.[docs]'
uv run mkdocs serve
```

## 贡献

- 开 issue / PR 之前请先读 [`CONTRIBUTING.md`](CONTRIBUTING.md)。
- Bug 报告、功能请求、论文复现问题各有专用 issue 模板。
- 参与活动遵守[行为准则](CODE_OF_CONDUCT.md)。
- 安全问题走私下披露 —— 见 [`SECURITY.md`](SECURITY.md)。

待办任务与里程碑在
[GitHub issue 区](https://github.com/jajupmochi/swiss-river-network-benchmark/issues)跟踪。

## 引用

如果本基准在学术工作中对你有帮助，请同时引用软件和论文。

**软件（首个 release 后补 Zenodo DOI）：**

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

**论文（ICPR 2026 投稿 —— 占位，录用后更新）：**

```bibtex
@inproceedings{jia_transformers_rivertemp_2026,
  author    = {Linlin Jia and Benjamin Fankhauser},
  title     = {Benchmarking Transformers on Spatio-Temporal River Water Temperature Modeling},
  booktitle = {International Conference on Pattern Recognition (ICPR)},
  year      = {2026},
  note      = {Under review.}
}
```

GitHub 还带一份机读的 [`CITATION.cff`](CITATION.cff) —— 仓库页面上的 "Cite this
repository" 按钮会自动生成上面两条引用。

## 致谢

本基准建立在 **Benjamin Fankhauser** 以及**伯尔尼大学**水文组的早期工作基础上。
感谢瑞士联邦环境局（FOEN）、苏黎世**废弃物、水、能源与空气管理局（AWEL）**，
以及合作的公共观测机构提供监测数据，让本数据集成为可能。

基础设施与工具：[PyTorch](https://pytorch.org/)、
[PyTorch Geometric](https://pyg.org/)、[Ray Tune](https://www.ray.io/ray-tune)、
[Hugging Face](https://huggingface.co/)、[Gradio](https://www.gradio.app/)、
[Streamlit](https://streamlit.io/)、[MkDocs Material](https://squidfunk.github.io/mkdocs-material/)、
[uv](https://github.com/astral-sh/uv)。

## 许可证

MIT —— 见 [`LICENSE`](LICENSE)。
