# 瑞士河网基准

!!! info "ICPR 2026 投稿"
    《Benchmarking Transformers on Spatio-Temporal River Water
    Temperature Modeling》对应的开源代码、数据集与论文图表。

瑞士河网基准是一个面向河流水温时空预测的**可复现**基准。提供三套真实
图数据集、八种参考方法，以及和论文完全一致的训练 / 评估 / 扫描流水线。

<div class="grid cards" markdown>

-   :material-rocket-launch-outline: __30 秒安装__

    ---

    ```bash
    git clone https://github.com/jajupmochi/swiss-river-network-benchmark.git
    cd swiss-river-network-benchmark
    uv sync --no-cache
    uv run srn --help
    ```

-   :material-flask-outline: __复现论文__

    ---

    用一个 CLI 跑完整个训练 / 评估 / 扫描。每张论文图都有对应 notebook。

-   :material-application-braces: __在线 demo__

    ---

    Hugging Face Space、本地 Streamlit、双击安装器 —— 三者共享同一套
    可视化代码。

-   :material-book-open-variant: __API 参考__

    ---

    由 mkdocstrings 从源码自动生成。接入新方法时参考即可。

</div>

## 本站点其他章节大多未翻译成中文

MkDocs 会自动回退到英文原文。随着项目成熟会补齐全部译文，欢迎 PR。

[:material-github: GitHub](https://github.com/jajupmochi/swiss-river-network-benchmark){ .md-button .md-button--primary }
[:material-emoticon: Hugging Face Space](https://huggingface.co/spaces/jajupmochi/swiss-river-network-benchmark){ .md-button }
