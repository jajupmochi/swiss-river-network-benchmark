# Swiss River Network Benchmark

!!! info "ICPR-2026-Einreichung"
    Open-Source-Referenzcode, Datensätze und Abbildungen zu
    **„Benchmarking Transformers on Spatio-Temporal River Water
    Temperature Modeling"**.

Der Swiss River Network Benchmark ist ein reproduzierbarer Benchmark für
die Vorhersage der Wassertemperatur in Flüssen. Er enthält drei reale
Graph-Datensätze, acht Referenzmethoden und die exakte Trainings- /
Evaluations- / Sweep-Pipeline des Papers.

<div class="grid cards" markdown>

-   :material-rocket-launch-outline: __Installation in 30 s__

    ---

    ```bash
    git clone https://github.com/jajupmochi/swiss-river-network-benchmark.git
    cd swiss-river-network-benchmark
    uv sync --no-cache
    uv run srn --help
    ```

-   :material-flask-outline: __Paper reproduzieren__

    ---

    Ein CLI-Kommando für Training / Evaluation / Sweep. Jede Abbildung
    hat ein eigenes Notebook.

-   :material-application-braces: __Live-Demo__

    ---

    Hugging Face Space, lokales Streamlit, Desktop-Installer — alle mit
    derselben Visualisierungsebene.

-   :material-book-open-variant: __API-Referenz__

    ---

    Aus dem Quellcode generiert mit `mkdocstrings`.

</div>

## Die übrigen Seiten sind noch nicht auf Deutsch übersetzt

MkDocs zeigt automatisch den englischen Originaltext, bis Übersetzungen
folgen. Pull-Requests willkommen.

[:material-github: GitHub](https://github.com/jajupmochi/swiss-river-network-benchmark){ .md-button .md-button--primary }
[:material-emoticon: Hugging Face Space](https://huggingface.co/spaces/jajupmochi/swiss-river-network-benchmark){ .md-button }
