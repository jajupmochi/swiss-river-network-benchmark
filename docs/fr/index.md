# Swiss River Network Benchmark

!!! info "Soumission ICPR 2026"
    Code de référence, jeux de données et figures en open source pour
    **« Benchmarking Transformers on Spatio-Temporal River Water
    Temperature Modeling »**.

Swiss River Network Benchmark est un benchmark reproductible pour la
prévision spatio-temporelle de la température de l'eau en rivière. Il
propose trois jeux de données graphes réels, huit méthodes de
référence et le pipeline exact d'entraînement / évaluation / balayage
du papier.

<div class="grid cards" markdown>

-   :material-rocket-launch-outline: __Installation en 30 s__

    ---

    ```bash
    git clone https://github.com/jajupmochi/swiss-river-network-benchmark.git
    cd swiss-river-network-benchmark
    uv sync --no-cache
    uv run srn --help
    ```

-   :material-flask-outline: __Reproduire le papier__

    ---

    Un seul CLI couvre l'entraînement, l'évaluation et le balayage.
    Chaque figure a son notebook.

-   :material-application-braces: __Démo en ligne__

    ---

    Hugging Face Space, Streamlit local, installeur desktop — même
    couche de visualisation.

-   :material-book-open-variant: __Référence API__

    ---

    Générée depuis le code source avec `mkdocstrings`.

</div>

## Les autres pages ne sont pas encore traduites en français

MkDocs retombe automatiquement sur l'anglais en attendant. Les
pull-requests sont bienvenues.

[:material-github: GitHub](https://github.com/jajupmochi/swiss-river-network-benchmark){ .md-button .md-button--primary }
[:material-emoticon: Hugging Face Space](https://huggingface.co/spaces/jajupmochi/swiss-river-network-benchmark){ .md-button }
