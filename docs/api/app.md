# Interactive workbench

Programmatic surface of the local/Hugging-Face workbench (see the
[workbench guide](../user-guide/workbench.md) for the UI). All functions read the released
CSVs or the user's inputs and return Plotly figures / metrics; they are pure and importable,
which is what the automated test suite exercises.

::: swissrivernetwork.app.workbench
    options:
      members:
        - build_demo
        - detect_resources
        - device_choices
        - stations_of
        - rmse
        - mae
        - nse
        - load_user_model
        - run_inference
        - stream_inference
        - residual_analysis
        - seasonal_error
        - threshold_exceedance
        - model_ranking
        - train_eval
        - eval_predictions
        - handle_upload
        - read_any
        - forecast_outlook
        - plot_radar
