# Benchmark pipeline

The executable drivers that turn raw data into tuned models and result CSVs. Each is a
`python -m swissrivernetwork.benchmark.*` entry point and is also exposed through the
[`srn` CLI](cli.md).

::: swissrivernetwork.benchmark.data_preparation

::: swissrivernetwork.benchmark.ray_tune

::: swissrivernetwork.benchmark.ray_evaluation

::: swissrivernetwork.benchmark.train_single_model

::: swissrivernetwork.benchmark.train_isolated_station

::: swissrivernetwork.benchmark.run_win_len_sweep
