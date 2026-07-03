"""Benchmark package for the Swiss River Network water-temperature models.

Bundles the executable entry points (``data_preparation``, ``ray_tune``,
``ray_evaluation``, ``train_single_model``, ``train_isolated_station``,
``run_win_len_sweep``) together with the dataset primitives (``dataset``),
model definitions (``model``, ``transformer``, ``nn``), the training loop
(``training``), shared glue helpers (``util``), plotting (``visualize``),
and result post-processing (``visualize_results``).
"""
