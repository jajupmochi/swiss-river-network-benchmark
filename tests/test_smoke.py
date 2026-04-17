"""Import every benchmark module to catch syntax/import regressions early.

No GPU, no data, no network required — this test only verifies that each
module parses and its top-level imports resolve.
"""

import importlib

import pytest

BENCHMARK_MODULES = [
    "swissrivernetwork",
    "swissrivernetwork.benchmark",
    "swissrivernetwork.benchmark.dataset",
    "swissrivernetwork.benchmark.model",
    "swissrivernetwork.benchmark.training",
    "swissrivernetwork.benchmark.transformer",
    "swissrivernetwork.benchmark.util",
    "swissrivernetwork.benchmark.visualize",
    "swissrivernetwork.benchmark.ray_tune",
    "swissrivernetwork.benchmark.ray_evaluation",
    "swissrivernetwork.benchmark.train_single_model",
    "swissrivernetwork.benchmark.train_isolated_station",
]


@pytest.mark.parametrize("module_name", BENCHMARK_MODULES)
def test_module_imports(module_name: str) -> None:
    importlib.import_module(module_name)
