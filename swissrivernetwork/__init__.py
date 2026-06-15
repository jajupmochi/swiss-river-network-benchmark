"""Swiss River Network Benchmark — spatio-temporal river water-temperature modeling.

The top-level package is intentionally thin. Heavy numeric dependencies
(torch, ray, torch-geometric) are imported lazily by the submodules under
``swissrivernetwork.benchmark`` so that ``import swissrivernetwork`` is cheap
and usable inside ``srn --help`` or when building docs without a GPU.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("swissrivernetwork")
except PackageNotFoundError:  # editable install without dist-info
    __version__ = "0.0.0+local"

__all__ = ["__version__"]
