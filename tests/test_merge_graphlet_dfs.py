"""Regression test for :func:`swissrivernetwork.benchmark.util.merge_graphlet_dfs`.

The bug history
---------------
Before commit ``4daeff3``, this function used ``pd.merge(..., how="outer")``.
At ``eval_wl > trained_wl`` the neighbour dumps were missing the last
``eval_wl - trained_wl`` days, which the outer join filled with NaN — and
those NaNs then leaked silently into the sweep tables.

The fix: inner join. If a neighbour is missing a day, drop that day from
the target dataframe. Graphlet is only able to predict on days where
**every** neighbour already has a prediction.

This test pins that contract.
"""

from __future__ import annotations

import pandas as pd
import pytest

from swissrivernetwork.benchmark.util import merge_graphlet_dfs


def _target(days: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "epoch_day": range(days),
            "air_temperature": [10.0 + i * 0.1 for i in range(days)],
            "water_temperature": [8.0 + i * 0.1 for i in range(days)],
        }
    )


def _neighbor(name: str, days: range | list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "epoch_day": list(days),
            f"wt_hat_{name}": [5.0 + d * 0.1 for d in days],
        }
    )


def test_inner_join_at_matching_window() -> None:
    """When neighbours cover all target days, every row survives."""
    df = _target(days=10)
    neighs = [_neighbor("a", range(10)), _neighbor("b", range(10))]
    out = merge_graphlet_dfs(df, neighs)
    assert len(out) == 10
    assert set(out.columns) >= {"epoch_day", "wt_hat_a", "wt_hat_b"}
    assert out["wt_hat_a"].isna().sum() == 0
    assert out["wt_hat_b"].isna().sum() == 0


def test_inner_join_drops_missing_neighbor_days() -> None:
    """Days without a neighbour prediction must be dropped, not NaN-filled.

    This mirrors ``eval_wl > trained_wl`` in the paper sweep.
    """
    df = _target(days=10)
    # Neighbour 'a' is missing the last 3 days (simulating the short dump).
    neighs = [_neighbor("a", range(7)), _neighbor("b", range(10))]
    out = merge_graphlet_dfs(df, neighs)
    assert len(out) == 7, "inner join should drop rows where any neighbour is missing"
    assert out["epoch_day"].max() == 6
    assert out["wt_hat_a"].isna().sum() == 0
    assert out["wt_hat_b"].isna().sum() == 0


def test_inner_join_no_common_days_yields_empty() -> None:
    """Edge case: no neighbour/target intersection → empty dataframe (no NaN)."""
    df = _target(days=5)
    neighs = [_neighbor("a", range(10, 20))]
    out = merge_graphlet_dfs(df, neighs)
    assert len(out) == 0


def test_nan_in_neighbor_triggers_assertion() -> None:
    """The function keeps a defensive NaN guard on neighbour columns."""
    df = _target(days=5)
    bad = _neighbor("a", range(5))
    bad.loc[2, "wt_hat_a"] = float("nan")
    with pytest.raises(AssertionError):
        merge_graphlet_dfs(df, [bad])
