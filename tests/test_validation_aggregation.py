"""Regression test for the validation-metric aggregation fix (B1, audit 2026-07-04).

For a *windowed nowcasting* validation set the per-station ``validation_ave_rmse`` must
aggregate the overlapping windows to one prediction per day (``longest_history``). The
trial/checkpoint selection metric ``validation_mse`` must stay over *all* samples
(unaggregated) — the same value it had before the fix — so model selection, and therefore
the published results, are unaffected.

This test drives ``compute_all_metrics_unified`` directly and asserts exactly that: turning
aggregation on (nowcasting) vs off (forecasting) leaves ``validation_mse`` byte-identical
while changing ``validation_ave_rmse``.
"""

import types

import torch
import torch.nn as nn

from swissrivernetwork.benchmark.training import compute_all_metrics_unified


class _IdentityScaler:
    """Stand-in normalizer whose inverse transform is the identity."""

    def inverse_transform(self, x):
        return x


def _run(is_extrapolation: bool):
    # Two overlapping windows predict each of days 10/11/12. The first window per day (the
    # longest history) is accurate; the second is wrong. Aggregation keeps only the first.
    days = [10, 10, 11, 11, 12, 12]
    preds = [1.0, 9.0, 3.0, 9.0, 5.0, 9.0]
    targets = [1.0, 1.0, 3.0, 3.0, 5.0, 5.0]
    epoch_days = [torch.tensor([[float(d)]]) for d in days]
    masks = [torch.tensor([[True]]) for _ in days]
    pred_t = [torch.tensor([[p]], dtype=torch.float32) for p in preds]
    targ_t = [torch.tensor([[t]], dtype=torch.float32) for t in targets]

    def extractor(inp, iter_idx):
        start, end = iter_idx
        return torch.cat(inp[start:end], dim=0).flatten()

    dataloader_valid = types.SimpleNamespace(dataset=types.SimpleNamespace(window_len=5))
    return compute_all_metrics_unified(
        epoch_days,
        masks,
        pred_t,
        targ_t,
        dataloader_valid,
        {"s0": _IdentityScaler()},
        nn.MSELoss(),
        station_iterator=[("s0", (0, len(days)))],
        station_data_extractor=extractor,
        is_extrapolation=is_extrapolation,
    )


def test_aggregation_fixes_ave_rmse_without_changing_selection_metric():
    mse_off, ave_off, rmse_off = _run(is_extrapolation=True)  # forecasting -> no aggregation
    mse_on, ave_on, rmse_on = _run(is_extrapolation=False)  # nowcasting  -> aggregation on

    # Safety property: the selection metric is byte-identical with/without aggregation, so the
    # committed results (chosen by validation_mse) are unaffected by the fix.
    assert mse_on == mse_off
    assert rmse_on == rmse_off

    # The fix engages: aggregating to the longest-history window per day recovers the accurate
    # predictions, so validation_ave_rmse drops (to ~0) vs the unaggregated value (~4.4).
    assert ave_off > 1.0
    assert ave_on < 1e-6

    # Sanity: the unaggregated MSE is the mean of squared residuals over all six samples.
    assert abs(mse_off - (0 + 64 + 0 + 36 + 0 + 16) / 6) < 1e-4
