# Graphlet NaN at `W > trained_wl`

## The symptom

For any `graphlet` / `transformer_graphlet` row with evaluation window
**greater than** the training window, the output tables contained NaNs
in the last `W - trained_wl` timestamps.

## The root cause

`merge_graphlet_dfs` used to join the target station's test dataframe
and the neighbours' predicted dataframes with `how="outer"`. At
`W > trained_wl`, the neighbours simply don't have predictions for the
extra horizon — they were trained at a shorter window. An outer join
fabricates NaN-filled rows for those dates, which then silently leaked
into the sweep metrics.

## The fix

`merge_graphlet_dfs` now uses `how="inner"`. A graphlet method can only
produce an honest prediction on days where **every** neighbour already
has one. If the training window is 90 and the eval window is 180, the
graphlet sweep produces 90 valid predictions and no NaNs.

## Knock-on effects

- Fewer rows per station in sweep CSVs at `W > 90`. That's correct —
  don't scale metrics back up.
- Downstream notebooks aggregate over valid rows only.

See the `merge_graphlet_dfs` docstring in
`swissrivernetwork/benchmark/util.py` for the inline rationale.
