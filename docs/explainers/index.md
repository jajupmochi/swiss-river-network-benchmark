# Explainers

Long-form write-ups on non-obvious decisions inside the codebase.

- [Eval-window leakage](eval-window-leakage.md) — the Phase-1 dump path
  that made pre-`4daeff3` graphlet sweep rows unreliable at W ≠ 90.
- [Graphlet NaN fix](graphlet-nan-fix.md) — why `merge_graphlet_dfs`
  switched from outer to inner join at W > trained_wl.

These pages exist to keep the *why* out of the git log and into the
docs, where people actually read it.
