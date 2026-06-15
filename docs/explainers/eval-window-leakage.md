# Eval-window leakage

## The symptom

Pre-`4daeff3` window-length sweep rows for `graphlet` and
`transformer_graphlet` were *too good* at every W ≠ 90.

## The root cause

Phase 1 (isolated methods) wrote their wl-90-trained predictions into a
dump directory whose path key dropped the eval-window suffix. Phase 2
(graphlet) then read "*the*" dump regardless of the requested eval
window, which in practice meant it always pulled the prediction made
with the longest W available on disk.

This is eval-time leakage: the neighbour features seen at W = 30 were
actually computed with W = 360 worth of context.

## The fix

The dump path now carries a unique `-evalwl{W}` suffix per requested
eval window, and the graphlet reader looks up the exact matching key
before falling back. See
[`4daeff3`](https://github.com/jajupmochi/swiss-river-network-benchmark/commit/4daeff3)
for the commit.

## What you should re-run

- Every graphlet / transformer_graphlet sweep row at W ≠ 90.
- Everything at W = 90 is unaffected because the isolated predictions
  there were written by the canonical (trained) window.

## How to check you're clean

```bash
ls swissrivernetwork/benchmark/dump/predictions/ | grep -E 'evalwl'
```

You should see one directory per requested eval window. If you see
a directory without the suffix, it was written by a pre-`4daeff3`
binary — delete it and re-run the sweep.
