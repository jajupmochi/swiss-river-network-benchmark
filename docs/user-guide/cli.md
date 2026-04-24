# CLI reference

Every subcommand of `srn` forwards to a Python module under
`swissrivernetwork.benchmark`. Forwarding preserves `__main__` semantics
so the drivers can still be invoked via `python -m` if you prefer.

| Command | Driver |
| --- | --- |
| `srn prepare-data` | `swissrivernetwork.benchmark.data_preparation` |
| `srn tune -m <method> -g <graph> -n <n> -wl <wl> [-pe <pe>]` | `swissrivernetwork.benchmark.ray_tune` |
| `srn evaluate` | `swissrivernetwork.benchmark.ray_evaluation` |
| `srn sweep` | `swissrivernetwork.benchmark.run_win_len_sweep` |
| `srn train-single` | `swissrivernetwork.benchmark.train_single_model` |
| `srn train-isolated` | `swissrivernetwork.benchmark.train_isolated_station` |
| `srn app gradio` | `swissrivernetwork.app.gradio_app` |
| `srn app streamlit` | Streamlit launches `swissrivernetwork/app/streamlit_app.py` |
| `srn version` | Print the installed package version |

Driver-specific flags go after `--`:

```bash
uv run srn tune -- -m transformer_embedding -g swiss-2010 -n 200 -wl 90 -pe rope
```

## Common flags

- `-m / --method` — one of `lstm`, `lstm_embedding`, `graphlet`, `stgnn`,
  `transformer`, `transformer_graphlet`, `transformer_embedding`,
  `transformer_stgnn`.
- `-g / --graph` — one of `swiss-1990`, `swiss-2010`, `zurich`.
- `-n / --num-trials` — Ray Tune trial count.
- `-wl / --window-len` — temporal window length (days).
- `-pe / --positional-encoding` — `sinusoidal`, `learnable`, `rope`.

See the `.claude/skills/run-benchmark/SKILL.md` playbook for the full
reference including the environment variables that must stay set
(notably `RAY_CHDIR_TO_TRIAL_DIR=0`).
