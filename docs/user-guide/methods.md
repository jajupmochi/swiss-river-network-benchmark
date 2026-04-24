# Methods

All eight reference methods live under `swissrivernetwork/benchmark/model.py`
and are wired up through `swissrivernetwork/benchmark/training.py`.

| `method` | Temporal backbone | Graph backbone | Typical use |
| --- | --- | --- | --- |
| `lstm` | LSTM | — | Per-station baseline |
| `graphlet` | LSTM | LSTM on neighbor predictions | Two-step graph variant |
| `lstm_embedding` | LSTM | station embedding | Shared per-graph model |
| `stgnn` | LSTM | PyG GNN | Joint spatio-temporal |
| `transformer` | Transformer | — | Isolated Transformer |
| `transformer_graphlet` | Transformer | same as `graphlet` | Graph variant of Transformer |
| `transformer_embedding` | Transformer | station embedding | Shared Transformer |
| `transformer_stgnn` | Transformer | PyG GNN | Joint Transformer ST-GNN |

## Transformer positional encodings

Each Transformer variant can use one of:

- `sinusoidal`
- `learnable`
- `rope`

Set with `-pe` on the CLI or via `config["positional_encoding"]` inside
the `__main__` blocks of the training drivers.

## Where hyperparameters live

- `ray_tune.py` — search space definitions. This is the source of truth
  for *tuned* runs.
- `train_single_model.py` / `train_isolated_station.py` — inline dicts
  in `__main__` for manual / debug training.

See [CLI reference](cli.md) for the mapping from command-line flags to
config keys.
