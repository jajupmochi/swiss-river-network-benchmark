# Command-line interface

The `srn` console entry point — a thin, argv-forwarding wrapper around the benchmark
drivers (see the [CLI user guide](../user-guide/cli.md) for usage). Every subcommand
forwards its flags to the underlying `python -m swissrivernetwork.benchmark.*` module.

::: swissrivernetwork.cli
