# Developer guide

Conventions for hacking on the benchmark.

- [Working style](working-style.md) — the conventions that keep
  contributions from stepping on each other.
- [Release process](release-process.md) — what a `v*.*.*` tag triggers
  and how to cut one.

## At-a-glance

- Python 3.12 + `uv` is the only supported env.
- Don't edit the driver modules under `swissrivernetwork/benchmark/`
  directly for new features — add a new file and wire it up from the
  CLI.
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/).
- There is no test suite: smoke-test by running the relevant entry
  point.
