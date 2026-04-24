# Release process

## Cut a release

1. Bump `version` in `pyproject.toml`.
2. Update [`CHANGELOG.md`](https://github.com/jajupmochi/swiss-river-network-benchmark/blob/main/CHANGELOG.md)
   — move the `[Unreleased]` section into a dated `[x.y.z]` header.
3. Commit: `chore(release): v<x.y.z>`.
4. Tag and push:
   ```bash
   git tag -a v<x.y.z> -m "v<x.y.z>"
   git push origin v<x.y.z>
   ```

## What the tag triggers

`.github/workflows/release.yml` runs on every `v*.*.*` tag:

1. Build wheel + sdist on `ubuntu-latest`.
2. Build a desktop bundle on `ubuntu-latest`, `macos-latest`,
   `windows-latest` using `packaging/swissrivernetwork.spec`.
3. Upload all artefacts to the GitHub release — one `.whl`, one sdist,
   three desktop bundles.

## Follow-ups you do by hand

- Publish the wheel to PyPI (`uv run twine upload dist/*.whl`) once the
  project has a token.
- Update the Hugging Face Space: `git push hf main:main`.
- Mint a Zenodo DOI and paste it into
  [`CITATION.cff`](https://github.com/jajupmochi/swiss-river-network-benchmark/blob/main/CITATION.cff).
- Deploy the docs version: `uv run mike deploy --push v<x.y.z> latest`.
