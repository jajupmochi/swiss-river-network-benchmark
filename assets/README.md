# `assets/` — visual identity and documentation assets

This directory holds the polished, version-controlled visuals that the README,
the documentation site, the social preview, and the demo apps link to. Files
here are **hand-authored SVGs** so they scale cleanly and render identically on
GitHub, PyPI, Hugging Face, and the documentation site.

```
assets/
├── logo/           # brand identity
│   ├── logo.svg            # 512×512 square logo + wordmark
│   ├── logo-mark.svg       # 128×128 glyph-only mark (good for avatars)
│   └── favicon.svg         # 32×32 simplified favicon
├── social/
│   ├── social-card.svg     # 1200×630 Open Graph / Twitter card
│   └── banner.svg          # 1600×320 wide banner for README / docs hero
└── diagrams/
    └── architecture.svg    # data → model zoo → drivers → outputs
```

## Rasterization

GitHub renders SVGs directly, so you generally do not need PNG copies. When
you *do* need a PNG (PyPI long-description, CMS uploads, etc.), run:

```bash
# requires `cairosvg` (included in the `docs` extra) or `rsvg-convert`
uv run python scripts/export_assets.py --format png --dpi 2
```

The exporter writes rasterised copies to `assets/export/` (git-ignored).

## Status

These are **placeholder** visuals — they encode the correct colour palette
(slate + sky-blue + violet + amber) and the intended composition (river graph
with a temperature thermometer accent) so the README does not ship without a
brand, but they should be regenerated with a professional designer or an
image-generation model before a v1.0 release. See issue #logo-refresh in the
tracker.

## Licensing

All files under `assets/` are released under the same MIT licence as the rest
of the repository (see `../LICENSE`).
