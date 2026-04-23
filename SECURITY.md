# Security Policy

## Supported Versions

The Swiss River Network Benchmark is a research artefact. Security support is provided on
a best-effort basis for the latest release and the `main` branch.

| Version | Supported |
| --- | --- |
| `main` (unreleased) | ✅ |
| `0.1.x` | ✅ |
| `< 0.1` | ❌ |

## Reporting a Vulnerability

**Please do not open a public GitHub issue for security concerns.**

Instead, report privately via one of the following channels:

1. GitHub's "Report a vulnerability" button
   (<https://github.com/jajupmochi/swiss-river-network-benchmark/security/advisories/new>).
2. Email: **linlin.jia@unibe.ch** — please include "swiss-river-network-benchmark security"
   in the subject line. For encrypted communication, request the maintainer's PGP key in
   the first message.

When reporting, please include:

- A description of the vulnerability and the affected component.
- Steps to reproduce (minimal code or command).
- Version / commit hash.
- Potential impact: data leak, arbitrary code execution, denial of service, supply-chain, …
- Whether you intend to publicly disclose and on what timeline.

### What to expect

- Acknowledgement within **5 working days**.
- An initial triage within **10 working days** indicating severity and likely fix ETA.
- A private security advisory on GitHub that credits the reporter (unless you prefer to
  stay anonymous).
- Coordinated disclosure after a fix is released.

## Scope

This repository distributes research code. Particular areas where security bugs matter:

- Dependency vulnerabilities in `uv.lock` or `requirements.txt`. Dependabot is configured
  for `pip` under `.github/dependabot.yml`.
- Accidental secret leaks (API keys, tokens, dataset credentials) committed to the repo.
- Code that fetches, executes, or serializes untrusted input — e.g. the Gradio / Streamlit
  demo apps, or notebook widgets that accept user uploads.
- Unsafe model / pickle loading from untrusted checkpoints.

## Out of scope

- Best practices or hardening suggestions without an associated exploit — these are
  welcome as regular issues.
- Vulnerabilities in third-party services (Hugging Face, Weights & Biases, Ray) should be
  reported upstream.
- Scientific correctness issues (e.g. "the numbers in Table 3 don't match your code") —
  use the **Paper reproduction** issue template instead.

Thanks for helping keep the benchmark and its users safe.
