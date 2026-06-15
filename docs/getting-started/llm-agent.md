# LLM-agent install

The full playbook lives at
[`.claude/skills/install/SKILL.md`](https://github.com/jajupmochi/swiss-river-network-benchmark/blob/main/.claude/skills/install/SKILL.md).
Paste the snippet below into any compatible coding agent:

```text
Install the Swiss River Network Benchmark by cloning
https://github.com/jajupmochi/swiss-river-network-benchmark.git into the
current directory, running `uv sync --no-cache --all-extras`,
smoke-checking with `uv run pytest -q`, and then starting the Streamlit
UI via `uv run srn app streamlit`. Read .claude/skills/install/SKILL.md
for the complete playbook before starting.
```

Compatible agents:

- [Claude Code](https://claude.com/claude-code) (picks up the skill
  automatically via `.claude/skills/`).
- OpenAI Codex, Gemini CLI, GitHub Copilot CLI — follow the `SKILL.md`
  steps as a checklist.

The playbook covers prerequisites, install, CLI / GPU / package smoke
tests, and UI launch — every step is idempotent.
