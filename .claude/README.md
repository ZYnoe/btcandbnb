# `.claude/` — Project-scoped agent context

This directory exists so any LLM agent (Claude Code, future agents, etc.)
can pick up the project without the user re-explaining context every
session.

## Layout

```
.claude/
├── README.md         ← this file
└── skills/           ← actionable workflow skills
    ├── run-experiment.md       — submit + monitor + fetch SLURM jobs
    ├── update-qiskit-api.md    — handle Qiskit API breaking changes
    ├── extend-to-n-assets.md   — refactor 2-asset → N-asset
    └── analyze-results.md      — generate Markdown report + plot
```

The bulk of project context lives in **`/CLAUDE.md`** at the project
root, which Claude Code loads automatically.

## How skills work

Each `skills/<name>.md` is a single Markdown file with YAML
frontmatter:

```yaml
---
name: skill-name
description: When to invoke this skill. Should be specific.
---

# Skill body — workflow, code patterns, gotchas
```

When the user asks something that matches a skill's `description`,
the agent should follow that skill's body directly. Don't re-derive
the workflow from scratch.

## Adding a new skill

1. Pick a focused workflow that recurs across sessions.
2. Write a single-purpose `.md` file under `skills/`.
3. Frontmatter `description` should be specific enough that the
   agent can decide when to invoke vs. skip.
4. Body should be terse, actionable, and include the gotchas
   discovered while doing the work the first time.
5. Commit + push so the skill is available to future agents.

## What does NOT belong here

- General Python tutorials.
- Code that should be in `src/`.
- Risk disclaimers (already in `runner.py` and every report).
- API documentation that belongs in docstrings.
- Anything user-specific that shouldn't be in version control.
