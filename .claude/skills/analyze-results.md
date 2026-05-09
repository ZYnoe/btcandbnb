---
name: analyze-results
description: Generate a Markdown analysis report and verdict plot from any outputs/ directory. Use after a run completes (locally or after rsync from cluster), or when extending the report with new sections.
---

# Analyze Results

`analyze.py` at the project root reads any `outputs/` directory and
writes:

- `<dir>/analysis_report.md` — 9-section structured report
- `<dir>/verdict.png` — Sharpe-vs-BTC curve + ranked bar chart

```bash
uv run python analyze.py outputs_quantum/             # standard
uv run python analyze.py outputs_quantum/ --out r.md  # custom path
uv run python analyze.py outputs_quantum/ --no-plot   # md only
```

## Report sections (and what triggers each)

1. **Snapshot** — file inventory + completeness check. Auto-flags
   "Incomplete run" if `comparison_summary.csv` has < 11 rows.
2. **Data Fundamentals** — μ, σ, ρ from `basic_stats.json` + asset
   domination prediction.
3. **Methods table** — every successful row from `comparison_summary.csv`,
   ranked by Sharpe descending, plus a "Failed runs" sub-section.
4. **Cross-validation** — quantum-exact ↔ classical-grid agreement.
   Auto-checks the 0.05-step neighbour constraint. **Critical sanity gate.**
5. **Approximate solvers** — QAOA / SamplingVQE Δobjective vs exact.
6. **Runtime** — classical-baseline-relative ratios, illustrating QUBO
   mapping cost.
7. **Discrete-weight landscape** — Sharpe peak / vol minimum / drawdown
   minimum locations, plus the diversification paradox callout when
   it's present.
8. **QAOA / SamplingVQE distribution** — top sampled bitstrings (only
   if `quantum_samples.csv` is non-empty).
9. **Caveats** — risk disclaimer (always present).

## When to extend `analyze.py`

If you add a new metric to `comparison.py::COLUMNS`:

- Add the column to the methods table in `_methods_table()`.
- Decide if it deserves its own section or just a column.

If you add a new file to `outputs/`:

- Register it in `_load()` with safe-missing handling.
- Add a section that surfaces what the file teaches.

If you add a new solver type (e.g., D-Wave annealer):

- Update `_methods_table()` color/order if needed.
- The cross-validation in `_cross_validation()` looks for "exact"
  solver — if your new solver type also has an exact mode, expose it.

## Failure tolerance

Every section handles the file being missing:

```python
if d["summary"] is None:
    return md + "_summary missing._\n"
```

Don't `raise` from a section function; return a "missing" message
instead so the report still renders something useful.

## When to ask the user

- Adding a section that materially changes interpretation (e.g.,
  a "Recommended weight" callout that could be misread as advice).
- Removing the caveats section (don't — it's load-bearing).
- Changing the verdict plot's marker scheme (current: blue circle =
  classical, orange star = quantum, grey square = benchmark).

## When to NOT ask

- Fixing typos.
- Adding new metrics columns to existing tables.
- Improving plot aesthetics.
- Tightening "Incomplete run" detection (e.g., add row-count
  assertions).
