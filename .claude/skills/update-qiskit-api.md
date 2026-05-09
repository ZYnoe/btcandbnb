---
name: update-qiskit-api
description: When a Qiskit ecosystem upgrade breaks imports or changes the Sampler/QAOA API, update src/_qiskit_compat.py to handle the new path. Use when a user reports import errors after `uv sync` brings a newer qiskit, or when CI starts failing on a quantum test.
---

# Update Qiskit API

The Qiskit stack ships breaking API changes every 6-9 months.
`src/_qiskit_compat.py` is the **only** module that imports qiskit
directly. All other modules access qiskit symbols through this shim.
Keep that boundary intact.

## Decision tree when something breaks

1. **Is it an `ImportError`?** → adding a new path to the relevant
   try/except block, see "Adding a new fallback" below.
2. **Is it a `TypeError` from a Sampler call?** → `build_sampler()`
   needs a new branch (V1 → V2 → V3 transitions land here).
3. **Is it a runtime error inside QAOA / SamplingVQE?** → check if the
   `MinimumEigenOptimizer` API changed (in `qiskit_optimization.algorithms`).
4. **Is it the `algorithm_globals.random_seed` no longer existing?**
   → reproducibility regression; check the new replacement (often
   `numpy.random.default_rng(seed)` passed explicitly).

## Adding a new fallback path

Pattern (already used for QAOA/NumPyMinimumEigensolver/COBYLA):

```python
for _modname in ("new_path_name", "qiskit_algorithms", "qiskit.algorithms"):
    try:
        _mod = __import__(_modname, fromlist=["QAOA"])
        QAOA = getattr(_mod, "QAOA", None)
        NumPyMinimumEigensolver = getattr(_mod, "NumPyMinimumEigensolver", None)
        if QAOA is not None and NumPyMinimumEigensolver is not None:
            break
    except Exception as e:
        logger.debug("%s import failed: %s", _modname, e)
```

Always prepend new paths (newer versions first). Keep old paths so
older qiskit installations still work.

## Sampler version transitions

`build_sampler()` currently tries:

1. `qiskit.primitives.StatevectorSampler` (V2, ideal simulator) — preferred for qiskit 2.x
2. `qiskit_aer.primitives.SamplerV2` — V2 with shot noise, transpile-quirky
3. `qiskit_aer.primitives.Sampler` — V1, deprecated but works on older qiskit-algorithms

If a V3 emerges (e.g., `BackendSamplerV3`), prepend a new try block at the
top. Keep V2 and V1 fallbacks intact.

## Verifying a change

Two tests that must still pass:

```bash
# 1. All 19 unit tests
uv run pytest tests -v

# 2. End-to-end with all quantum solvers
uv run python main.py --use-quantum --quantum-mode both --quantum-solver all \
  --quantum-weight-step 0.2 --qaoa-reps 1 --qaoa-shots 256 \
  --output-dir /tmp/verify --no-plots

# Check: comparison_summary.csv must have all 6 quantum rows with success=True
column -t -s, /tmp/verify/comparison_summary.csv
rm -rf /tmp/verify
```

If end-to-end works locally, push and re-run on SLURM (see
`.claude/skills/run-experiment.md`).

## What NOT to do

- Don't `import qiskit` outside `_qiskit_compat.py` — that's the whole
  point of the shim.
- Don't catch `ImportError` and silently set things to `None` without
  logging — failures must be visible. Use `logger.debug` (not `error`)
  so it doesn't spam users with extra installs.
- Don't drop V1 fallback even if you're certain it's unused — older
  qiskit-algorithms releases still need it.
