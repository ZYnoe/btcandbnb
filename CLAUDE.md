# Portfolio Optimizer — Project Context for LLM Maintainers

> **Read this file before making any changes.** It captures the load-bearing
> design decisions that aren't obvious from grepping the code. The README
> targets users; this targets the next agent that has to extend or repair
> the project.

## What this project is

A teaching/comparison codebase that runs **classical** (Grid Search,
Mean-Variance) and **quantum/hybrid** (QAOA, SamplingVQE via Qiskit)
portfolio optimization on BTC/BNB historical data, side-by-side, on the
same μ and Σ. The point isn't to win an optimization contest — for 2
assets the classical answer is essentially closed-form. The point is:

1. Demonstrate that QUBO/Ising encoding works (quantum exact ≡ classical grid).
2. Quantify how much wall time the quantum mapping costs (3-5 orders of magnitude).
3. Show how QAOA / SamplingVQE approximate the exact answer on the same problem.

## What this project is NOT

- **Not** a trading system. Every CLI output ends with a 4-line risk
  disclaimer. Maintain this — see `runner.py::RISK_DISCLAIMER`.
- **Not** a general N-asset framework. Hardcoded for 2 (see
  `.claude/skills/extend-to-n-assets.md` if extending).
- **Not** a high-performance quantum library. Uses Qiskit's reference
  implementations on top of qiskit-aer's CPU simulator.

## Build & run — `uv` only

System Python (3.9 on this machine, varies on cluster) is too old. uv
auto-fetches a 3.10+ interpreter from `pyproject.toml::requires-python`
and creates `.venv/` here.

```bash
uv sync --all-extras                       # install (first run ~2 min)
uv run pytest tests -v                     # 19 tests, ~6 s
uv run python main.py --use-quantum        # full classical+quantum end-to-end
uv run python analyze.py outputs/          # post-run Markdown report + plot
```

Never use system `pip` / `python3`. Never write `requirements.txt`
("user explicitly dislikes it; pyproject.toml + uv.lock are the only
manifests" — feedback memory).

## Architecture (file-by-file)

### `src/` (the actual logic)

| File | Role | Notes |
|---|---|---|
| `data.py` | yfinance download → align → μ, Σ, ρ | Tries `Adj Close` then `Close`; passes `auto_adjust=False` to pin behavior. |
| `metrics.py` | Pure-function risk/return indicators | No I/O. ~10 functions: returns, Sharpe, Sortino, MaxDD, VaR/CVaR. |
| `optimizer.py` | Grid Search + scipy SLSQP + 3 benchmarks | All 2-asset specific; `optimizer.evaluate_benchmarks` returns 100% BTC, 100% BNB, 50/50. |
| `quantum_binary.py` | 2-qubit asset-selection QUBO with budget | `min λxᵀΣx − μᵀx` s.t. cardinality. |
| `quantum_discrete_weights.py` | One-hot K+1 qubit weight grid | `min Σ cₖ zₖ` s.t. `Σ zₖ = 1`. K = `1/weight_step`. |
| `_qiskit_compat.py` | Qiskit version-tolerant import shim | **Only** module that imports qiskit directly. |
| `parallel.py` | ProcessPoolExecutor for the 6 quantum solvers | + a CPU-bound smoke test (`--parallel-smoke-test`). |
| `comparison.py` | Unified row schema + CSV/JSON writer | `COLUMNS` tuple defines the schema. |
| `runner.py` | End-to-end orchestration | Writes results **incrementally** (see Invariant 4). |
| `plots.py` | 9 PNG charts | Each plot wrapped in try/except. |
| `config.py` | Config dataclass + CLI validation | All CLI flags terminate here. |
| `utils.py` | seed, logger, safe_import | |

### Top-level

- `main.py` — argparse → `Config` → `run`. Stays thin.
- `analyze.py` — Markdown report + verdict plot from any outputs/ dir.
- `submit.sh` — SLURM job; runs smoke test, pytest, classical-only,
  then full classical+quantum, then a validation Python snippet.
- `pyproject.toml` + `uv.lock` — uv-managed deps.
- `tests/` — 4 files, 19 tests. Quantum tests `pytest.importorskip`.

## Load-bearing invariants (do NOT break)

1. **Classical and quantum solvers share the same μ, Σ.** Computed once
   in `data.compute_market_data`, passed everywhere. Any divergence
   makes the comparison meaningless.

2. **Quantum exact (NumPyMinimumEigensolver) ≡ classical grid at the
   same step.** This is the cross-validation in `analyze.py::_cross_validation`
   and in submit.sh's tail Python check. If it stops holding, the
   QUBO encoding (Σ off-diagonal coefficient, linear/quadratic split)
   has a bug — investigate `quantum_binary.py::_build_qp` first.

3. **`comparison_summary.csv` always has one row per requested
   solver.** Even on failure: `success=False`, `error_message`
   populated. Never silently drop. The `_on_error` callback in
   `runner.py` enforces this for parallel quantum tasks.

4. **Outputs are written incrementally.** Every solver row hits disk
   before the next solver starts. Reason: SLURM time limit kills
   were losing partial state. See `runner.py::_flush_summary`.

5. **All quantum imports go through `_qiskit_compat.py`.** Other
   modules never `import qiskit` directly. Reason: Qiskit ships
   breaking changes every 6-9 months; the shim probes multiple paths
   in priority order.

6. **Risk disclaimer.** Every CLI run + every Markdown report ends
   with the 4-line risk notice. Never weaken this.

## Known traps & how they're handled

| Trap | Symptom | Fix in code |
|---|---|---|
| Qiskit 2.x removed V1 Sampler | "Invalid circuits, expected Sequence[QuantumCircuit]" | `build_sampler()` tries `StatevectorSampler` (V2) first |
| `OMP_NUM_THREADS=$NCPU` + multiprocessing | 25% CPU efficiency, single core busy | `submit.sh` pins all BLAS vars to 1 |
| SLURM `--output=logs/%j.out` with missing dir | Job CANCELLED, exit 0:53, elapsed 0:00:00 | Use root-level `--output=portfolio-opt-%j.out` |
| yfinance `auto_adjust` flipped | Missing `Adj Close` column | Pass `auto_adjust=False`, fall back to `Close` |
| Spawn vs fork on Linux | qiskit-aer C++ globals corrupt under fork | `mp.get_context("spawn")` — eats ~5 s import per worker but is reproducible |
| QAOA on 21 qubits + reps=2 | Single solver runs minutes; SLURM 30-min timeout | `QUANTUM_WEIGHT_STEP=0.1` cuts 21 qubits to 11 (~50× faster) |

## Testing

`uv run pytest tests -v` covers:
- Metrics math (returns compounding, Sharpe, MaxDD ranges, VaR/CVaR signs)
- Grid + MV + benchmarks
- Quantum: candidate count, one-hot constraint, exact ≡ grid
- Binary: budget=1 → 1 selected, budget=2 → both selected

Quantum tests use `pytest.importorskip("qiskit_optimization")` so
they auto-skip on machines without qiskit.

## SLURM workflow (Yonsei cluster)

- SSH alias: `yonsei_jang` (home `/home/zeyu/`)
- Default partition: `amd2` (32-core CPU nodes, no GPU despite g1/g2 naming)
- Submit from project root: `sbatch submit.sh`
- Default resources: `--cpus-per-task=4 --mem=8G --time=00:30:00`
- See `.claude/skills/run-experiment.md` for the full workflow

## Recent history (commits worth knowing)

- `820b3e1` — initial commit
- `c5c591c` — incremental flush of comparison_summary
- `de27729` — ProcessPoolExecutor for 6 quantum solvers (the parallelism fix)
- `bdfe3b4` — `analyze.py` programmatic report

## When to ask the user

- Window changes (start/end dates) — defaults are last 2 years.
- Adding new tickers or extending N — has multi-file consequences.
- Anything that touches the risk disclaimer.
- Disabling tests or skipping pytest in submit.sh.

## When to NOT ask

- Bug fixes that don't change behavior.
- Refactors that preserve all `comparison_summary` columns.
- Adding plot variants (each plot is fault-tolerant; can't break others).
- Updating Qiskit imports — that's `.claude/skills/update-qiskit-api.md`.
