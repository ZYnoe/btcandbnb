# BTC / BNB Portfolio Optimizer — Classical vs Quantum (Qiskit)

A small but complete research-grade tool that asks two questions on the same
historical data:

1. **How should we split between BTC and BNB** to obtain a good
   risk-adjusted return on the past window?
2. **Do classical (Grid Search, Mean-Variance) and quantum/hybrid (QAOA,
   SamplingVQE) optimizers agree** on the answer when given the same `mu`
   and `Sigma`?

> ⚠️ **Risk notice — read first**
> - Past returns do **not** predict future returns.
> - Crypto assets are extremely volatile and can lose value rapidly.
> - The quantum algorithms here only solve the QUBO under the given
>   `mu` / `Sigma`; they do **not** forecast prices.
> - This tool is for research / education only and is **not** investment advice.

---

## 1. What this project does

- Downloads BTC-USD and BNB-USD daily prices from Yahoo Finance.
- Builds annualized `mu`, `Sigma`, vol and correlation.
- Runs **classical** optimizers (Grid Search + Mean-Variance + 3 benchmarks).
- Runs **quantum** optimizers under the same `mu` / `Sigma`:
  - Binary asset selection (2 qubits, with a budget constraint).
  - One-hot discretized BTC weight (K+1 qubits, K = 1/`weight_step`).
- Compares everything in a single CSV/JSON table and renders 9 plots.

The same `mu` / `Sigma` are shared across classical and quantum to make the
comparison **fair**: any difference is due to the algorithm, not the data.

## 2. Install (uv)

This project is managed exclusively with
[`uv`](https://github.com/astral-sh/uv). `uv` reads `pyproject.toml` +
`uv.lock`, picks a Python 3.10+ interpreter automatically, and creates a
`.venv` in this directory.

```bash
cd portfolio_optimizer
uv sync --all-extras
```

`qiskit-finance` is **not** required — this project hand-builds the
`QuadraticProgram` from `qiskit-optimization` directly. If you want it
anyway, install via the optional extra:

```bash
uv sync --extra finance
```

## 3. Run only the classical optimizer

```bash
uv run python main.py --start 2023-01-01 --end 2025-01-01 \
                      --step 0.01 --objective maximize_sharpe
```

Outputs `outputs/comparison_summary.csv` with five rows: Grid Search,
Mean-Variance, and 3 benchmarks.

## 4. Run classical + quantum comparison

```bash
uv run python main.py --start 2023-01-01 --end 2025-01-01 \
                      --step 0.01 --objective maximize_sharpe \
                      --use-quantum --quantum-mode both --quantum-solver all \
                      --quantum-weight-step 0.05 \
                      --qaoa-reps 1 --qaoa-shots 2048
```

The quantum part adds two binary-selection rows (exact, qaoa, and
optionally sampling_vqe) plus the same set for the discrete-weight model.

## 5. CLI flags (full list)

### Data
- `--start`, `--end` — ISO dates. Default: last 2 years to today.
- `--frequency` — yfinance interval (`1d` default).
- `--annualization-factor` — 365 for crypto (24/7), 252 for equities.
- `--output-dir` — defaults to `outputs/`.
- `--tickers` — override the two tickers (must be exactly 2).

### Classical
- `--step` — grid step over BTC weight (e.g. 0.005, 0.01, 0.05).
- `--objective` — `maximize_sharpe | minimize_volatility | maximize_return | minimize_cvar | constrained_sharpe`.
- `--risk-free-rate` — default 0.
- `--risk-aversion` — λ in `mu·w − λ·wᵀΣw`. Default 1.
- `--max-drawdown` — threshold for `constrained_sharpe` (e.g. -0.5 = -50%).

### Quantum
- `--use-quantum` — turns on quantum runs.
- `--quantum-mode` — `binary | discrete | both`.
- `--quantum-solver` — `exact | qaoa | sampling_vqe | all`.
- `--quantum-weight-step` — discrete grid (e.g. 0.05 → 21 qubits).
- `--qaoa-reps`, `--qaoa-shots`, `--qaoa-seed`.
- `--binary-budget` — 1 (pick one) or 2 (pick both). Default 1.
- `--no-budget-constraint` — drop the cardinality constraint.

### Misc
- `--no-plots`, `--save-intermediate`, `--verbose`.

## 6. Output files

Saved under `--output-dir` (default `outputs/`):

| File | Contents |
|---|---|
| `comparison_summary.csv` / `.json` | Unified rows (one per method/solver). |
| `grid_search_results.csv` | Every grid candidate + all metrics. |
| `quantum_discrete_weight_results.csv` | Pre-evaluated discrete candidates + objective. |
| `quantum_samples.csv` | Top sampled bitstrings/probabilities from QAOA / SamplingVQE. |
| `basic_stats.json` | mu, Sigma, vol, corr, data window. |
| `prices.csv`, `returns.csv` | Only with `--save-intermediate`. |
| `01_prices.png … 09_optimized_cumulative.png` | 9 plots. |

## 7. How to read `comparison_summary.csv`

Each row is one optimizer × solver. Columns:

- `method` — Classic Grid Search, Classic Mean-Variance, Quantum Binary Selection,
  Quantum Discrete Weights, Benchmark *.
- `solver` — `grid step=…`, `SLSQP …`, `exact`, `qaoa`, `sampling_vqe`, `fixed`.
- `btc_weight`, `bnb_weight` — chosen allocation.
- `annual_return`, `annual_volatility`, `sharpe_ratio`, `sortino_ratio`,
  `max_drawdown`, `var_95`, `cvar_95`, `final_cumulative_return` — metrics on the
  same data window.
- `objective` — value of the QUBO objective `λ·wᵀΣw − μᵀw`.
- `runtime_seconds`, `success`, `error_message`, `note`.

**Verification you should run mentally:** the
`Quantum Discrete Weights / exact` row should pick the same BTC weight as
the classical grid (using `--quantum-weight-step` as step, with the
`maximize_return_minus_risk` objective at the same `risk_aversion`). If
the QAOA row picks something different, see the `note` column — it is an
approximate sampling-based solver and may not converge in 1 rep.

## 8. Classical algorithms — quick recap

- **Grid Search** evaluates the BTC weight on a uniform grid in `[0, 1]`,
  computes every metric, and picks the row that satisfies the objective.
  Trivially exhaustive for 2 assets; great as ground truth.
- **Mean-Variance** uses scipy `SLSQP` with the equality constraint
  `w_BTC + w_BNB = 1` and box bounds. For two assets the surface is convex
  for vol-minimization and Sharpe-maximization, so SLSQP converges fast.

## 9. Quantum algorithms — quick recap

- **Binary selection (2 qubits)** encodes "buy / not-buy" indicators
  `x_BTC, x_BNB ∈ {0,1}` and minimizes `λ·xᵀΣx − μᵀx` with a cardinality
  constraint `x_BTC + x_BNB = budget`. We solve it via
  `MinimumEigenOptimizer` wrapping `NumPyMinimumEigensolver` (exact),
  `QAOA`, and optionally `SamplingVQE`.
- **One-hot discrete weights (K+1 qubits)** introduces variables
  `z_0..z_K` for the K+1 candidate BTC weights `0, step, 2·step, …, 1`.
  We precompute `c_k = λ·Var(w_k) − Return(w_k)` and minimize
  `Σ c_k z_k` with the one-hot constraint `Σ z_k = 1`.

Why discretize? A QUBO/Ising formulation needs binary variables. Two-asset
continuous weights are 1D continuous — QAOA can't directly handle that, so
we replace `w` by a one-hot pick from a finite set. Smaller `step` ⇒
finer grid ⇒ more qubits ⇒ slower simulation.

## 10. Why classical usually wins for two assets

For two assets the search space is one-dimensional (BTC weight); a grid
of 100 points covers it densely in milliseconds and finds the *exact*
minimum under every objective. Quantum/hybrid solvers spend qubits and
optimization steps to approximate an answer that the grid already has in
closed form. The quantum module is here for **education and validation**:
it shows the QUBO mapping, the one-hot encoding, the QAOA sampling
interface, and lets you confirm that quantum exact solvers reproduce the
classical answer.

## 11. Educational value of the quantum side

Even though it is not the practical winner for 2 assets, it demonstrates:

- How to encode a portfolio problem as a `QuadraticProgram` and lift it
  into Ising / QUBO form.
- How `MinimumEigenOptimizer` orchestrates a classical solver
  (`NumPyMinimumEigensolver`) and a quantum solver (`QAOA`,
  `SamplingVQE`) with exactly the same interface.
- How sampling noise / optimizer reps matter — small `--qaoa-reps` often
  produces an off-optimum bitstring; bumping `--qaoa-reps 3` and shots
  closes the gap.
- How to verify a quantum result against a known ground truth.

## 12. Qiskit compatibility notes

The Qiskit ecosystem ships breaking changes regularly: `qiskit.algorithms`
has moved to `qiskit-algorithms`, primitive constructors have changed
signatures, and `SamplingVQE` is in `qiskit_algorithms.minimum_eigensolvers`
in newer releases. We handle this via:

- `src/_qiskit_compat.py` probes multiple import paths in priority order.
- Each quantum solver call is wrapped in a `try/except` that returns a
  result row with `success=False` and a readable `error_message` rather
  than crashing the whole pipeline.
- If qiskit is uninstalled altogether, the classical part still runs and
  the quantum rows simply do not appear (a warning is logged).
- We rely on `qiskit-optimization` directly; `qiskit-finance` is
  optional.

## 13. Tests

```bash
uv run pytest tests -v
```

Quantum tests use `pytest.importorskip` so they auto-skip if qiskit is
missing.

## 14. Reproducibility

- `--qaoa-seed` (default 42) seeds NumPy, `random`, and Qiskit's
  `algorithm_globals.random_seed` if available.
- The Aer Sampler is constructed with `seed_transpiler` and per-run
  `seed`; results are deterministic given the same package versions.
- yfinance prices change as Yahoo updates; pin `--start --end` to keep
  the data window stable across runs.

---

**Final reminder.** The optimal allocation reported here is *the optimum
on a past window*. It tells you what *would have* worked and lets you
compare optimizers on equal footing. It does **not** tell you what to do
tomorrow.
