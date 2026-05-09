---
name: extend-to-n-assets
description: Refactor the codebase from 2 assets (BTC/BNB) to N assets. Use when the user asks to add ETH, SOL, etc., or run multi-asset experiments. Significant cross-file changes — clarify scope with user before starting.
---

# Extend to N Assets

The codebase is hardcoded for **2 assets** in several specific spots.
Most of the metric and optimizer code is already vector-friendly, so
the change is more careful surgery than rewrite. **Always ask the user
first** which assets and confirm the change is wanted before starting —
this is a multi-file refactor with comparison-summary schema impact.

## Files that need changes

### `src/config.py`
- `tickers: tuple[str, ...]` already accepts N. Default is 2-tuple.
- `binary_budget`: validation hardcodes `(1, 2)`. Generalize to
  `1 ≤ budget ≤ N`.

### `src/data.py`
- ✓ Already N-asset clean. `download_prices` loops over `tickers`,
  `compute_market_data` returns `mu` (N,), `Sigma` (N, N).

### `src/metrics.py`
- ✓ Already pure vector ops. No changes.

### `src/optimizer.py`
- `grid_search` is the bottleneck. 1-D scan over BTC weight. For N
  assets you need either:
  - Simplex grid via `itertools.product` filtered to `sum=1`. 100^N
    blows up; cap at N=3 or use coarser step.
  - Or skip grid for N>2 and use only Mean-Variance.
- `mean_variance_optimize` is N-asset clean (uses `len(weights)`,
  scipy SLSQP with sum-constraint).
- `evaluate_benchmarks` needs new entries: equal-weight, random-weight,
  market-cap-weighted (if you have caps).

### `src/quantum_binary.py`
- `ASSET_LABELS = ("BTC", "BNB")` and `VAR_NAMES` are hardcoded.
  Generalize to use config tickers.
- `_build_qp` already works for any N — `qp.binary_var(v)` for each
  asset; quadratic dict over all (i, i) and (i, j) pairs.
- `_decode` handles any number of bits; just generalize the
  `(weights, bitstring, selected, note)` mapping.

### `src/quantum_discrete_weights.py`
- **Hardest file**: 2-asset one-hot becomes (K+1)^N qubits for N assets.
  At N=3, K=10: 11^3 = 1331 qubits. Statevector infeasible.
- Realistic options:
  - For N=3-4: use logarithmic encoding (K+1 candidates → log2(K+1)
    qubits per asset, no one-hot constraint needed but decode is harder).
  - Limit discrete model to N=2 only; add a log warning for N≥3.

### `src/comparison.py`
- `COLUMNS` tuple has `btc_weight`, `bnb_weight`. Generalize to
  `weight_<ticker>` columns OR add a single `weights` JSON column.
  Prefer per-ticker columns for CSV grep-ability.

### `src/plots.py`
- Plots 4 (Sharpe vs BTC weight) and 5 (MaxDD vs BTC weight) become
  N-D. Replacement options: PCA scatter, pairwise weight scatter, or
  just skip for N>2.
- Plot 3 (return-vol scatter) still works — every method has one
  point regardless of N.
- Plot 9 (cumulative returns) still works.

### `src/parallel.py`
- ✓ No changes — passes whatever data through to workers.

### `src/runner.py`
- `optimized_selections` list at end uses `[btc_weight, bnb_weight]`.
  Generalize to N-vector from the columns.

### `analyze.py`
- `_data_section` 2-asset domination check needs to handle N. For
  N=2, keep current logic; for N>2, drop the "X strictly dominates Y"
  call-out.
- `_methods_table` shows `BTC` column. Generalize to N columns OR
  collapse to a single "weights" column.

### `tests/`
- All quantum tests use 2-asset fixtures. Add N=3 fixtures alongside;
  don't replace.

## Don't do

- **Don't add new "BTC" / "BNB" string constants.** Use indices and
  the config's `tickers` tuple.
- **Don't break the comparison_summary schema.** If you add columns,
  append them at the end so existing analysis tools (analyze.py,
  user notebooks) still work.
- **Don't silently change the default tickers.** Keep BTC/BNB default;
  N-asset is opt-in via `--tickers BTC-USD ETH-USD SOL-USD`.

## Validation after the refactor

```bash
# 1. Original 2-asset behavior must be unchanged
uv run pytest tests -v
uv run python main.py --use-quantum --no-plots
diff outputs/comparison_summary.csv known_good_2asset.csv

# 2. New N-asset path
uv run python main.py --tickers BTC-USD ETH-USD SOL-USD --use-quantum \
  --quantum-mode binary --quantum-solver exact --no-plots
# Check: 3 binary variables in QUBO, sum-budget constraint includes all 3
```

## Estimated scope

- Skeleton (rename hardcoded labels): ~2-4 hours
- Mean-Variance + binary quantum: works without much extra effort
- Discrete weights + plots: 1-2 days if done properly
- Tests update: half day

If the user just wants 3-4 assets with binary selection, do the
skeleton + binary path and explicitly skip discrete weights for N>2
with a warning.
