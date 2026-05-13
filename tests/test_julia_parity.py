"""Cross-language parity: Julia grid_search at step=s must match Python grid_search
at the same step, on the same μ/Σ. This is the Julia analogue of Invariant 2
(quantum-exact ≡ classical-grid)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Skip the whole module if the Julia binary or the project directory is missing.
JULIA = shutil.which("julia")
JULIA_PROJECT = REPO_ROOT / "julia"
if JULIA is None or not (JULIA_PROJECT / "Project.toml").exists():
    pytest.skip("julia binary or julia/ project not available", allow_module_level=True)


def _write_market_fixture(tmp_dir: Path) -> dict:
    """Synthetic 2-asset market that drives both Python grid_search and
    the Julia run.jl pipeline. We write the same returns.csv + basic_stats.json
    that src/runner.py would write, so Julia sees identical inputs."""
    rng = np.random.default_rng(42)
    n = 250
    btc = rng.normal(0.0015, 0.02, n)
    bnb = rng.normal(0.0005, 0.03, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    returns = pd.DataFrame({"BTC-USD": btc, "BNB-USD": bnb}, index=idx)
    mu = returns.mean().to_numpy() * 365
    Sigma = returns.cov().to_numpy() * 365

    returns.to_csv(tmp_dir / "returns.csv")
    stats = {
        "tickers": list(returns.columns),
        "rows": len(returns),
        "start": str(returns.index.min().date()),
        "end": str(returns.index.max().date()),
        "annualization_factor": 365,
        "annualized_return": {t: float(mu[i]) for i, t in enumerate(returns.columns)},
        "annualized_volatility": {
            t: float(np.sqrt(Sigma[i, i])) for i, t in enumerate(returns.columns)
        },
        "covariance": [[float(x) for x in row] for row in Sigma],
        "correlation": [[float(x) for x in row] for row in returns.corr().to_numpy()],
    }
    (tmp_dir / "basic_stats.json").write_text(json.dumps(stats))
    return {"returns": returns, "mu": mu, "Sigma": Sigma}


def test_julia_grid_matches_python_grid(tmp_path):
    """Same returns, same step → same best BTC weight, exactly."""
    fixture = _write_market_fixture(tmp_path)

    # Python grid (in-process)
    from src.optimizer import grid_search as py_grid

    step = 0.05
    _grid, py_best = py_grid(
        fixture["returns"], fixture["mu"], fixture["Sigma"],
        step=step, objective="maximize_sharpe",
    )

    # Julia grid (subprocess)
    result = subprocess.run(
        [JULIA, f"--project={JULIA_PROJECT}", str(REPO_ROOT / "julia" / "run.jl"),
         "--outputs", str(tmp_path), "--step", str(step)],
        capture_output=True, text=True, timeout=300, cwd=REPO_ROOT,
    )
    assert result.returncode == 0, f"julia run.jl failed:\n{result.stderr}"

    julia_summary = pd.read_csv(tmp_path / "comparison_summary_julia.csv")
    julia_grid_row = julia_summary[julia_summary["method"] == "Classic Julia Grid Search"]
    assert len(julia_grid_row) == 1, "expected exactly one Julia grid row"

    py_btc = float(py_best["btc_weight"])
    jl_btc = float(julia_grid_row.iloc[0]["btc_weight"])
    assert py_btc == pytest.approx(jl_btc, abs=1e-9), (
        f"Julia grid best {jl_btc} != Python grid best {py_btc} at step {step}"
    )
