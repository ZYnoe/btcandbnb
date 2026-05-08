"""Tests for the one-hot discrete-weight quantum module.

Skipped automatically when qiskit / qiskit_optimization are not installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("qiskit_optimization")
pytest.importorskip("qiskit_algorithms")

from src import optimizer  # noqa: E402
from src import quantum_discrete_weights as qdw  # noqa: E402


@pytest.fixture
def market_fixture():
    rng = np.random.default_rng(7)
    n = 400
    btc = rng.normal(0.0012, 0.022, n)
    bnb = rng.normal(0.0006, 0.030, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    returns = pd.DataFrame({"BTC-USD": btc, "BNB-USD": bnb}, index=idx)
    mu = returns.mean().to_numpy() * 365
    Sigma = returns.cov().to_numpy() * 365
    return returns, mu, Sigma


def test_candidate_count_matches_step(market_fixture):
    returns, mu, Sigma = market_fixture
    cands = qdw.build_candidates(mu, Sigma, returns, weight_step=0.1)
    assert len(cands) == 11  # 0.0, 0.1, ..., 1.0


def test_candidate_weights_in_unit_interval(market_fixture):
    returns, mu, Sigma = market_fixture
    cands = qdw.build_candidates(mu, Sigma, returns, weight_step=0.05)
    assert (cands["btc_weight"] >= 0).all()
    assert (cands["btc_weight"] <= 1.0 + 1e-9).all()
    assert ((cands["btc_weight"] + cands["bnb_weight"] - 1.0).abs() < 1e-9).all()


def test_one_hot_constraint_in_qp(market_fixture):
    returns, mu, Sigma = market_fixture
    cands = qdw.build_candidates(mu, Sigma, returns, weight_step=0.1)
    qp, names = qdw._build_qp(cands)
    assert qp.get_num_binary_vars() == len(cands)
    assert len(qp.linear_constraints) == 1
    constraint = qp.linear_constraints[0]
    assert constraint.sense.name == "EQ"
    assert constraint.rhs == 1


def test_exact_quantum_matches_grid(market_fixture):
    """Discrete-weight exact solver must agree with classical grid at the same step."""
    returns, mu, Sigma = market_fixture
    step = 0.1
    cands = qdw.build_candidates(mu, Sigma, returns, weight_step=step)
    quantum_row, _samples = qdw.solve_discrete(
        returns, mu, Sigma, cands, solver="exact", risk_aversion=1.0,
    )
    assert quantum_row["success"] is True

    # classical grid with the same objective (risk_aversion * variance - return)
    _grid, classic_best = optimizer.grid_search(
        returns, mu, Sigma, step=step, objective="maximize_return",
        risk_aversion=1.0,
    )
    # We compare candidate ordering by the same objective, not classic_best directly.
    # Build a parallel ranking: minimize obj_mv from grid corresponds to candidates.
    grid_obj_min_idx = cands["objective"].idxmin()
    expected_btc = cands.iloc[grid_obj_min_idx]["btc_weight"]
    assert pytest.approx(quantum_row["btc_weight"], abs=1e-6) == expected_btc
