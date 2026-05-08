"""Tests for the binary-selection quantum module.

Skipped when Qiskit is missing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("qiskit_optimization")
pytest.importorskip("qiskit_algorithms")

from src import quantum_binary as qbin  # noqa: E402


@pytest.fixture
def market_fixture():
    rng = np.random.default_rng(11)
    n = 400
    # Make BTC strictly more attractive (higher mean, lower vol) so budget=1 picks BTC.
    btc = rng.normal(0.002, 0.018, n)
    bnb = rng.normal(0.0001, 0.030, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    returns = pd.DataFrame({"BTC-USD": btc, "BNB-USD": bnb}, index=idx)
    mu = returns.mean().to_numpy() * 365
    Sigma = returns.cov().to_numpy() * 365
    return returns, mu, Sigma


def test_binary_budget_one_selects_single_asset(market_fixture):
    returns, mu, Sigma = market_fixture
    row = qbin.solve_binary(
        returns, mu, Sigma, solver="exact",
        budget=1, no_budget_constraint=False,
    )
    assert row["success"] is True
    bits = row["bitstring"]
    assert bits.count("1") == 1
    assert pytest.approx(row["btc_weight"] + row["bnb_weight"], abs=1e-9) == 1.0


def test_binary_budget_two_picks_both(market_fixture):
    returns, mu, Sigma = market_fixture
    row = qbin.solve_binary(
        returns, mu, Sigma, solver="exact",
        budget=2, no_budget_constraint=False,
    )
    assert row["success"] is True
    assert row["bitstring"] == "11"
    assert pytest.approx(row["btc_weight"], abs=1e-9) == 0.5
    assert pytest.approx(row["bnb_weight"], abs=1e-9) == 0.5


def test_binary_qaoa_runs(market_fixture):
    """QAOA may approximate, but solver must not crash; row must have success flag."""
    returns, mu, Sigma = market_fixture
    row = qbin.solve_binary(
        returns, mu, Sigma, solver="qaoa",
        budget=1, qaoa_reps=1, qaoa_shots=512, qaoa_seed=0,
    )
    # success may be True or False if qaoa not available; either way no exception
    assert "success" in row
    assert "error_message" in row
