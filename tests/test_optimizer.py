"""Tests for classical optimizer module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import optimizer


@pytest.fixture
def market_fixture():
    """Synthetic data where BTC has higher mean and BNB is noisier."""
    rng = np.random.default_rng(0)
    n = 500
    btc = rng.normal(0.0015, 0.02, n)
    bnb = rng.normal(0.0005, 0.04, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    returns = pd.DataFrame({"BTC-USD": btc, "BNB-USD": bnb}, index=idx)
    mu = returns.mean().to_numpy() * 365
    Sigma = returns.cov().to_numpy() * 365
    return returns, mu, Sigma


def test_grid_search_returns_best_weights_summing_to_one(market_fixture):
    returns, mu, Sigma = market_fixture
    grid, best = optimizer.grid_search(returns, mu, Sigma, step=0.05, objective="maximize_sharpe")
    assert best["success"] is True
    assert pytest.approx(best["btc_weight"] + best["bnb_weight"], abs=1e-9) == 1.0
    assert grid["btc_weight"].tolist() == sorted(grid["btc_weight"].tolist())


def test_grid_search_minimize_volatility_picks_low_vol(market_fixture):
    returns, mu, Sigma = market_fixture
    _grid, best = optimizer.grid_search(
        returns, mu, Sigma, step=0.05, objective="minimize_volatility",
    )
    # since BTC is less noisy than BNB in fixture, optimum should lean to BTC
    assert best["btc_weight"] > 0.5


def test_mean_variance_runs_and_normalizes(market_fixture):
    returns, mu, Sigma = market_fixture
    res = optimizer.mean_variance_optimize(
        returns, mu, Sigma, objective="maximize_sharpe",
    )
    assert res["success"] is True
    assert 0.0 <= res["btc_weight"] <= 1.0
    assert pytest.approx(res["btc_weight"] + res["bnb_weight"], abs=1e-6) == 1.0


def test_mean_variance_skips_unsupported_objective(market_fixture):
    returns, mu, Sigma = market_fixture
    res = optimizer.mean_variance_optimize(returns, mu, Sigma, objective="minimize_cvar")
    assert res["success"] is False
    assert "minimize_cvar" in res["error_message"]


def test_benchmarks_have_expected_rows(market_fixture):
    returns, mu, Sigma = market_fixture
    rows = optimizer.evaluate_benchmarks(returns, mu, Sigma)
    labels = {r["method"] for r in rows}
    assert {"Benchmark 100% BTC", "Benchmark 100% BNB", "Benchmark 50/50"} == labels
    for r in rows:
        assert r["success"] is True
        assert pytest.approx(r["btc_weight"] + r["bnb_weight"], abs=1e-9) == 1.0
