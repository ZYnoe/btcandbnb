"""Unit tests for src.metrics — pure-function checks, no I/O."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import metrics


@pytest.fixture
def synthetic_returns() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 400
    btc = rng.normal(0.001, 0.03, n)
    bnb = 0.6 * btc + rng.normal(0.0008, 0.025, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({"BTC-USD": btc, "BNB-USD": bnb}, index=idx)


def test_cumulative_returns_matches_compounding():
    r = pd.Series([0.1, -0.05, 0.02])
    cum = metrics.cumulative_returns(r)
    expected = (1.1 * 0.95 * 1.02) - 1
    assert pytest.approx(cum.iloc[-1], rel=1e-12) == expected


def test_portfolio_returns_dot_product(synthetic_returns):
    weights = [0.6, 0.4]
    port = metrics.portfolio_returns(weights, synthetic_returns)
    expected = 0.6 * synthetic_returns["BTC-USD"] + 0.4 * synthetic_returns["BNB-USD"]
    pd.testing.assert_series_equal(port, expected, check_names=False)


def test_weights_must_sum_to_one_warns():
    with pytest.warns(UserWarning):
        metrics.portfolio_returns(
            [0.3, 0.3],
            pd.DataFrame({"a": [0.0, 0.1], "b": [0.0, 0.1]}),
        )


def test_max_drawdown_within_minus_one_zero(synthetic_returns):
    port = metrics.portfolio_returns([0.5, 0.5], synthetic_returns)
    mdd = metrics.max_drawdown(port)
    assert -1.0 <= mdd <= 0.0


def test_var_cvar_non_negative(synthetic_returns):
    port = metrics.portfolio_returns([0.5, 0.5], synthetic_returns)
    var = metrics.value_at_risk(port)
    cvar = metrics.conditional_value_at_risk(port)
    assert var >= 0.0
    assert cvar >= var or np.isnan(cvar)  # CVaR is at least as large as VaR


def test_sharpe_zero_vol_returns_nan():
    flat = pd.Series(np.zeros(50))
    assert np.isnan(metrics.sharpe_ratio(flat))


def test_summarize_weights_keys(synthetic_returns):
    mu = synthetic_returns.mean().to_numpy() * 365
    Sigma = synthetic_returns.cov().to_numpy() * 365
    out = metrics.summarize_weights(np.array([0.5, 0.5]), synthetic_returns, mu, Sigma)
    expected = {
        "annual_return", "annual_volatility", "sharpe_ratio", "sortino_ratio",
        "max_drawdown", "var_95", "cvar_95", "final_cumulative_return", "objective_mv",
    }
    assert expected.issubset(out.keys())
