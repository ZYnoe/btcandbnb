"""Risk and performance metrics. All pure, type-hinted, no I/O."""

from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
import pandas as pd

ArrayLike = pd.Series | np.ndarray | Iterable[float]


def _as_array(x: ArrayLike) -> np.ndarray:
    if isinstance(x, pd.Series):
        return x.to_numpy(dtype=float)
    return np.asarray(list(x), dtype=float) if not isinstance(x, np.ndarray) else x.astype(float)


def cumulative_returns(daily_returns: ArrayLike) -> pd.Series:
    """Convert a series of simple daily returns into a cumulative-return curve.

    cum[i] = (1 + r_0)(1 + r_1)...(1 + r_i) - 1
    """
    arr = _as_array(daily_returns)
    if arr.size == 0:
        return pd.Series(dtype=float)
    curve = np.cumprod(1.0 + arr) - 1.0
    if isinstance(daily_returns, pd.Series):
        return pd.Series(curve, index=daily_returns.index)
    return pd.Series(curve)


def portfolio_returns(weights: ArrayLike, asset_returns: pd.DataFrame) -> pd.Series:
    """Daily returns of a fixed-weight portfolio."""
    w = _as_array(weights)
    if w.shape[0] != asset_returns.shape[1]:
        raise ValueError(
            f"weights length {w.shape[0]} != number of assets {asset_returns.shape[1]}"
        )
    if not np.isclose(w.sum(), 1.0, atol=1e-6):
        warnings.warn(f"weights sum to {w.sum():.6f}, not 1.0", stacklevel=2)
    return asset_returns.dot(w)


def annualized_return(daily_returns: ArrayLike, factor: int = 365) -> float:
    """Mean daily return * factor."""
    arr = _as_array(daily_returns)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr) * factor)


def annualized_volatility(daily_returns: ArrayLike, factor: int = 365) -> float:
    """Std of daily returns * sqrt(factor)."""
    arr = _as_array(daily_returns)
    if arr.size < 2:
        return float("nan")
    return float(np.std(arr, ddof=1) * np.sqrt(factor))


def sharpe_ratio(
    daily_returns: ArrayLike, risk_free_rate: float = 0.0, factor: int = 365
) -> float:
    """(annual return - rf) / annual vol. NaN when vol is 0."""
    vol = annualized_volatility(daily_returns, factor)
    if not np.isfinite(vol) or vol == 0.0:
        return float("nan")
    return (annualized_return(daily_returns, factor) - risk_free_rate) / vol


def sortino_ratio(
    daily_returns: ArrayLike, risk_free_rate: float = 0.0, factor: int = 365
) -> float:
    """Like Sharpe but uses downside deviation (returns below 0)."""
    arr = _as_array(daily_returns)
    if arr.size < 2:
        return float("nan")
    downside = arr[arr < 0]
    if downside.size == 0:
        return float("nan")
    downside_vol = float(np.std(downside, ddof=1) * np.sqrt(factor))
    if not np.isfinite(downside_vol) or downside_vol == 0.0:
        return float("nan")
    ann_ret = annualized_return(arr, factor)
    return (ann_ret - risk_free_rate) / downside_vol


def max_drawdown(daily_returns: ArrayLike) -> float:
    """Worst peak-to-trough drop on the cumulative wealth curve. Returns a non-positive number."""
    arr = _as_array(daily_returns)
    if arr.size == 0:
        return float("nan")
    wealth = np.cumprod(1.0 + arr)
    peak = np.maximum.accumulate(wealth)
    drawdown = wealth / peak - 1.0
    return float(drawdown.min())


def value_at_risk(daily_returns: ArrayLike, level: float = 0.95) -> float:
    """Historical VaR. Returns a non-negative number representing the loss magnitude.

    e.g. 0.04 means 4% worst-case daily loss at the (1-level) tail.
    """
    arr = _as_array(daily_returns)
    if arr.size == 0:
        return float("nan")
    quantile = float(np.quantile(arr, 1.0 - level))
    return float(-quantile) if quantile < 0 else 0.0


def conditional_value_at_risk(daily_returns: ArrayLike, level: float = 0.95) -> float:
    """Mean loss in the worst (1-level) tail. Non-negative magnitude."""
    arr = _as_array(daily_returns)
    if arr.size == 0:
        return float("nan")
    cutoff = np.quantile(arr, 1.0 - level)
    tail = arr[arr <= cutoff]
    if tail.size == 0:
        return float("nan")
    mean_tail = float(np.mean(tail))
    return float(-mean_tail) if mean_tail < 0 else 0.0


def portfolio_objective(
    weights: ArrayLike,
    asset_returns: pd.DataFrame,
    mu: np.ndarray,
    Sigma: np.ndarray,
    mode: str,
    risk_aversion: float = 1.0,
    risk_free_rate: float = 0.0,
    factor: int = 365,
) -> float:
    """Compute the objective value to minimize for a given mode.

    All modes return a *minimization* target — Sharpe-like maxes are negated internally.
    """
    w = _as_array(weights)
    daily = portfolio_returns(w, asset_returns)
    ann_ret = float(w @ mu)
    ann_var = float(w @ Sigma @ w)
    ann_vol = float(np.sqrt(max(ann_var, 0.0)))
    if mode == "maximize_sharpe":
        if ann_vol == 0.0:
            return float("inf")
        return -((ann_ret - risk_free_rate) / ann_vol)
    if mode == "minimize_volatility":
        return ann_vol
    if mode == "maximize_return":
        return -ann_ret
    if mode == "maximize_return_minus_risk":
        return -(ann_ret - risk_aversion * ann_var)
    if mode == "minimize_cvar":
        return conditional_value_at_risk(daily)
    if mode == "constrained_sharpe":
        if ann_vol == 0.0:
            return float("inf")
        return -((ann_ret - risk_free_rate) / ann_vol)
    raise ValueError(f"unknown objective mode: {mode}")


def summarize_weights(
    weights: ArrayLike,
    asset_returns: pd.DataFrame,
    mu: np.ndarray,
    Sigma: np.ndarray,
    risk_free_rate: float = 0.0,
    risk_aversion: float = 1.0,
    factor: int = 365,
) -> dict:
    """All metrics for one weight vector. Used by every optimizer + benchmark + quantum result."""
    w = _as_array(weights)
    daily = portfolio_returns(w, asset_returns)
    ann_ret = annualized_return(daily, factor)
    ann_vol = annualized_volatility(daily, factor)
    sharpe = sharpe_ratio(daily, risk_free_rate, factor)
    sortino = sortino_ratio(daily, risk_free_rate, factor)
    mdd = max_drawdown(daily)
    var = value_at_risk(daily)
    cvar = conditional_value_at_risk(daily)
    cum = cumulative_returns(daily)
    final_cum = float(cum.iloc[-1]) if len(cum) else float("nan")
    obj_mv = float(risk_aversion * (w @ Sigma @ w) - (w @ mu))
    return {
        "annual_return": ann_ret,
        "annual_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": mdd,
        "var_95": var,
        "cvar_95": cvar,
        "final_cumulative_return": final_cum,
        "objective_mv": obj_mv,
    }
