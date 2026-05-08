"""Classical portfolio optimization for two assets: grid search, mean-variance, benchmarks."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .metrics import portfolio_objective, summarize_weights

logger = logging.getLogger("portfolio_optimizer")


def _select_from_grid(grid: pd.DataFrame, objective: str, max_drawdown: float) -> pd.Series:
    """Pick the best row from a grid of candidate weights given the objective name.

    constrained_sharpe filters by the max-drawdown threshold first; falls back to
    full grid (with a warning) if filter empties.
    """
    if objective == "maximize_sharpe":
        return grid.loc[grid["sharpe_ratio"].idxmax()]
    if objective == "minimize_volatility":
        return grid.loc[grid["annual_volatility"].idxmin()]
    if objective == "maximize_return":
        return grid.loc[grid["annual_return"].idxmax()]
    if objective == "minimize_cvar":
        return grid.loc[grid["cvar_95"].idxmin()]
    if objective == "constrained_sharpe":
        eligible = grid[grid["max_drawdown"] >= max_drawdown]
        if eligible.empty:
            logger.warning(
                "No grid candidates satisfy max_drawdown >= %.3f; using full grid.",
                max_drawdown,
            )
            eligible = grid
        return eligible.loc[eligible["sharpe_ratio"].idxmax()]
    raise ValueError(f"unknown objective: {objective}")


def grid_search(
    asset_returns: pd.DataFrame,
    mu: np.ndarray,
    Sigma: np.ndarray,
    step: float = 0.01,
    objective: str = "maximize_sharpe",
    risk_free_rate: float = 0.0,
    risk_aversion: float = 1.0,
    max_drawdown: float = -1.0,
) -> tuple[pd.DataFrame, dict]:
    """Brute-force scan of BTC weight in [0, 1] with the given step.

    Returns (full grid DataFrame, best result dict).
    """
    start = time.perf_counter()
    n_steps = int(round(1.0 / step)) + 1
    btc_weights = np.linspace(0.0, 1.0, n_steps)

    rows: list[dict] = []
    for w_btc in btc_weights:
        w_btc = float(round(w_btc, 10))  # avoid floating fuzz on large grids
        w = np.array([w_btc, 1.0 - w_btc])
        m = summarize_weights(
            w, asset_returns, mu, Sigma,
            risk_free_rate=risk_free_rate, risk_aversion=risk_aversion,
        )
        m["btc_weight"] = w_btc
        m["bnb_weight"] = 1.0 - w_btc
        rows.append(m)
    grid = pd.DataFrame(rows)

    best_row = _select_from_grid(grid, objective, max_drawdown)
    elapsed = time.perf_counter() - start
    best = _row_to_result(
        best_row,
        method="Classic Grid Search",
        solver=f"grid step={step} obj={objective}",
        runtime=elapsed,
    )
    logger.info("Grid search done: best BTC weight = %.4f (%s).", best["btc_weight"], objective)
    return grid, best


def _row_to_result(row: pd.Series, method: str, solver: str, runtime: float) -> dict:
    """Format a one-row pandas series into our standard result dict."""
    return {
        "method": method,
        "solver": solver,
        "btc_weight": float(row["btc_weight"]),
        "bnb_weight": float(row["bnb_weight"]),
        "annual_return": float(row["annual_return"]),
        "annual_volatility": float(row["annual_volatility"]),
        "sharpe_ratio": float(row["sharpe_ratio"]),
        "sortino_ratio": float(row["sortino_ratio"]),
        "max_drawdown": float(row["max_drawdown"]),
        "var_95": float(row["var_95"]),
        "cvar_95": float(row["cvar_95"]),
        "objective": float(row["objective_mv"]),
        "final_cumulative_return": float(row["final_cumulative_return"]),
        "runtime_seconds": float(runtime),
        "success": True,
        "error_message": "",
        "note": "",
    }


def mean_variance_optimize(
    asset_returns: pd.DataFrame,
    mu: np.ndarray,
    Sigma: np.ndarray,
    objective: str = "maximize_sharpe",
    risk_free_rate: float = 0.0,
    risk_aversion: float = 1.0,
) -> dict:
    """SLSQP-based continuous optimizer.

    Note: ``minimize_cvar`` and ``constrained_sharpe`` are routed back to a fine grid
    inside grid_search — scipy gradient methods don't play well with quantile-based
    or hard-constrained objectives over only 2 variables.
    """
    if objective in ("minimize_cvar", "constrained_sharpe"):
        return {
            "method": "Classic Mean-Variance",
            "solver": f"skipped (use grid for {objective})",
            "btc_weight": float("nan"),
            "bnb_weight": float("nan"),
            "annual_return": float("nan"),
            "annual_volatility": float("nan"),
            "sharpe_ratio": float("nan"),
            "sortino_ratio": float("nan"),
            "max_drawdown": float("nan"),
            "var_95": float("nan"),
            "cvar_95": float("nan"),
            "objective": float("nan"),
            "final_cumulative_return": float("nan"),
            "runtime_seconds": 0.0,
            "success": False,
            "error_message": f"mean_variance does not implement {objective}; see grid result",
            "note": "",
        }

    start = time.perf_counter()
    mode = objective if objective != "maximize_return_minus_risk" else "maximize_return_minus_risk"

    def obj_fn(w: np.ndarray) -> float:
        return portfolio_objective(
            w, asset_returns, mu, Sigma, mode=mode,
            risk_aversion=risk_aversion, risk_free_rate=risk_free_rate,
        )

    constraints = ({"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},)
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    x0 = np.array([0.5, 0.5])

    try:
        res = minimize(obj_fn, x0, method="SLSQP", bounds=bounds, constraints=constraints)
        elapsed = time.perf_counter() - start
        if not res.success:
            return _failed_mv(elapsed, res.message)
        w = np.clip(res.x, 0.0, 1.0)
        w = w / w.sum()  # renormalize after clipping
        m = summarize_weights(
            w, asset_returns, mu, Sigma,
            risk_free_rate=risk_free_rate, risk_aversion=risk_aversion,
        )
        return {
            "method": "Classic Mean-Variance",
            "solver": f"SLSQP obj={objective}",
            "btc_weight": float(w[0]),
            "bnb_weight": float(w[1]),
            "annual_return": m["annual_return"],
            "annual_volatility": m["annual_volatility"],
            "sharpe_ratio": m["sharpe_ratio"],
            "sortino_ratio": m["sortino_ratio"],
            "max_drawdown": m["max_drawdown"],
            "var_95": m["var_95"],
            "cvar_95": m["cvar_95"],
            "objective": m["objective_mv"],
            "final_cumulative_return": m["final_cumulative_return"],
            "runtime_seconds": float(elapsed),
            "success": True,
            "error_message": "",
            "note": "",
        }
    except Exception as e:  # noqa: BLE001
        elapsed = time.perf_counter() - start
        return _failed_mv(elapsed, str(e))


def _failed_mv(elapsed: float, message: str) -> dict:
    return {
        "method": "Classic Mean-Variance",
        "solver": "SLSQP",
        "btc_weight": float("nan"),
        "bnb_weight": float("nan"),
        "annual_return": float("nan"),
        "annual_volatility": float("nan"),
        "sharpe_ratio": float("nan"),
        "sortino_ratio": float("nan"),
        "max_drawdown": float("nan"),
        "var_95": float("nan"),
        "cvar_95": float("nan"),
        "objective": float("nan"),
        "final_cumulative_return": float("nan"),
        "runtime_seconds": float(elapsed),
        "success": False,
        "error_message": message,
        "note": "",
    }


def evaluate_benchmarks(
    asset_returns: pd.DataFrame,
    mu: np.ndarray,
    Sigma: np.ndarray,
    risk_free_rate: float = 0.0,
    risk_aversion: float = 1.0,
) -> list[dict]:
    """100% BTC, 100% BNB, 50/50 — one row each, same schema as the optimizers."""
    benchmarks: list[tuple[str, np.ndarray]] = [
        ("Benchmark 100% BTC", np.array([1.0, 0.0])),
        ("Benchmark 100% BNB", np.array([0.0, 1.0])),
        ("Benchmark 50/50", np.array([0.5, 0.5])),
    ]
    out: list[dict] = []
    for label, w in benchmarks:
        m = summarize_weights(
            w, asset_returns, mu, Sigma,
            risk_free_rate=risk_free_rate, risk_aversion=risk_aversion,
        )
        out.append({
            "method": label,
            "solver": "fixed",
            "btc_weight": float(w[0]),
            "bnb_weight": float(w[1]),
            "annual_return": m["annual_return"],
            "annual_volatility": m["annual_volatility"],
            "sharpe_ratio": m["sharpe_ratio"],
            "sortino_ratio": m["sortino_ratio"],
            "max_drawdown": m["max_drawdown"],
            "var_95": m["var_95"],
            "cvar_95": m["cvar_95"],
            "objective": m["objective_mv"],
            "final_cumulative_return": m["final_cumulative_return"],
            "runtime_seconds": 0.0,
            "success": True,
            "error_message": "",
            "note": "",
        })
    return out
