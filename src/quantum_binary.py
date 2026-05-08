"""Quantum Model A: binary asset selection.

Variables x_BTC, x_BNB ∈ {0, 1}; objective ``min λ x^T Σ x − μ^T x``
with optional cardinality constraint ``x_BTC + x_BNB == budget``.

Mapping into portfolio weights:
- budget=1: pick one ticker → that weight = 1.
- budget=2 (or no constraint, both selected): equal-weight 0.5 / 0.5.
- no_budget_constraint and zero selected: invalid_portfolio (weight = 0/0).
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from . import _qiskit_compat as qc
from .metrics import summarize_weights

logger = logging.getLogger("portfolio_optimizer")

ASSET_LABELS = ("BTC", "BNB")
VAR_NAMES = ("x_BTC", "x_BNB")


def _empty_result(method: str, solver: str, error: str, runtime: float = 0.0) -> dict:
    return {
        "method": method,
        "solver": solver,
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
        "runtime_seconds": float(runtime),
        "success": False,
        "error_message": error,
        "note": "",
        "bitstring": "",
        "selected_assets": "",
    }


def _build_qp(
    mu: np.ndarray,
    Sigma: np.ndarray,
    risk_aversion: float,
    budget: int,
    no_budget_constraint: bool,
):
    """Construct a 2-variable binary QuadraticProgram. Returns the QP object."""
    if not qc.QISKIT_AVAILABLE:
        raise qc.QiskitNotAvailable(
            "qiskit_optimization not available; cannot build QuadraticProgram"
        )
    qp = qc.QuadraticProgram(name="binary_selection")
    for v in VAR_NAMES:
        qp.binary_var(v)
    linear = {VAR_NAMES[i]: float(-mu[i]) for i in range(2)}
    quadratic = {
        (VAR_NAMES[0], VAR_NAMES[0]): float(risk_aversion * Sigma[0, 0]),
        (VAR_NAMES[1], VAR_NAMES[1]): float(risk_aversion * Sigma[1, 1]),
        (VAR_NAMES[0], VAR_NAMES[1]): float(2.0 * risk_aversion * Sigma[0, 1]),
    }
    qp.minimize(linear=linear, quadratic=quadratic)
    if not no_budget_constraint:
        qp.linear_constraint(
            linear={VAR_NAMES[0]: 1, VAR_NAMES[1]: 1},
            sense="==",
            rhs=int(budget),
            name="budget",
        )
    return qp


def _decode(x: np.ndarray) -> tuple[np.ndarray, str, str, str]:
    """Map a 2-bit selection vector to (weights, bitstring, selected_assets, note)."""
    selected = [ASSET_LABELS[i] for i in range(2) if int(round(float(x[i]))) == 1]
    bitstring = "".join(str(int(round(float(b)))) for b in x)
    if not selected:
        return np.zeros(2), bitstring, "(none)", "invalid_portfolio: no asset selected"
    if len(selected) == 1:
        weights = np.zeros(2)
        weights[ASSET_LABELS.index(selected[0])] = 1.0
        return weights, bitstring, selected[0], ""
    return (
        np.array([0.5, 0.5]),
        bitstring,
        "+".join(selected),
        "both selected; weights split equally (0.5/0.5)",
    )


def _solve(qp: Any, solver: Any, sampler: Any | None = None):
    """Run a MinimumEigenOptimizer over the QP and return its result."""
    if qc.MinimumEigenOptimizer is None:
        raise qc.QiskitNotAvailable("MinimumEigenOptimizer is not importable")
    optimizer = qc.MinimumEigenOptimizer(solver)
    return optimizer.solve(qp)


def _wrap_result(
    raw: Any,
    *,
    method: str,
    solver_label: str,
    asset_returns: pd.DataFrame,
    mu: np.ndarray,
    Sigma: np.ndarray,
    risk_free_rate: float,
    risk_aversion: float,
    elapsed: float,
    note: str = "",
) -> dict:
    x = np.asarray(raw.x, dtype=float)
    weights, bitstring, selected, decode_note = _decode(x)
    final_note = "; ".join(s for s in (decode_note, note) if s)

    if weights.sum() <= 0:
        # invalid; still write a row for transparency
        return {
            "method": method,
            "solver": solver_label,
            "btc_weight": 0.0,
            "bnb_weight": 0.0,
            "annual_return": float("nan"),
            "annual_volatility": float("nan"),
            "sharpe_ratio": float("nan"),
            "sortino_ratio": float("nan"),
            "max_drawdown": float("nan"),
            "var_95": float("nan"),
            "cvar_95": float("nan"),
            "objective": float(raw.fval),
            "final_cumulative_return": float("nan"),
            "runtime_seconds": float(elapsed),
            "success": True,
            "error_message": "",
            "note": final_note,
            "bitstring": bitstring,
            "selected_assets": selected,
        }

    m = summarize_weights(
        weights, asset_returns, mu, Sigma,
        risk_free_rate=risk_free_rate, risk_aversion=risk_aversion,
    )
    return {
        "method": method,
        "solver": solver_label,
        "btc_weight": float(weights[0]),
        "bnb_weight": float(weights[1]),
        "annual_return": m["annual_return"],
        "annual_volatility": m["annual_volatility"],
        "sharpe_ratio": m["sharpe_ratio"],
        "sortino_ratio": m["sortino_ratio"],
        "max_drawdown": m["max_drawdown"],
        "var_95": m["var_95"],
        "cvar_95": m["cvar_95"],
        "objective": float(raw.fval),
        "final_cumulative_return": m["final_cumulative_return"],
        "runtime_seconds": float(elapsed),
        "success": True,
        "error_message": "",
        "note": final_note,
        "bitstring": bitstring,
        "selected_assets": selected,
    }


def solve_binary(
    asset_returns: pd.DataFrame,
    mu: np.ndarray,
    Sigma: np.ndarray,
    *,
    solver: str,
    risk_aversion: float = 1.0,
    risk_free_rate: float = 0.0,
    budget: int = 1,
    no_budget_constraint: bool = False,
    qaoa_reps: int = 1,
    qaoa_shots: int = 2048,
    qaoa_seed: int = 42,
) -> dict:
    """Solve the binary-selection QUBO with one of {exact, qaoa, sampling_vqe}.

    Wraps every step in try/except — on any qiskit failure we return a row with
    ``success=False`` and ``error_message`` set, so the comparison table still has
    a deterministic shape.
    """
    method = "Quantum Binary Selection"
    solver_label = solver
    start = time.perf_counter()

    if not qc.QISKIT_AVAILABLE:
        return _empty_result(
            method, solver_label,
            "qiskit_optimization not installed; quantum binary selection skipped",
        )

    try:
        qp = _build_qp(mu, Sigma, risk_aversion, budget, no_budget_constraint)

        if solver == "exact":
            engine = qc.build_exact()
            note = ""
        elif solver == "qaoa":
            sampler = qc.build_sampler(qaoa_shots, qaoa_seed)
            engine = qc.build_qaoa(sampler, qaoa_reps, qaoa_seed)
            note = "QAOA approximate solver; result may differ from exact"
        elif solver == "sampling_vqe":
            sampler = qc.build_sampler(qaoa_shots, qaoa_seed)
            engine = qc.build_sampling_vqe(sampler, num_qubits=2, reps=qaoa_reps, seed=qaoa_seed)
            note = "SamplingVQE approximate solver; result may differ from exact"
        else:
            raise ValueError(f"unknown solver: {solver}")

        raw = _solve(qp, engine)
        elapsed = time.perf_counter() - start
        return _wrap_result(
            raw, method=method, solver_label=solver_label,
            asset_returns=asset_returns, mu=mu, Sigma=Sigma,
            risk_free_rate=risk_free_rate, risk_aversion=risk_aversion,
            elapsed=elapsed, note=note,
        )
    except qc.QiskitNotAvailable as e:
        elapsed = time.perf_counter() - start
        return _empty_result(method, solver_label, f"unavailable: {e}", elapsed)
    except Exception as e:  # noqa: BLE001
        elapsed = time.perf_counter() - start
        logger.warning("Quantum binary %s failed: %s", solver, e)
        return _empty_result(method, solver_label, f"{type(e).__name__}: {e}", elapsed)
