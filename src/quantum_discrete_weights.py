"""Quantum Model B: discretized BTC weight via one-hot encoding.

Continuous w ∈ [0, 1] is replaced by a grid w_k = k * step (k = 0..K), and
binary indicator variables z_k with constraint Σ z_k = 1 select exactly one
candidate. The cost coefficient for each candidate is precomputed:

    c_k = risk_aversion * Var(w_k) − Return(w_k)

so the QUBO becomes ``min Σ c_k z_k`` s.t. ``Σ z_k = 1``.

This is purely a teaching mapping: it shows how an essentially continuous
problem can be cast as QUBO at the cost of more qubits. Classical grid search
on the same step is equivalent and faster.
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


def build_candidates(
    mu: np.ndarray,
    Sigma: np.ndarray,
    asset_returns: pd.DataFrame,
    weight_step: float,
    risk_aversion: float = 1.0,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """Pre-evaluate every candidate BTC weight on the discrete grid."""
    n = int(round(1.0 / weight_step)) + 1
    rows: list[dict] = []
    for k in range(n):
        w_btc = round(min(1.0, k * weight_step), 10)
        w = np.array([w_btc, 1.0 - w_btc])
        var = float(w @ Sigma @ w)
        ret = float(w @ mu)
        m = summarize_weights(
            w, asset_returns, mu, Sigma,
            risk_free_rate=risk_free_rate, risk_aversion=risk_aversion,
        )
        rows.append({
            "k": k,
            "btc_weight": w_btc,
            "bnb_weight": 1.0 - w_btc,
            "annual_return": m["annual_return"],
            "annual_volatility": m["annual_volatility"],
            "sharpe_ratio": m["sharpe_ratio"],
            "sortino_ratio": m["sortino_ratio"],
            "max_drawdown": m["max_drawdown"],
            "var_95": m["var_95"],
            "cvar_95": m["cvar_95"],
            "final_cumulative_return": m["final_cumulative_return"],
            "annual_variance": var,
            "objective": float(risk_aversion * var - ret),
        })
    return pd.DataFrame(rows)


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
        "selected_weight_index": -1,
    }


def _build_qp(candidates: pd.DataFrame):
    """One binary variable per candidate; minimize Σ c_k z_k subject to Σ z_k = 1."""
    if not qc.QISKIT_AVAILABLE:
        raise qc.QiskitNotAvailable(
            "qiskit_optimization not available; cannot build discrete QUBO"
        )
    qp = qc.QuadraticProgram(name="discrete_weights")
    n = len(candidates)
    var_names = [f"z_{k}" for k in range(n)]
    for v in var_names:
        qp.binary_var(v)
    linear = {var_names[k]: float(candidates.iloc[k]["objective"]) for k in range(n)}
    qp.minimize(linear=linear)
    qp.linear_constraint(linear={v: 1 for v in var_names}, sense="==", rhs=1, name="one_hot")
    return qp, var_names


def _decode(x: np.ndarray, candidates: pd.DataFrame) -> tuple[int, str, float]:
    """Pick the index of the (largest, in case of ambiguity) z_k = 1."""
    bits = [int(round(float(b))) for b in x]
    bitstring = "".join(str(b) for b in bits)
    if sum(bits) == 0:
        return -1, bitstring, float("nan")
    chosen = max(range(len(bits)), key=lambda k: (bits[k], -abs(0.5 - float(candidates.iloc[k]["btc_weight"]))))
    if bits[chosen] == 0:
        return -1, bitstring, float("nan")
    return chosen, bitstring, float(candidates.iloc[chosen]["btc_weight"])


def _samples_from_qaoa(raw: Any, candidates: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    """Extract top-k sampled bitstrings/probabilities from a QAOA/SamplingVQE result.

    qiskit_optimization stores them as ``raw.samples``; field shape varies by version
    so we defensively probe for ``.x``, ``.fval``, ``.probability``.
    """
    rows: list[dict] = []
    samples = getattr(raw, "samples", None)
    if not samples:
        return pd.DataFrame(rows)
    for s in samples:
        try:
            x = np.asarray(getattr(s, "x", []), dtype=float)
            fval = float(getattr(s, "fval", float("nan")))
            prob = float(getattr(s, "probability", float("nan")))
        except Exception:
            continue
        if x.size == 0:
            continue
        idx, bitstring, w_btc = _decode(x, candidates)
        rows.append({
            "bitstring": bitstring,
            "selected_weight_index": idx,
            "btc_weight": w_btc,
            "objective": fval,
            "probability": prob,
        })
    df = pd.DataFrame(rows).sort_values("probability", ascending=False)
    return df.head(top_k)


def _wrap_result(
    raw: Any,
    candidates: pd.DataFrame,
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
    idx, bitstring, w_btc = _decode(x, candidates)
    if idx < 0:
        return {
            "method": method,
            "solver": solver_label,
            "btc_weight": float("nan"),
            "bnb_weight": float("nan"),
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
            "error_message": "decoder picked no candidate (one-hot violated)",
            "note": note,
            "bitstring": bitstring,
            "selected_weight_index": idx,
        }

    weights = np.array([w_btc, 1.0 - w_btc])
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
        "note": note,
        "bitstring": bitstring,
        "selected_weight_index": idx,
    }


def solve_discrete(
    asset_returns: pd.DataFrame,
    mu: np.ndarray,
    Sigma: np.ndarray,
    candidates: pd.DataFrame,
    *,
    solver: str,
    risk_aversion: float = 1.0,
    risk_free_rate: float = 0.0,
    qaoa_reps: int = 1,
    qaoa_shots: int = 2048,
    qaoa_seed: int = 42,
) -> tuple[dict, pd.DataFrame]:
    """Solve the one-hot discrete-weight QUBO. Returns (result_dict, samples_df)."""
    method = "Quantum Discrete Weights"
    solver_label = solver
    start = time.perf_counter()
    samples = pd.DataFrame()

    if not qc.QISKIT_AVAILABLE:
        return (
            _empty_result(
                method, solver_label,
                "qiskit_optimization not installed; quantum discrete weights skipped",
            ),
            samples,
        )

    try:
        qp, _names = _build_qp(candidates)

        if solver == "exact":
            engine = qc.build_exact()
            note = ""
        elif solver == "qaoa":
            sampler = qc.build_sampler(qaoa_shots, qaoa_seed)
            engine = qc.build_qaoa(sampler, qaoa_reps, qaoa_seed)
            note = "QAOA approximate solver; result may differ from exact"
        elif solver == "sampling_vqe":
            sampler = qc.build_sampler(qaoa_shots, qaoa_seed)
            engine = qc.build_sampling_vqe(
                sampler, num_qubits=len(candidates), reps=qaoa_reps, seed=qaoa_seed,
            )
            note = "SamplingVQE approximate solver; result may differ from exact"
        else:
            raise ValueError(f"unknown solver: {solver}")

        if qc.MinimumEigenOptimizer is None:
            raise qc.QiskitNotAvailable("MinimumEigenOptimizer is not importable")
        optimizer = qc.MinimumEigenOptimizer(engine)
        raw = optimizer.solve(qp)
        elapsed = time.perf_counter() - start

        result = _wrap_result(
            raw, candidates, method=method, solver_label=solver_label,
            asset_returns=asset_returns, mu=mu, Sigma=Sigma,
            risk_free_rate=risk_free_rate, risk_aversion=risk_aversion,
            elapsed=elapsed, note=note,
        )
        if solver != "exact":
            samples = _samples_from_qaoa(raw, candidates)
            samples.insert(0, "solver", solver)
        return result, samples
    except qc.QiskitNotAvailable as e:
        elapsed = time.perf_counter() - start
        return _empty_result(method, solver_label, f"unavailable: {e}", elapsed), samples
    except Exception as e:  # noqa: BLE001
        elapsed = time.perf_counter() - start
        logger.warning("Quantum discrete %s failed: %s", solver, e)
        return _empty_result(method, solver_label, f"{type(e).__name__}: {e}", elapsed), samples
