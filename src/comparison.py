"""Unified comparison table — assembles every (classical + quantum + benchmark) result."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .utils import ensure_dir, to_jsonable

COLUMNS: tuple[str, ...] = (
    "method",
    "solver",
    "btc_weight",
    "bnb_weight",
    "annual_return",
    "annual_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "var_95",
    "cvar_95",
    "objective",
    "final_cumulative_return",
    "runtime_seconds",
    "success",
    "error_message",
    "note",
)


def assemble_summary(rows: list[dict]) -> pd.DataFrame:
    """Coerce a heterogeneous list of result dicts into the unified schema."""
    normalized: list[dict] = []
    for r in rows:
        normalized.append({c: r.get(c, "") for c in COLUMNS})
    return pd.DataFrame(normalized, columns=list(COLUMNS))


def save_summary(df: pd.DataFrame, output_dir: str | Path) -> tuple[Path, Path]:
    """Write summary as both CSV and JSON in ``output_dir``."""
    out = ensure_dir(output_dir)
    csv_path = out / "comparison_summary.csv"
    json_path = out / "comparison_summary.json"
    df.to_csv(csv_path, index=False)
    payload = [{k: to_jsonable(v) for k, v in row.items()} for row in df.to_dict(orient="records")]
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    return csv_path, json_path


def best_classical_row(df: pd.DataFrame) -> pd.Series | None:
    classic = df[df["method"].str.startswith("Classic", na=False) & df["success"]]
    if classic.empty:
        return None
    return classic.loc[classic["sharpe_ratio"].astype(float).idxmax()]


def best_quantum_row(df: pd.DataFrame) -> pd.Series | None:
    qm = df[df["method"].str.startswith("Quantum", na=False) & df["success"]]
    if qm.empty:
        return None
    # rank by Sharpe where finite, otherwise by lowest objective
    qm_finite = qm[qm["sharpe_ratio"].apply(lambda x: pd.notna(x) and x != float("inf"))]
    if not qm_finite.empty:
        return qm_finite.loc[qm_finite["sharpe_ratio"].astype(float).idxmax()]
    return qm.loc[qm["objective"].astype(float).idxmin()]
