"""Plotting layer — 9 PNG charts. Each plot is independently fault-tolerant."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; safe in CLI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import MarketData
from .metrics import cumulative_returns, portfolio_returns
from .utils import ensure_dir

logger = logging.getLogger("portfolio_optimizer")


def _safe(name: str, output_dir: Path, fn: Callable[[Path], None]) -> str | None:
    """Wrap a single-plot call. Failures are logged, not raised."""
    path = output_dir / name
    try:
        fn(path)
        plt.close("all")
        return str(path)
    except Exception as e:  # noqa: BLE001
        plt.close("all")
        logger.warning("Plot %s failed: %s", name, e)
        return None


def plot_prices(market: MarketData, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for col in market.prices.columns:
        ax.plot(market.prices.index, market.prices[col], label=col)
    ax.set_title("BTC vs BNB historical prices")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)


def plot_cumulative_returns(market: MarketData, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for col in market.returns.columns:
        cum = cumulative_returns(market.returns[col])
        ax.plot(cum.index, cum.values, label=col)
    ax.set_title("Cumulative returns (per asset)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return")
    ax.axhline(0, color="black", lw=0.5)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)


def plot_return_vol_scatter(summary: pd.DataFrame, grid: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    if not grid.empty:
        ax.scatter(
            grid["annual_volatility"], grid["annual_return"],
            c=grid["btc_weight"], cmap="viridis", s=20, alpha=0.5, label="grid (BTC weight color)",
        )
    success = summary[summary["success"]].copy()
    for _, row in success.iterrows():
        ax.scatter(
            row["annual_volatility"], row["annual_return"],
            marker="*" if str(row["method"]).startswith("Quantum") else "o",
            s=140, edgecolor="black",
            label=f"{row['method']} ({row['solver']})",
        )
        ax.annotate(
            f"BTC={row['btc_weight']:.2f}",
            (row["annual_volatility"], row["annual_return"]),
            textcoords="offset points", xytext=(6, 4), fontsize=8,
        )
    ax.set_title("Risk vs return (grid + optimizers + benchmarks)")
    ax.set_xlabel("Annualized volatility")
    ax.set_ylabel("Annualized return")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=140)


def plot_sharpe_vs_btc(grid: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(grid["btc_weight"], grid["sharpe_ratio"], marker="o", ms=3)
    best_idx = grid["sharpe_ratio"].idxmax()
    ax.axvline(grid.loc[best_idx, "btc_weight"], color="red", linestyle="--", label="grid optimum")
    ax.set_title("Sharpe ratio vs BTC weight (grid)")
    ax.set_xlabel("BTC weight")
    ax.set_ylabel("Sharpe ratio")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)


def plot_drawdown_vs_btc(grid: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(grid["btc_weight"], grid["max_drawdown"], marker="o", ms=3, color="darkred")
    ax.set_title("Max drawdown vs BTC weight (grid)")
    ax.set_xlabel("BTC weight")
    ax.set_ylabel("Max drawdown (negative)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)


def plot_sharpe_bar(summary: pd.DataFrame, path: Path) -> None:
    success = summary[summary["success"] & summary["sharpe_ratio"].notna()].copy()
    if success.empty:
        raise RuntimeError("nothing to plot for Sharpe bar — no successful runs")
    success["label"] = success["method"] + "\n" + success["solver"].astype(str)
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ["tab:blue" if str(m).startswith("Classic") else
              "tab:orange" if str(m).startswith("Quantum") else "tab:gray"
              for m in success["method"]]
    ax.bar(success["label"], success["sharpe_ratio"].astype(float), color=colors)
    ax.set_title("Sharpe ratio by method (classical = blue, quantum = orange, benchmark = gray)")
    ax.set_ylabel("Sharpe ratio")
    ax.tick_params(axis="x", rotation=35)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")
        tick.set_fontsize(8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)


def plot_quantum_samples(samples: pd.DataFrame, path: Path) -> None:
    if samples.empty:
        raise RuntimeError("no QAOA / SamplingVQE samples to plot")
    fig, ax = plt.subplots(figsize=(10, 5))
    df = samples.copy()
    if "btc_weight" in df.columns:
        labels = df.apply(
            lambda r: f"BTC={r['btc_weight']:.2f}" if pd.notna(r.get("btc_weight"))
            else r.get("bitstring", ""),
            axis=1,
        )
    else:
        labels = df.get("bitstring", pd.Series([""] * len(df)))
    ax.bar(range(len(df)), df["probability"].astype(float))
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_title("QAOA / SamplingVQE top sampled solutions (probability)")
    ax.set_ylabel("Probability")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)


def plot_objective_landscape(
    candidates: pd.DataFrame,
    classical_optimum_btc: float | None,
    qaoa_optimum_btc: float | None,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(candidates["btc_weight"], candidates["objective"], marker="o", ms=4, label="objective")
    if classical_optimum_btc is not None and np.isfinite(classical_optimum_btc):
        ax.axvline(classical_optimum_btc, color="green", linestyle="--", label="classical exact")
    if qaoa_optimum_btc is not None and np.isfinite(qaoa_optimum_btc):
        ax.axvline(qaoa_optimum_btc, color="red", linestyle=":", label="QAOA solution")
    ax.set_title("Discrete weight objective landscape")
    ax.set_xlabel("BTC weight")
    ax.set_ylabel("Objective (lower is better)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)


def plot_optimized_cumulative(
    market: MarketData,
    selections: list[tuple[str, np.ndarray]],
    path: Path,
) -> None:
    if not selections:
        raise RuntimeError("no selections to plot for cumulative-return comparison")
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, w in selections:
        try:
            ret = portfolio_returns(np.asarray(w, dtype=float), market.returns)
            cum = cumulative_returns(ret)
            ax.plot(cum.index, cum.values, label=label)
        except Exception as e:  # noqa: BLE001
            logger.warning("Skipping %s in cumulative plot: %s", label, e)
    ax.set_title("Cumulative returns of optimized portfolios vs benchmarks")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)


def make_all_plots(
    *,
    market: MarketData,
    grid: pd.DataFrame,
    summary: pd.DataFrame,
    quantum_samples: pd.DataFrame,
    candidates: pd.DataFrame | None,
    classical_grid_btc: float | None,
    quantum_qaoa_btc: float | None,
    optimized_selections: list[tuple[str, np.ndarray]],
    output_dir: str | Path,
) -> list[str]:
    """Render all 9 plots; return list of saved paths (None entries dropped)."""
    out = ensure_dir(output_dir)
    saved: list[str | None] = []
    saved.append(_safe("01_prices.png", out, lambda p: plot_prices(market, p)))
    saved.append(_safe("02_cumulative_returns.png", out, lambda p: plot_cumulative_returns(market, p)))
    saved.append(_safe(
        "03_return_vol_scatter.png", out,
        lambda p: plot_return_vol_scatter(summary, grid, p),
    ))
    saved.append(_safe("04_sharpe_vs_btc.png", out, lambda p: plot_sharpe_vs_btc(grid, p)))
    saved.append(_safe("05_drawdown_vs_btc.png", out, lambda p: plot_drawdown_vs_btc(grid, p)))
    saved.append(_safe("06_sharpe_bar.png", out, lambda p: plot_sharpe_bar(summary, p)))
    saved.append(_safe("07_quantum_samples.png", out, lambda p: plot_quantum_samples(quantum_samples, p)))
    if candidates is not None and not candidates.empty:
        saved.append(_safe(
            "08_objective_landscape.png", out,
            lambda p: plot_objective_landscape(candidates, classical_grid_btc, quantum_qaoa_btc, p),
        ))
    else:
        saved.append(None)
    saved.append(_safe(
        "09_optimized_cumulative.png", out,
        lambda p: plot_optimized_cumulative(market, optimized_selections, p),
    ))
    return [s for s in saved if s]
