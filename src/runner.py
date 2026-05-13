"""Top-level orchestration: download → classical → quantum → save → plot → print.

Results are written **incrementally**: every time a new row is produced
(classical, benchmark, or any quantum solver) we re-write
``comparison_summary.csv`` and ``comparison_summary.json`` so a SLURM
time-limit kill never loses partial progress. ``quantum_samples.csv`` and
``quantum_discrete_weight_results.csv`` are also written as soon as their
data exists, not deferred to the end.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import _qiskit_compat as qc
from .comparison import (
    assemble_summary,
    best_classical_row,
    best_quantum_row,
    save_summary,
)
from .config import Config
from .data import compute_market_data, MarketData
from .optimizer import evaluate_benchmarks, grid_search, mean_variance_optimize
from .parallel import (
    compute_workers,
    print_parallelism_report,
    quantum_worker,
    run_quantum_parallel,
)
from .plots import make_all_plots
from .quantum_discrete_weights import build_candidates
from .utils import ensure_dir, set_random_seed, setup_logger, to_jsonable

RISK_DISCLAIMER = (
    "RISK NOTICE — please read carefully:\n"
    "  1. Past returns do NOT predict future results.\n"
    "  2. Crypto assets are extremely volatile and can lose value rapidly.\n"
    "  3. Quantum algorithms here only solve the QUBO under the given mu/Sigma; "
    "they do not forecast prices.\n"
    "  4. This tool is for research/education only and is NOT investment advice."
)

QUANTUM_SAMPLES_COLUMNS = (
    "solver", "bitstring", "selected_weight_index", "btc_weight", "objective", "probability",
)


def _quantum_solvers_for(config: Config) -> list[str]:
    if config.quantum_solver == "all":
        return ["exact", "qaoa", "sampling_vqe"]
    return [config.quantum_solver]


def _flush_summary(rows: list[dict], output_dir: Path) -> None:
    """Re-write comparison_summary.csv + .json from current rows. Called after each new row."""
    if not rows:
        return
    summary = assemble_summary(rows)
    save_summary(summary, output_dir)


def _flush_samples(samples_dfs: list[pd.DataFrame], output_dir: Path) -> None:
    """Re-write quantum_samples.csv from accumulated sample frames."""
    path = output_dir / "quantum_samples.csv"
    if samples_dfs:
        pd.concat(samples_dfs, ignore_index=True).to_csv(path, index=False)
    elif not path.exists():
        pd.DataFrame(columns=list(QUANTUM_SAMPLES_COLUMNS)).to_csv(path, index=False)


def _print_summary(market: MarketData, summary: pd.DataFrame, output_dir: Path, log: logging.Logger) -> None:
    sep = "=" * 72
    print()
    print(sep)
    print("PORTFOLIO OPTIMIZATION — RESULT SUMMARY")
    print(sep)
    print(f"Data window : {market.returns.index.min().date()} → {market.returns.index.max().date()}")
    print(f"Tickers     : {', '.join(market.tickers)}")
    print(f"Annualization factor: {market.annualization_factor}")
    print(f"Rows used   : {len(market.returns)}")
    print()
    print("Annualized stats per asset:")
    for t, mu_v, vol_v in zip(market.tickers, market.mu, market.vol):
        print(f"  {t}: mu = {mu_v:+.4f}    vol = {vol_v:.4f}")
    print(f"  correlation BTC↔BNB: {market.corr[0, 1]:+.4f}")
    print()

    classic = best_classical_row(summary)
    quantum = best_quantum_row(summary)

    if classic is not None:
        print(f"Classical optimum : {classic['method']} ({classic['solver']})")
        print(f"  weights          : BTC {classic['btc_weight']:.4f}  /  BNB {classic['bnb_weight']:.4f}")
        print(f"  Sharpe / Vol     : {classic['sharpe_ratio']:.4f}  /  {classic['annual_volatility']:.4f}")
        print(f"  Annual return    : {classic['annual_return']:+.4f}")
        print(f"  Max drawdown     : {classic['max_drawdown']:+.4f}")
    else:
        print("Classical optimum : (none — no successful classical row)")
    print()

    if quantum is not None:
        print(f"Quantum optimum   : {quantum['method']} ({quantum['solver']})")
        print(f"  weights          : BTC {quantum['btc_weight']}  /  BNB {quantum['bnb_weight']}")
        print(f"  Sharpe / Vol     : {quantum['sharpe_ratio']}  /  {quantum['annual_volatility']}")
        print(f"  note             : {quantum.get('note', '')}")
    else:
        print("Quantum optimum   : (no successful quantum row — see comparison_summary.csv)")
    print()

    benchmarks = summary[summary["method"].str.startswith("Benchmark", na=False)]
    if not benchmarks.empty:
        print("Benchmarks:")
        for _, row in benchmarks.iterrows():
            print(
                f"  {row['method']:<22} BTC={row['btc_weight']:.2f} "
                f"Sharpe={row['sharpe_ratio']:.4f}  "
                f"AnnRet={row['annual_return']:+.4f}  MaxDD={row['max_drawdown']:+.4f}"
            )
    print()
    print(f"Outputs saved to: {output_dir.resolve()}")
    print()
    print(RISK_DISCLAIMER)
    print(sep)


def run(config: Config) -> int:
    """End-to-end. Returns shell-style exit code.

    All output files are flushed incrementally so a job kill (SLURM time
    limit, OOM, Ctrl-C) at any point still leaves consistent partial state
    on disk: every completed row is already in ``comparison_summary.csv``.
    """
    config.validate()
    log = setup_logger(config.verbose)
    set_random_seed(config.qaoa_seed)
    output_dir = ensure_dir(config.output_path)

    # Resolve the worker count and emit a diagnostic banner *before* any work,
    # so the SLURM .out file always shows what parallelism we actually got.
    workers = compute_workers(config.workers)
    print_parallelism_report(config.workers, workers)

    log.info("Downloading market data for %s ...", ", ".join(config.tickers))
    try:
        market = compute_market_data(
            tuple(config.tickers),
            start=config.start, end=config.end,
            frequency=config.frequency,
            annualization_factor=config.annualization_factor,
        )
    except Exception as e:  # noqa: BLE001
        log.error("Data download failed: %s", e)
        return 2

    # returns.csv is always dumped — the Julia subproject (julia/run.jl) reads
    # it to share μ/Σ-aligned daily returns with the Python pipeline
    # (Invariant 1). prices.csv stays opt-in to avoid clutter.
    market.returns.to_csv(output_dir / "returns.csv")
    if config.save_intermediate:
        market.prices.to_csv(output_dir / "prices.csv")

    with open(output_dir / "basic_stats.json", "w", encoding="utf-8") as fh:
        json.dump(to_jsonable(market.basic_stats()), fh, indent=2, ensure_ascii=False)

    rows: list[dict] = []
    samples_dfs: list[pd.DataFrame] = []
    candidates_df = pd.DataFrame()
    landmarks: dict[str, float | None] = {"qaoa_discrete_btc": None}

    # Always create an empty placeholder so downstream tooling sees a stable schema
    _flush_samples(samples_dfs, output_dir)

    # ---- 1. Classical grid (cheap; flush so we have something even if next step crashes)
    log.info("Running classical grid search...")
    grid, classic_grid_row = grid_search(
        market.returns, market.mu, market.Sigma,
        step=config.step,
        objective=config.objective,
        risk_free_rate=config.risk_free_rate,
        risk_aversion=config.risk_aversion,
        max_drawdown=config.max_drawdown,
    )
    grid.to_csv(output_dir / "grid_search_results.csv", index=False)
    rows.append(classic_grid_row)
    _flush_summary(rows, output_dir)

    # ---- 2. Mean-Variance
    log.info("Running mean-variance optimization...")
    mv_row = mean_variance_optimize(
        market.returns, market.mu, market.Sigma,
        objective=config.objective,
        risk_free_rate=config.risk_free_rate,
        risk_aversion=config.risk_aversion,
    )
    rows.append(mv_row)
    _flush_summary(rows, output_dir)

    # ---- 3. Benchmarks (cheap; useful baselines even if quantum dies)
    log.info("Evaluating benchmarks...")
    benchmark_rows = evaluate_benchmarks(
        market.returns, market.mu, market.Sigma,
        risk_free_rate=config.risk_free_rate,
        risk_aversion=config.risk_aversion,
    )
    rows.extend(benchmark_rows)
    _flush_summary(rows, output_dir)

    # ---- 4. Quantum (slow; the only place worth parallelizing in this codebase)
    if config.use_quantum:
        if not qc.QISKIT_AVAILABLE:
            log.warning(
                "Qiskit is not importable; skipping quantum runs. Install qiskit + "
                "qiskit-optimization + qiskit-algorithms (and qiskit-aer for QAOA "
                "shots) to enable."
            )
        else:
            log.info("Qiskit availability: %s", qc.availability_summary())
            do_binary = config.quantum_mode in ("binary", "both")
            do_discrete = config.quantum_mode in ("discrete", "both")
            solvers = _quantum_solvers_for(config)

            # If discrete is requested, pre-compute candidates ONCE in the
            # parent. Workers receive a copy via the pickled payload —
            # cheaper than re-evaluating per worker.
            if do_discrete:
                candidates_df = build_candidates(
                    market.mu, market.Sigma, market.returns,
                    weight_step=config.quantum_weight_step,
                    risk_aversion=config.risk_aversion,
                    risk_free_rate=config.risk_free_rate,
                )
                candidates_df.to_csv(
                    output_dir / "quantum_discrete_weight_results.csv", index=False,
                )
                log.info(
                    "Built %d candidate weights with step=%.4f for discrete-weight QUBO.",
                    len(candidates_df), config.quantum_weight_step,
                )

            # Build the task list. Each entry is one solver call we want to run.
            payloads: list[dict] = []
            if do_binary:
                for solver in solvers:
                    if solver == "sampling_vqe" and not qc.SAMPLING_VQE_AVAILABLE:
                        log.info("Skipping binary sampling_vqe — not available in installed Qiskit.")
                        continue
                    payloads.append({
                        "kind": "binary",
                        "solver": solver,
                        "asset_returns": market.returns,
                        "mu": market.mu,
                        "Sigma": market.Sigma,
                        "risk_aversion": config.risk_aversion,
                        "risk_free_rate": config.risk_free_rate,
                        "budget": config.binary_budget,
                        "no_budget_constraint": config.no_budget_constraint,
                        "qaoa_reps": config.qaoa_reps,
                        "qaoa_shots": config.qaoa_shots,
                        "qaoa_seed": config.qaoa_seed,
                    })
            if do_discrete:
                for solver in solvers:
                    if solver == "sampling_vqe" and not qc.SAMPLING_VQE_AVAILABLE:
                        log.info("Skipping discrete sampling_vqe — not available in installed Qiskit.")
                        continue
                    payloads.append({
                        "kind": "discrete",
                        "solver": solver,
                        "asset_returns": market.returns,
                        "mu": market.mu,
                        "Sigma": market.Sigma,
                        "candidates": candidates_df,
                        "risk_aversion": config.risk_aversion,
                        "risk_free_rate": config.risk_free_rate,
                        "qaoa_reps": config.qaoa_reps,
                        "qaoa_shots": config.qaoa_shots,
                        "qaoa_seed": config.qaoa_seed,
                    })

            print(f"[parallel] quantum task plan: {len(payloads)} task(s) "
                  f"({'binary, ' if do_binary else ''}{'discrete' if do_discrete else ''})", flush=True)

            # Side-effect callback: incrementally write summary + samples after
            # each result, regardless of which worker it came from. Runs in the
            # parent process, so file writes are naturally serialized.
            def _on_result(payload: dict, result: dict) -> None:
                row = result["row"]
                rows.append(row)
                _flush_summary(rows, output_dir)
                samples = result.get("samples")
                if samples is not None and not samples.empty:
                    samples_dfs.append(samples)
                    _flush_samples(samples_dfs, output_dir)
                if (
                    payload["kind"] == "discrete"
                    and payload["solver"] == "qaoa"
                    and row.get("success")
                    and pd.notna(row.get("btc_weight"))
                ):
                    landmarks["qaoa_discrete_btc"] = float(row["btc_weight"])

            def _on_error(payload: dict, exc: Exception) -> None:
                # Don't swallow — surface to the log AND record a failed row so
                # the comparison_summary always carries every requested task.
                msg = f"{type(exc).__name__}: {exc}"
                log.error(
                    "[parallel] worker for %s/%s raised: %s",
                    payload["kind"], payload["solver"], msg,
                )
                method = (
                    "Quantum Binary Selection" if payload["kind"] == "binary"
                    else "Quantum Discrete Weights"
                )
                rows.append({
                    "method": method,
                    "solver": payload["solver"],
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
                    "runtime_seconds": float("nan"),
                    "success": False,
                    "error_message": f"worker raised {msg}",
                    "note": "",
                })
                _flush_summary(rows, output_dir)

            # Dispatch: parallel pool when it's worth it, in-process loop otherwise.
            if not payloads:
                log.info("[parallel] no quantum tasks to run")
            elif workers <= 1 or len(payloads) == 1:
                reason = (
                    "workers=1 (sequential)" if workers <= 1
                    else "only 1 task, no point spawning a pool"
                )
                print(f"[parallel] running {len(payloads)} task(s) in-process: {reason}", flush=True)
                for payload in payloads:
                    try:
                        result = quantum_worker(payload)
                    except Exception as e:  # noqa: BLE001
                        _on_error(payload, e)
                    else:
                        _on_result(payload, result)
            else:
                run_quantum_parallel(
                    payloads=payloads,
                    workers=workers,
                    on_result=_on_result,
                    on_error=_on_error,
                )

    # ---- 5. Plots (last; failure here doesn't lose any data)
    samples_df = pd.concat(samples_dfs, ignore_index=True) if samples_dfs else pd.DataFrame()
    summary = assemble_summary(rows)
    classic_best = best_classical_row(summary)

    optimized_selections: list[tuple[str, np.ndarray]] = []
    if classic_best is not None and pd.notna(classic_best["btc_weight"]):
        optimized_selections.append((
            f"Classical best ({classic_best['method']})",
            np.array([classic_best["btc_weight"], classic_best["bnb_weight"]], dtype=float),
        ))
    quantum_rows = [r for r in rows if str(r.get("method", "")).startswith("Quantum")]
    q_disc = [
        r for r in quantum_rows
        if r.get("success")
        and r.get("method", "").startswith("Quantum Discrete")
        and pd.notna(r.get("btc_weight"))
    ]
    if q_disc:
        best_q = max(q_disc, key=lambda r: r.get("sharpe_ratio") or float("-inf"))
        optimized_selections.append((
            f"Quantum discrete best ({best_q['solver']})",
            np.array([best_q["btc_weight"], best_q["bnb_weight"]], dtype=float),
        ))
    optimized_selections.extend([
        ("Benchmark 100% BTC", np.array([1.0, 0.0])),
        ("Benchmark 100% BNB", np.array([0.0, 1.0])),
        ("Benchmark 50/50", np.array([0.5, 0.5])),
    ])

    if not config.no_plots:
        log.info("Rendering plots...")
        make_all_plots(
            market=market,
            grid=grid,
            summary=summary,
            quantum_samples=samples_df,
            candidates=candidates_df if not candidates_df.empty else None,
            classical_grid_btc=float(classic_grid_row["btc_weight"]),
            quantum_qaoa_btc=landmarks.get("qaoa_discrete_btc"),
            optimized_selections=optimized_selections,
            output_dir=output_dir,
        )

    _print_summary(market, summary, Path(output_dir), log)
    return 0
