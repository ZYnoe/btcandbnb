"""CLI entry point. Thin: argparse → Config → src.runner.run."""

from __future__ import annotations

import argparse
import sys

from src.config import (
    CLASSIC_OBJECTIVES,
    Config,
    QUANTUM_MODES,
    QUANTUM_SOLVERS,
    TICKERS_DEFAULT,
    default_end,
    default_start,
)
from src.runner import run


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="portfolio-optimizer",
        description=(
            "Compare classical vs quantum (Qiskit) portfolio optimization for BTC/BNB.\n"
            "This tool is for research and education only and is NOT investment advice."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # data
    p.add_argument("--start", default=default_start(), help="ISO date, e.g. 2023-01-01")
    p.add_argument("--end", default=default_end(), help="ISO date, e.g. 2025-01-01")
    p.add_argument("--frequency", default="1d", help="yfinance interval (1d, 1wk, ...)")
    p.add_argument(
        "--annualization-factor", type=int, default=365,
        help="365 for crypto (24/7), 252 for traditional equities",
    )
    p.add_argument("--output-dir", default="outputs", help="Directory for CSV/JSON/PNG outputs")
    p.add_argument(
        "--tickers", nargs=2, metavar=("BTC", "BNB"), default=list(TICKERS_DEFAULT),
        help="Override tickers (must be exactly 2)",
    )

    # classical
    p.add_argument("--step", type=float, default=0.01, help="Grid step over BTC weight (0,0.5]")
    p.add_argument("--objective", choices=CLASSIC_OBJECTIVES, default="maximize_sharpe")
    p.add_argument("--risk-free-rate", type=float, default=0.0)
    p.add_argument("--risk-aversion", type=float, default=1.0)
    p.add_argument(
        "--max-drawdown", type=float, default=-0.5,
        help="Threshold (negative) for constrained_sharpe; e.g. -0.5 = -50%%",
    )

    # quantum
    p.add_argument("--use-quantum", action="store_true", help="Enable quantum solvers")
    p.add_argument("--quantum-mode", choices=QUANTUM_MODES, default="both")
    p.add_argument("--quantum-solver", choices=QUANTUM_SOLVERS, default="all")
    p.add_argument("--quantum-weight-step", type=float, default=0.05)
    p.add_argument("--qaoa-reps", type=int, default=1)
    p.add_argument("--qaoa-shots", type=int, default=2048)
    p.add_argument("--qaoa-seed", type=int, default=42)
    p.add_argument("--binary-budget", type=int, choices=(1, 2), default=1)
    p.add_argument("--no-budget-constraint", action="store_true")

    # misc
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--save-intermediate", action="store_true",
                   help="Also save prices.csv and returns.csv to output dir")
    p.add_argument("--verbose", action="store_true")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = Config(
        start=args.start,
        end=args.end,
        frequency=args.frequency,
        annualization_factor=args.annualization_factor,
        output_dir=args.output_dir,
        tickers=tuple(args.tickers),
        step=args.step,
        objective=args.objective,
        risk_free_rate=args.risk_free_rate,
        risk_aversion=args.risk_aversion,
        max_drawdown=args.max_drawdown,
        use_quantum=args.use_quantum,
        quantum_mode=args.quantum_mode,
        quantum_solver=args.quantum_solver,
        quantum_weight_step=args.quantum_weight_step,
        qaoa_reps=args.qaoa_reps,
        qaoa_shots=args.qaoa_shots,
        qaoa_seed=args.qaoa_seed,
        binary_budget=args.binary_budget,
        no_budget_constraint=args.no_budget_constraint,
        no_plots=args.no_plots,
        save_intermediate=args.save_intermediate,
        verbose=args.verbose,
    )
    return run(config)


if __name__ == "__main__":
    sys.exit(main())
