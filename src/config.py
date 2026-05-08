"""Config dataclass produced by main.py's argparse."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

CLASSIC_OBJECTIVES = (
    "maximize_sharpe",
    "minimize_volatility",
    "maximize_return",
    "minimize_cvar",
    "constrained_sharpe",
)
QUANTUM_MODES = ("binary", "discrete", "both")
QUANTUM_SOLVERS = ("exact", "qaoa", "sampling_vqe", "all")
TICKERS_DEFAULT = ("BTC-USD", "BNB-USD")


def default_start() -> str:
    return (date.today() - timedelta(days=730)).isoformat()


def default_end() -> str:
    return date.today().isoformat()


@dataclass
class Config:
    # data
    start: str = field(default_factory=default_start)
    end: str = field(default_factory=default_end)
    frequency: str = "1d"
    annualization_factor: int = 365
    output_dir: str = "outputs"
    tickers: tuple[str, str] = TICKERS_DEFAULT

    # classical
    step: float = 0.01
    objective: str = "maximize_sharpe"
    risk_free_rate: float = 0.0
    risk_aversion: float = 1.0
    max_drawdown: float = -0.5  # threshold for constrained_sharpe (e.g. -0.5 = -50%)

    # quantum
    use_quantum: bool = False
    quantum_mode: str = "both"
    quantum_solver: str = "all"
    quantum_weight_step: float = 0.05
    qaoa_reps: int = 1
    qaoa_shots: int = 2048
    qaoa_seed: int = 42
    binary_budget: int = 1
    no_budget_constraint: bool = False

    # parallelism
    workers: int = 0  # 0 → auto-detect (SLURM_CPUS_PER_TASK or cpu_count//2)
    parallel_smoke_test: bool = False

    # misc
    no_plots: bool = False
    save_intermediate: bool = False
    verbose: bool = False

    def validate(self) -> None:
        if self.objective not in CLASSIC_OBJECTIVES:
            raise ValueError(
                f"--objective must be one of {CLASSIC_OBJECTIVES}, got {self.objective!r}"
            )
        if self.quantum_mode not in QUANTUM_MODES:
            raise ValueError(
                f"--quantum-mode must be one of {QUANTUM_MODES}, got {self.quantum_mode!r}"
            )
        if self.quantum_solver not in QUANTUM_SOLVERS:
            raise ValueError(
                f"--quantum-solver must be one of {QUANTUM_SOLVERS}, got {self.quantum_solver!r}"
            )
        if not 0 < self.step <= 0.5:
            raise ValueError(f"--step must be in (0, 0.5], got {self.step}")
        if not 0 < self.quantum_weight_step <= 0.5:
            raise ValueError(
                f"--quantum-weight-step must be in (0, 0.5], got {self.quantum_weight_step}"
            )
        if self.binary_budget not in (1, 2):
            raise ValueError(f"--binary-budget must be 1 or 2, got {self.binary_budget}")
        if self.qaoa_reps < 1:
            raise ValueError(f"--qaoa-reps must be >= 1, got {self.qaoa_reps}")
        if self.qaoa_shots < 1:
            raise ValueError(f"--qaoa-shots must be >= 1, got {self.qaoa_shots}")
        if self.annualization_factor <= 0:
            raise ValueError("--annualization-factor must be positive")
        if self.workers < 0 or self.workers > 64:
            raise ValueError(f"--workers must be in [0, 64], got {self.workers}")

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)
