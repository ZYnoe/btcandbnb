"""Low-level helpers: random seeding, logging, safe imports, paths."""

from __future__ import annotations

import importlib
import logging
import os
import random
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np


def set_random_seed(seed: int) -> None:
    """Set every RNG we can reach so experiments are reproducible.

    Always sets numpy and stdlib random; sets ``algorithm_globals.random_seed``
    if qiskit's algorithm utilities are installed. Failure to set qiskit seed
    is silent — the caller may not have qiskit installed.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    for path in (
        "qiskit_algorithms.utils",
        "qiskit.utils",
    ):
        try:
            mod = importlib.import_module(path)
            ag = getattr(mod, "algorithm_globals", None)
            if ag is not None:
                ag.random_seed = seed
                break
        except Exception:
            continue


def safe_import(module_name: str) -> ModuleType | None:
    """Import a module, returning ``None`` on any failure (incl. ImportError, RuntimeError).

    Used for optional dependencies like qiskit so the caller can branch on availability.
    """
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def setup_logger(verbose: bool = False, name: str = "portfolio_optimizer") -> logging.Logger:
    """Configure a single project-wide logger. Idempotent."""
    logger = logging.getLogger(name)
    if logger.handlers:
        logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        return logger
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def ensure_dir(path: str | Path) -> Path:
    """Create directory if missing and return as Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy/pandas scalars to plain Python for json.dump."""
    import math

    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, (bool, int, str)) or obj is None:
        return obj
    try:
        return str(obj)
    except Exception:
        return None
