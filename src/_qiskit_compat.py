"""Qiskit version-tolerant shims.

The qiskit ecosystem moves fast: ``qiskit.algorithms`` moved out into the
``qiskit-algorithms`` package, ``Sampler`` lives in ``qiskit.primitives`` or
``qiskit_aer.primitives`` depending on what's installed, and
``SamplingVQE`` only exists in newer ``qiskit-algorithms`` releases.

This module probes what is importable and exposes a small surface area
(``QISKIT_AVAILABLE``, ``build_sampler``, ``build_qaoa``, …) so the rest of
the codebase can write straight-line code without try/except sprinkled
everywhere. If anything fails the per-attribute symbol is ``None`` and the
helpers raise ``QiskitNotAvailable`` with a readable message.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("portfolio_optimizer")


class QiskitNotAvailable(RuntimeError):
    pass


# --- qiskit-optimization --------------------------------------------------

QuadraticProgram = None
MinimumEigenOptimizer = None
try:
    from qiskit_optimization import QuadraticProgram as _QP  # type: ignore[import-not-found]
    from qiskit_optimization.algorithms import (  # type: ignore[import-not-found]
        MinimumEigenOptimizer as _MEO,
    )

    QuadraticProgram = _QP
    MinimumEigenOptimizer = _MEO
except Exception as e:  # noqa: BLE001
    logger.debug("qiskit_optimization import failed: %s", e)


# --- algorithms (QAOA / NumPyMinimumEigensolver / COBYLA / ag) ------------

QAOA = None
NumPyMinimumEigensolver = None
COBYLA = None
algorithm_globals = None
SamplingVQE = None

for _modname in ("qiskit_algorithms", "qiskit.algorithms"):
    try:
        _mod = __import__(_modname, fromlist=["QAOA", "NumPyMinimumEigensolver"])
        QAOA = getattr(_mod, "QAOA", None)
        NumPyMinimumEigensolver = getattr(_mod, "NumPyMinimumEigensolver", None)
        if QAOA is not None and NumPyMinimumEigensolver is not None:
            break
    except Exception as e:  # noqa: BLE001
        logger.debug("%s import failed: %s", _modname, e)

for _opts in ("qiskit_algorithms.optimizers", "qiskit.algorithms.optimizers"):
    try:
        _mod = __import__(_opts, fromlist=["COBYLA"])
        COBYLA = getattr(_mod, "COBYLA", None)
        if COBYLA is not None:
            break
    except Exception as e:  # noqa: BLE001
        logger.debug("%s import failed: %s", _opts, e)

for _utils in ("qiskit_algorithms.utils", "qiskit.utils"):
    try:
        _mod = __import__(_utils, fromlist=["algorithm_globals"])
        algorithm_globals = getattr(_mod, "algorithm_globals", None)
        if algorithm_globals is not None:
            break
    except Exception as e:  # noqa: BLE001
        logger.debug("%s import failed: %s", _utils, e)

for _vqe in (
    "qiskit_algorithms.minimum_eigensolvers",
    "qiskit.algorithms.minimum_eigensolvers",
):
    try:
        _mod = __import__(_vqe, fromlist=["SamplingVQE"])
        SamplingVQE = getattr(_mod, "SamplingVQE", None)
        if SamplingVQE is not None:
            break
    except Exception as e:  # noqa: BLE001
        logger.debug("%s import failed: %s", _vqe, e)

# --- ansatz for SamplingVQE ----------------------------------------------

RealAmplitudes = None
try:
    from qiskit.circuit.library import RealAmplitudes as _RA  # type: ignore[import-not-found]

    RealAmplitudes = _RA
except Exception as e:  # noqa: BLE001
    logger.debug("RealAmplitudes import failed: %s", e)


# --- aggregate availability ----------------------------------------------

QISKIT_AVAILABLE = (
    QuadraticProgram is not None
    and MinimumEigenOptimizer is not None
    and NumPyMinimumEigensolver is not None
)
QAOA_AVAILABLE = QISKIT_AVAILABLE and QAOA is not None and COBYLA is not None
SAMPLING_VQE_AVAILABLE = QAOA_AVAILABLE and SamplingVQE is not None and RealAmplitudes is not None


def build_sampler(shots: int, seed: int) -> Any:
    """Construct a Sampler primitive that works with the installed Qiskit stack.

    Strategy in priority order (whichever first imports & instantiates wins):

    1. ``qiskit.primitives.StatevectorSampler`` — V2, ideal simulator. Works
       with ``qiskit-algorithms 0.4+`` on top of ``qiskit 2.x`` because the
       diagonal-estimator code path uses V2 SamplerPubs.
    2. ``qiskit_aer.primitives.SamplerV2`` — V2 with shot-noise simulation;
       only works if the QAOA circuit is fully transpilable to Aer basis gates,
       which sometimes fails on the high-level ``QAOA`` instruction. Used as
       a fallback for completeness.
    3. ``qiskit_aer.primitives.Sampler`` (V1, deprecated) — last-resort, only
       works on older qiskit-algorithms releases that still call V1 ``run``.

    The seed and shots are wired in regardless of class, with permissive kwargs.
    """
    last_err: Exception | None = None

    try:
        from qiskit.primitives import StatevectorSampler  # type: ignore[import-not-found]

        try:
            return StatevectorSampler(default_shots=shots, seed=seed)
        except TypeError:
            return StatevectorSampler()
    except Exception as e:  # noqa: BLE001
        last_err = e
        logger.debug("StatevectorSampler unavailable: %s", e)

    try:
        from qiskit_aer.primitives import SamplerV2 as AerSamplerV2  # type: ignore[import-not-found]

        try:
            return AerSamplerV2(seed=seed, default_shots=shots)
        except TypeError:
            return AerSamplerV2(seed=seed)
    except Exception as e:  # noqa: BLE001
        last_err = e
        logger.debug("Aer SamplerV2 unavailable: %s", e)

    try:
        from qiskit_aer.primitives import Sampler as AerSamplerV1  # type: ignore[import-not-found]

        return AerSamplerV1(
            run_options={"shots": shots, "seed": seed},
            transpile_options={"seed_transpiler": seed},
        )
    except Exception as e:  # noqa: BLE001
        last_err = e
        logger.debug("Aer Sampler V1 unavailable: %s", e)

    raise QiskitNotAvailable(f"no Sampler available: {last_err}")


def build_qaoa(sampler: Any, reps: int, seed: int) -> Any:
    """QAOA(sampler, COBYLA(), reps=reps). Raises if QAOA not importable."""
    if not QAOA_AVAILABLE:
        raise QiskitNotAvailable("QAOA or COBYLA not importable in this Qiskit install")
    if algorithm_globals is not None:
        algorithm_globals.random_seed = seed
    return QAOA(sampler=sampler, optimizer=COBYLA(), reps=reps)


def build_sampling_vqe(sampler: Any, num_qubits: int, reps: int, seed: int) -> Any:
    """SamplingVQE with a RealAmplitudes ansatz. Raises if not importable."""
    if not SAMPLING_VQE_AVAILABLE:
        raise QiskitNotAvailable("SamplingVQE not importable in this Qiskit install")
    if algorithm_globals is not None:
        algorithm_globals.random_seed = seed
    ansatz = RealAmplitudes(num_qubits=num_qubits, reps=reps)
    return SamplingVQE(sampler=sampler, ansatz=ansatz, optimizer=COBYLA())


def build_exact() -> Any:
    if NumPyMinimumEigensolver is None:
        raise QiskitNotAvailable("NumPyMinimumEigensolver not importable")
    return NumPyMinimumEigensolver()


def availability_summary() -> dict[str, bool]:
    return {
        "qiskit_optimization": QuadraticProgram is not None,
        "minimum_eigen_optimizer": MinimumEigenOptimizer is not None,
        "exact": NumPyMinimumEigensolver is not None,
        "qaoa": QAOA_AVAILABLE,
        "sampling_vqe": SAMPLING_VQE_AVAILABLE,
    }
