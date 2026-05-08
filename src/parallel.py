"""Parallel execution layer.

What is parallelizable in this codebase:
- The 6 quantum solver calls (binary × {exact, qaoa, sampling_vqe} +
  discrete × {exact, qaoa, sampling_vqe}) are **independent**: each builds
  its own ``QuadraticProgram`` and runs ``MinimumEigenOptimizer`` against
  its own ``Sampler``. Running them concurrently with
  ``ProcessPoolExecutor`` gives true multi-core utilization.

What is NOT worth parallelizing here:
- ``grid_search`` over 101 candidates at step=0.01 finishes in ~25 ms; the
  cost of pickling and process spawn would exceed the win.
- Mean-Variance is a single SLSQP call (~2 ms).
- Benchmark eval is 3 vector ops.
- yfinance is one HTTPS request.

Process pool, not threads:
- The expensive work (StatevectorSampler matvec, COBYLA loop) lives in C
  extensions and already releases the GIL — but we still need separate
  Python interpreters to push past the *single Python evaluator*. So
  ``ProcessPoolExecutor`` is the correct choice; ``ThreadPoolExecutor``
  would be CPU-bound from the GIL side until the C call.

Spawn vs fork:
- We use ``mp.get_context("spawn")``. Fork inherits the parent's qiskit
  state — fine if no threads have started yet, but flaky in practice
  (numpy/aer C++ globals). Spawn pays a ~5-10 s import-qiskit penalty per
  worker but is reproducible. For solver tasks that take minutes, this is
  a clearly favorable trade.

BLAS oversubscription:
- Each worker process has its own BLAS thread pool. If we leave
  ``OMP_NUM_THREADS=4`` and run 4 workers, we get 4 × 4 = 16 threads
  competing for 4 cores. ``submit.sh`` sets ``OMP_NUM_THREADS=1`` etc.
  before launching Python so each worker uses exactly 1 core.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np

logger = logging.getLogger("portfolio_optimizer")


def compute_workers(requested: int) -> int:
    """Resolve ``--workers`` to a concrete worker count.

    requested = 0  → SLURM_CPUS_PER_TASK if set, else min(8, cpu_count() // 2).
    requested >= 1 → exact value (caller is explicit).
    """
    if requested >= 1:
        return requested
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm and slurm.isdigit():
        return max(1, int(slurm))
    cpu = os.cpu_count() or 1
    return max(1, min(8, cpu // 2))


def print_parallelism_report(requested: int, effective: int) -> None:
    """Diagnostic banner — printed *unconditionally* at startup.

    Goes to stdout (not the logger) so it shows up in SLURM .out files
    even when ``--verbose`` isn't set. Format intentionally matches the
    user-visible diagnostic schema: ``[parallel] key = value``.
    """
    print("=" * 60, flush=True)
    print(f"[parallel] requested workers = {requested}", flush=True)
    print(f"[parallel] effective workers = {effective}", flush=True)
    print(f"[parallel] os.cpu_count = {os.cpu_count()}", flush=True)
    for key in (
        "SLURM_JOB_ID",
        "SLURM_CPUS_PER_TASK",
        "SLURM_NTASKS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        val = os.environ.get(key, "(unset)")
        print(f"[parallel] {key} = {val}", flush=True)
    try:
        method = mp.get_start_method(allow_none=True) or "(default)"
    except Exception:
        method = "(unknown)"
    print(f"[parallel] multiprocessing start method = {method}", flush=True)
    print("=" * 60, flush=True)


# ---- Quantum worker -----------------------------------------------------

def quantum_worker(payload: dict) -> dict:
    """Top-level pickleable worker for one quantum solve.

    Input ``payload`` keys: ``kind`` ("binary" | "discrete"), ``solver``
    ("exact" | "qaoa" | "sampling_vqe"), plus the data needed by the
    underlying solver.

    Returns ``{"row": <result-dict>, "samples": <DataFrame|None>}``.

    Exceptions are NOT swallowed: they propagate to the executor and the
    caller's ``future.result()`` will re-raise. The caller logs the error
    and moves on to the next task.
    """
    pid = os.getpid()
    kind = payload["kind"]
    solver = payload["solver"]
    print(f"[worker] pid={pid} kind={kind} solver={solver} starting", flush=True)
    t0 = time.time()

    if kind == "binary":
        from .quantum_binary import solve_binary
        row = solve_binary(
            payload["asset_returns"], payload["mu"], payload["Sigma"],
            solver=solver,
            risk_aversion=payload["risk_aversion"],
            risk_free_rate=payload["risk_free_rate"],
            budget=payload["budget"],
            no_budget_constraint=payload["no_budget_constraint"],
            qaoa_reps=payload["qaoa_reps"],
            qaoa_shots=payload["qaoa_shots"],
            qaoa_seed=payload["qaoa_seed"],
        )
        samples = None
    elif kind == "discrete":
        from .quantum_discrete_weights import solve_discrete
        row, samples = solve_discrete(
            payload["asset_returns"], payload["mu"], payload["Sigma"],
            payload["candidates"],
            solver=solver,
            risk_aversion=payload["risk_aversion"],
            risk_free_rate=payload["risk_free_rate"],
            qaoa_reps=payload["qaoa_reps"],
            qaoa_shots=payload["qaoa_shots"],
            qaoa_seed=payload["qaoa_seed"],
        )
    else:
        raise ValueError(f"unknown task kind: {kind}")

    elapsed = time.time() - t0
    print(
        f"[worker] pid={pid} kind={kind} solver={solver} done in {elapsed:.1f}s "
        f"success={row.get('success')}",
        flush=True,
    )
    return {"row": row, "samples": samples}


# ---- CPU-bound smoke task (verifies multiprocessing actually parallelizes) --

def smoke_task(args: tuple[int, int]) -> dict:
    """One CPU-bound matmul loop for ~``seconds`` seconds. Pickleable."""
    idx, seconds = args
    pid = os.getpid()
    start = time.time()
    print(f"[smoke worker] pid={pid} task_id={idx} starting (target {seconds}s)", flush=True)
    rng = np.random.default_rng(idx)
    deadline = start + seconds
    n_iters = 0
    accum = 0.0
    # 250×250 matmul with OMP_NUM_THREADS=1 keeps each task firmly on one core
    while time.time() < deadline:
        a = rng.random((250, 250))
        b = rng.random((250, 250))
        accum += float((a @ b).sum())
        n_iters += 1
    elapsed = time.time() - start
    print(
        f"[smoke worker] pid={pid} task_id={idx} done in {elapsed:.1f}s, "
        f"n_iters={n_iters}",
        flush=True,
    )
    return {"task_id": idx, "pid": pid, "elapsed": elapsed, "n_iters": n_iters}


def run_smoke_test(workers: int, n_tasks: int = 4, seconds: int = 25) -> int:
    """Run ``n_tasks`` CPU-bound tasks across ``workers`` processes.

    Returns shell exit code: 0 if wall time is within the parallel
    expectation (≤ 1.5× the perfect-parallel baseline), 1 if it looks like
    sequential execution (≥ 70% of n_tasks × seconds).
    """
    print(f"[smoke] launching {n_tasks} CPU-bound tasks across {workers} workers", flush=True)
    print(f"[smoke] each task targets ~{seconds}s of work", flush=True)
    perfect = (seconds * n_tasks) / max(workers, 1)
    print(f"[smoke] perfect-parallel wall ≈ {perfect:.1f}s (n_tasks×s/workers)", flush=True)
    print(f"[smoke] expected total %CPU on top: ≈ {min(workers, n_tasks) * 100}%", flush=True)
    print("[smoke] ----- worker output below -----", flush=True)

    ctx = mp.get_context("spawn")
    t0 = time.time()
    pids: set[int] = set()
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futures = [ex.submit(smoke_task, (i, seconds)) for i in range(n_tasks)]
        for fut in as_completed(futures):
            r = fut.result()  # surfaces exceptions
            pids.add(r["pid"])
    wall = time.time() - t0
    print("[smoke] ----- end worker output -----", flush=True)
    print(f"[smoke] wall = {wall:.1f}s, unique pids = {len(pids)} ({sorted(pids)})", flush=True)

    sequential_threshold = 0.7 * seconds * n_tasks
    if wall >= sequential_threshold:
        print(
            f"[smoke] FAIL: wall {wall:.1f}s ≥ {sequential_threshold:.1f}s "
            f"(70% of sequential {seconds * n_tasks}s) — multiprocessing is NOT parallel",
            flush=True,
        )
        return 1
    if len(pids) < min(workers, n_tasks):
        print(
            f"[smoke] FAIL: only {len(pids)} unique pids but expected ≥ "
            f"{min(workers, n_tasks)} — workers did not fan out",
            flush=True,
        )
        return 1
    print(
        f"[smoke] PASS: {len(pids)} workers ran in parallel, wall {wall:.1f}s "
        f"vs sequential {seconds * n_tasks}s (speedup {seconds * n_tasks / wall:.2f}×)",
        flush=True,
    )
    return 0


# ---- Parallel quantum dispatch -----------------------------------------

def run_quantum_parallel(
    payloads: list[dict],
    workers: int,
    on_result,
    on_error,
) -> None:
    """Run quantum payloads, dispatching results to ``on_result`` per task.

    Workers are spawned exactly once; tasks are scheduled by the executor.
    ``on_result(payload, result)`` is called from the **main** process so
    side effects (CSV flush, in-memory state) are naturally serialized.
    ``on_error(payload, exc)`` is called when a worker raises.

    No silent fall-back to serial: caller is responsible for choosing
    workers ≥ 1; this function always uses a process pool.
    """
    if not payloads:
        return

    n_workers = min(workers, len(payloads))
    print(
        f"[parallel] using ProcessPoolExecutor max_workers={n_workers} total tasks={len(payloads)}",
        flush=True,
    )

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
        future_to_payload = {ex.submit(quantum_worker, p): p for p in payloads}
        for fut in as_completed(future_to_payload):
            payload = future_to_payload[fut]
            try:
                result = fut.result()
            except Exception as e:  # noqa: BLE001
                on_error(payload, e)
                continue
            on_result(payload, result)
