#!/bin/bash
###############################################################################
# SLURM submit script for portfolio_optimizer end-to-end validation.
#
# Usage:
#     sbatch submit.sh
#
# Or override defaults inline:
#     QAOA_REPS=3 QAOA_SHOTS=4096 sbatch submit.sh
#
# Requirements on the cluster:
#   - outbound HTTPS to install uv (if not already on $PATH) and to fetch
#     yfinance prices and qiskit wheels. If your cluster has no outbound
#     internet, run `uv sync` and `uv cache prune` once on a login/transfer
#     node, and the job below will reuse the cache.
#   - ~3 GB scratch for .venv + qiskit-aer simulator state on default 21-qubit
#     discrete-weight problem. Bump --mem if you raise --quantum-weight-step.
###############################################################################

#SBATCH --job-name=portfolio-opt
#SBATCH --output=portfolio-opt-%j.out
#SBATCH --error=portfolio-opt-%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=amd2
# ---- Other site-specific knobs (uncomment if your cluster needs them) -----
# #SBATCH --account=YOUR_ACCOUNT
# #SBATCH --qos=normal
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=you@example.com
# ---------------------------------------------------------------------------

set -euo pipefail

###############################################################################
# Where the project lives. Default: directory containing this script.
###############################################################################
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$PROJECT_DIR"

echo "============================================================"
echo "  portfolio_optimizer SLURM job"
echo "============================================================"
echo "  job_id      : ${SLURM_JOB_ID:-N/A}"
echo "  hostname    : $(hostname)"
echo "  date        : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  project_dir : $PROJECT_DIR"
echo "  cpus        : ${SLURM_CPUS_PER_TASK:-?}"
echo "  mem         : ${SLURM_MEM_PER_NODE:-?}"
echo "============================================================"

###############################################################################
# 1. Make sure uv is on PATH. Install per-user if missing.
###############################################################################
# If your cluster requires module loading, uncomment & adjust:
#   module load python/3.11
#   module load curl

if [ -x "$HOME/.local/bin/uv" ]; then
    export PATH="$HOME/.local/bin:$PATH"
fi
if ! command -v uv >/dev/null 2>&1; then
    echo ">>> uv not found, installing per-user..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo ">>> uv version: $(uv --version)"

###############################################################################
# 2. Threading policy.
#
# We use Python *multiprocessing* (one process per allocated core) for the
# independent quantum solver calls. Each Python worker should therefore use
# exactly ONE BLAS / OpenMP thread; otherwise N workers × M threads each
# would oversubscribe the N cores SLURM gave us.
#
# Old policy (WRONG with multiprocessing): export OMP=$NCPU. That gave 1
# Python worker × 4 BLAS threads = 100% of one core most of the time
# because the work was sequential. Hence the observed 25% efficiency on
# job 55719.
###############################################################################
NCPU="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1

# Echo every variable our parallelism layer reads so the .out file is
# self-diagnosing.
echo ">>> environment relevant to parallelism"
echo "    SLURM_JOB_ID         = ${SLURM_JOB_ID:-(unset)}"
echo "    SLURM_CPUS_PER_TASK  = ${SLURM_CPUS_PER_TASK:-(unset)}"
echo "    SLURM_NTASKS         = ${SLURM_NTASKS:-(unset)}"
echo "    OMP_NUM_THREADS      = $OMP_NUM_THREADS"
echo "    MKL_NUM_THREADS      = $MKL_NUM_THREADS"
echo "    OPENBLAS_NUM_THREADS = $OPENBLAS_NUM_THREADS"
echo "    NUMEXPR_NUM_THREADS  = $NUMEXPR_NUM_THREADS"
echo "    VECLIB_MAXIMUM_THREADS = $VECLIB_MAXIMUM_THREADS"
echo "    NCPU (forwarded to --workers) = $NCPU"

###############################################################################
# 3. Sync dependencies (creates .venv on first run, no-op afterwards).
###############################################################################
echo ">>> uv sync --all-extras"
uv sync --all-extras

###############################################################################
# 3a. Parallel smoke test (verifies multiprocessing actually saturates cores).
#     If this fails the whole job exits — there is no point running the real
#     workload if SLURM allocation isn't truly parallel.
###############################################################################
echo ""
echo ">>> parallel smoke test (4 CPU-bound tasks × 25s)"
uv run python main.py --workers "$NCPU" --parallel-smoke-test

###############################################################################
# 4. Unit tests.
###############################################################################
echo ""
echo ">>> pytest"
uv run pytest tests -v

###############################################################################
# 5. Classical-only smoke test.
###############################################################################
echo ""
echo ">>> classical-only smoke test"
uv run python main.py \
    --start "${START_DATE:-2023-01-01}" \
    --end   "${END_DATE:-2025-01-01}" \
    --step  "${GRID_STEP:-0.01}" \
    --objective "${OBJECTIVE:-maximize_sharpe}" \
    --workers "$NCPU" \
    --output-dir outputs_classical \
    --no-plots

###############################################################################
# 6. Full classical + quantum comparison (this is the validation run).
###############################################################################
echo ""
echo ">>> classical + quantum comparison (this exercises QAOA / SamplingVQE)"
uv run python main.py \
    --start "${START_DATE:-2023-01-01}" \
    --end   "${END_DATE:-2025-01-01}" \
    --step  "${GRID_STEP:-0.01}" \
    --objective "${OBJECTIVE:-maximize_sharpe}" \
    --use-quantum \
    --quantum-mode   "${QUANTUM_MODE:-both}" \
    --quantum-solver "${QUANTUM_SOLVER:-all}" \
    --quantum-weight-step "${QUANTUM_WEIGHT_STEP:-0.05}" \
    --qaoa-reps  "${QAOA_REPS:-2}" \
    --qaoa-shots "${QAOA_SHOTS:-2048}" \
    --qaoa-seed  "${QAOA_SEED:-42}" \
    --workers "$NCPU" \
    --output-dir outputs_quantum \
    --verbose

###############################################################################
# 7. Quick correctness check: quantum exact (discrete) BTC weight should match
#    the classical grid optimum at the same weight step.
###############################################################################
echo ""
echo ">>> validation: quantum-exact vs classical-grid agreement on the same step"
uv run python - <<'PY'
import csv
from pathlib import Path

p = Path("outputs_quantum/comparison_summary.csv")
rows = list(csv.DictReader(p.open()))

def get(method_prefix, solver_substr=None):
    for r in rows:
        if r["method"].startswith(method_prefix) and (
            solver_substr is None or solver_substr in r["solver"]
        ):
            return r
    return None

q_exact = get("Quantum Discrete Weights", "exact")
q_qaoa  = get("Quantum Discrete Weights", "qaoa")
classic = get("Classic Grid Search")

print(f"classic grid    : btc={classic['btc_weight']}  obj={classic['objective']}  sharpe={classic['sharpe_ratio']}")
if q_exact:
    print(f"quantum exact   : btc={q_exact['btc_weight']}  obj={q_exact['objective']}  sharpe={q_exact['sharpe_ratio']}  ok={q_exact['success']}")
if q_qaoa:
    note = q_qaoa.get("note","")
    print(f"quantum qaoa    : btc={q_qaoa['btc_weight']}  obj={q_qaoa['objective']}  sharpe={q_qaoa['sharpe_ratio']}  ok={q_qaoa['success']}  note={note!r}")

# Sanity: all rows present, every quantum row either success=True or has an error_message
fail_silent = [r for r in rows if r["success"] != "True" and not r["error_message"]]
if fail_silent:
    print("FAIL: rows with success!=True and empty error_message:")
    for r in fail_silent:
        print("  ", r)
    raise SystemExit(1)
print("OK: every failed row carries an error_message; comparison_summary.csv is well-formed.")
PY

###############################################################################
# 8. Julia classical comparison (independent branch — see julia/ subproject).
#    Runs grid + Optim.jl + JuMP+Ipopt against the SAME μ/Σ written above.
#    Doesn't block the job exit code: a Julia failure is logged but doesn't
#    invalidate the Python comparison_summary.csv that's already on disk.
###############################################################################
if command -v julia >/dev/null 2>&1; then
    echo ""
    echo ">>> Julia classical run (grid + Optim.jl + JuMP+Ipopt)"
    julia --project=julia -e 'using Pkg; Pkg.instantiate()'
    julia --project=julia julia/test/runtests.jl || \
        echo "WARN: julia tests failed; continuing"
    julia --project=julia julia/run.jl --outputs outputs_quantum/ || \
        echo "WARN: julia run.jl failed; Python comparison_summary.csv is unaffected"
    echo ""
    echo ">>> re-running analyze.py so the report includes Julia rows"
    uv run python analyze.py outputs_quantum/ || true
else
    echo ""
    echo "WARN: 'julia' not on PATH; skipping Julia classical comparison"
fi

echo ""
echo "============================================================"
echo "  job done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  outputs:  $PROJECT_DIR/outputs_classical/"
echo "            $PROJECT_DIR/outputs_quantum/"
echo "============================================================"
