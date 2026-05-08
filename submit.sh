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
# 2. Pin threading to the SLURM allocation. Qiskit-Aer + BLAS otherwise oversubscribe.
###############################################################################
NCPU="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="$NCPU"
export OPENBLAS_NUM_THREADS="$NCPU"
export MKL_NUM_THREADS="$NCPU"
export QISKIT_NUM_THREADS="$NCPU"
export PYTHONUNBUFFERED=1

###############################################################################
# 3. Sync dependencies (creates .venv on first run, no-op afterwards).
###############################################################################
echo ">>> uv sync --all-extras"
uv sync --all-extras

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

echo ""
echo "============================================================"
echo "  job done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  outputs:  $PROJECT_DIR/outputs_classical/"
echo "            $PROJECT_DIR/outputs_quantum/"
echo "============================================================"
