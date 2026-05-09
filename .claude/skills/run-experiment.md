---
name: run-experiment
description: Submit a SLURM job on the Yonsei cluster, monitor it, fetch results to local Mac, and produce an analysis report. Use when the user asks to "run on the cluster", "validate on HPC", or "re-run with different params".
---

# Run a SLURM Experiment

## Standard end-to-end workflow

```bash
# 1. Make sure all local commits are on GitHub
cd /Users/luzeyu/Desktop/finance/portfolio_optimizer
git push

# 2. Pull on cluster (SSH alias: yonsei_jang, home /home/zeyu/)
ssh yonsei_jang 'cd ~/btcandbnb && git pull origin main'

# 3. Submit (default 4 cpus / 8G / 30 min on amd2 partition)
ssh yonsei_jang 'cd ~/btcandbnb && sbatch submit.sh'

# 4. Monitor (run on cluster)
ssh yonsei_jang 'squeue -u $USER'
ssh yonsei_jang 'tail -f ~/btcandbnb/portfolio-opt-<jobid>.out'

# 5. After done — efficiency check
ssh yonsei_jang 'seff <jobid>'

# 6. Pull results back to Mac
rsync -av yonsei_jang:/home/zeyu/btcandbnb/outputs_quantum \
          yonsei_jang:/home/zeyu/btcandbnb/portfolio-opt-<jobid>.out \
          /Users/luzeyu/Desktop/finance/portfolio_optimizer/

# 7. Generate report
cd /Users/luzeyu/Desktop/finance/portfolio_optimizer
uv run python analyze.py outputs_quantum/
open outputs_quantum/analysis_report.md outputs_quantum/verdict.png
```

## Tunable parameters (env vars to sbatch)

| Var | Default | Effect |
|---|---|---|
| `QUANTUM_WEIGHT_STEP` | 0.05 | 0.1 → 11 qubits (~50× faster QAOA, fits in 30 min) |
| `QAOA_REPS` | 2 | 3-4 for better convergence at slow cost |
| `QAOA_SHOTS` | 2048 | 4096 for sharper distributions |
| `START_DATE`, `END_DATE` | last 2 years | Different historical windows |
| `OBJECTIVE` | maximize_sharpe | minimize_volatility, etc. |

Example: `QUANTUM_WEIGHT_STEP=0.1 QAOA_REPS=3 sbatch submit.sh`

## Common failure modes (most → least common)

### TIME LIMIT killed mid-quantum
**Symptom**: `sacct` shows State=TIMEOUT, `comparison_summary.csv` has < 11 rows.
**Cause**: 21-qubit QAOA / SamplingVQE on default step=0.05 takes 5-15 min each.
**Fix**: re-submit with `QUANTUM_WEIGHT_STEP=0.1`. The `runner.py` already
flushes per-solver, so older results from the killed job are still on disk.

### Job FAILED with elapsed 0:00:00
**Symptom**: `sacct` State=FAILED, ExitCode=0:53, no log files.
**Cause**: SBATCH `--output` path can't be opened (e.g., `logs/` dir missing).
**Fix**: ensure `submit.sh` has `--output=portfolio-opt-%j.out` (root-level,
no subdir), not `--output=logs/%j.out`.

### `uv sync` fails on compute node
**Symptom**: stderr shows network/DNS errors.
**Cause**: compute nodes may not have outbound HTTPS.
**Fix**: pre-warm cache on login node: `ssh yonsei_jang 'cd ~/btcandbnb && uv sync --all-extras'`.
NFS-shared `~/.cache/uv/` makes this a one-time cost.

### Code on cluster is stale
**Symptom**: `.out` lacks `[parallel]` banner, behavior matches old commit.
**Diagnose**: `ssh yonsei_jang 'cd ~/btcandbnb && git log --oneline -1'`.
**Fix**: `ssh yonsei_jang 'cd ~/btcandbnb && git fetch origin && git reset --hard origin/main'`.
The `--hard` is safe because the cluster repo has no local edits worth keeping.

### CPU efficiency < 50%
**Symptom**: `seff` shows 25-30%.
**Most likely cause**: 4 of 6 quantum tasks finish quickly; only 2 slow ones
(discrete qaoa + sampling_vqe at 21 qubits) keep cores busy. **This is
the natural ceiling** for the imbalanced workload, not a bug.
**To improve**: `QUANTUM_WEIGHT_STEP=0.1` makes all 6 tasks roughly equal
duration → all 4 cores stay busy → efficiency 70%+.

## Validating a run

After fetching results, the FIRST thing analyze.py prints is the
"cross-validation" section. It must show:

```
✓ PASS — Quantum Discrete picked BTC = X.XX, the nearest 0.05-step
neighbour of the classical grid optimum BTC = X.XXX.
```

If this fails, do NOT trust any other quantum result. Check
`quantum_binary.py::_build_qp` — most likely the Σ off-diagonal
coefficient (currently `2.0 * risk_aversion * Sigma[0,1]`) was
edited by mistake.
