#!/bin/bash
# Slurm sbatch script for one fold of federated training.
# Submit with the FOLD env var set, e.g.:
#     sbatch --export=ALL,FOLD=2 slurm/run_federated_fold.sh
#     sbatch --export=ALL,FOLD=3 slurm/run_federated_fold.sh
#
# Optional partition override:
#     sbatch --partition=a40 --export=ALL,FOLD=2 slurm/run_federated_fold.sh
#
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --output=/vol/bitbucket/fp223/EquitableEdu/slurm_logs/fed_fold_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=f.pala23@imperial.ac.uk

set -euo pipefail

if [[ -z "${FOLD:-}" ]]; then
    echo "ERROR: FOLD env var not set. Submit with --export=ALL,FOLD=<n>" >&2
    exit 2
fi

REPO=/vol/bitbucket/fp223/EquitableEdu
OUT="${REPO}/experiment_outputs/unifiedfl_fp_federated_experiment_fold${FOLD}"

# Use the existing venv (Python 3.12, torch 2.5.1+cu124).
# Also expose ~/.local/bin so the rclone binary installed there is on PATH.
export PATH="${HOME}/.local/bin:${REPO}/.venv/bin:${PATH}"

# CUDA — torch ships its own runtime, but sourcing this puts nvcc on PATH for
# any subprocesses that need it.
. /vol/cuda/12.4.0/setup.sh

cd "${REPO}"
mkdir -p "${OUT}"

# ─── Background sync to Google Drive ──────────────────────────────────────
# Canonical Drive layout: gdrive:EquitableEdu/Furkan Pala/experiment_outputs/
# (the space in "Furkan Pala" is intentional; quote it everywhere).
# Loop scope = the whole experiment_outputs/ tree, so concurrent fold jobs
# converge to the same Drive subtree. Errors swallowed; final flush in trap.
#   --update      copy only when source is newer (safe for live training files)
#   --no-traverse skip listing the remote each pass; cheap incremental scans
#   --quiet       per-loop output kept terse; full activity goes to log file
LOCAL_OUTPUTS="${REPO}/experiment_outputs"
DRIVE_OUTPUTS="gdrive:EquitableEdu/Furkan Pala/experiment_outputs"
RCLONE_LOG="${OUT}/rclone_sync.log"
SYNC_PID=""

if command -v rclone >/dev/null 2>&1 \
        && rclone listremotes 2>/dev/null | grep -q '^gdrive:'; then
    (
        # Loop body: any rclone failure is swallowed so the loop survives
        # transient network blips.
        while true; do
            rclone copy "${LOCAL_OUTPUTS}" "${DRIVE_OUTPUTS}" \
                --update --no-traverse \
                --log-file "${RCLONE_LOG}" --log-level INFO \
                >/dev/null 2>&1 || true
            sleep 60
        done
    ) &
    SYNC_PID=$!
    echo "  rclone sync loop pid=${SYNC_PID}  ${LOCAL_OUTPUTS}  ->  ${DRIVE_OUTPUTS}"
    echo "  rclone log: ${RCLONE_LOG}"

    cleanup () {
        if [[ -n "${SYNC_PID}" ]] && kill -0 "${SYNC_PID}" 2>/dev/null; then
            kill "${SYNC_PID}" 2>/dev/null || true
            wait "${SYNC_PID}" 2>/dev/null || true
        fi
        echo "  Final rclone flush  ${LOCAL_OUTPUTS}  ->  ${DRIVE_OUTPUTS}"
        rclone copy "${LOCAL_OUTPUTS}" "${DRIVE_OUTPUTS}" --update \
            --log-file "${RCLONE_LOG}" --log-level INFO || \
            echo "  WARNING: final rclone flush failed (see ${RCLONE_LOG})" >&2
    }
    trap cleanup EXIT
else
    echo "  WARNING: rclone or gdrive: remote not configured — Drive sync disabled."
fi

echo "================================================================"
echo "  FED TRAINING fold=${FOLD}  job=${SLURM_JOB_ID}  node=$(hostname)"
echo "  $(date)"
echo "================================================================"
nvidia-smi
python -V
echo "torch: $(python -c 'import torch; print(torch.__version__, torch.cuda.is_available())')"
echo "----------------------------------------------------------------"

python -u experiments/08_run_federated_training.py \
    --splits-dir   data/splits \
    --output-dir   "${OUT}" \
    --fold         "${FOLD}" \
    --conditioning topic \
    --fast-eval

echo "================================================================"
echo "  DONE fold=${FOLD}  $(date)"
echo "================================================================"
