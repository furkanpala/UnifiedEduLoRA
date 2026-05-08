#!/bin/bash
# Slurm sbatch script — resume of run_compare_indiv_vs_fed.sh after the
# individual back-fill (step 1) crashed on the missing client_2/fold3
# adapter. This script runs ONLY:
#   • step 2: federated back-fill (val + global_test, all 3 folds)
#   • step 3: indiv-vs-fed comparison (14 handles the one missing cell)
#
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name=fed_eval_compare
#SBATCH --output=/vol/bitbucket/fp223/EquitableEdu/slurm_logs/fed_eval_compare_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=f.pala23@imperial.ac.uk

set -euo pipefail

REPO=/vol/bitbucket/fp223/EquitableEdu
SPLITS="${REPO}/data/splits"
INDIV="${REPO}/experiment_outputs/unifiedfl_fp_individual_experiment/unifiedfl_fp_individual_experiment/unbalanced"
FED_BASE="${REPO}/experiment_outputs/unifiedfl_fp_federated_experiment"

export PATH="${HOME}/.local/bin:${REPO}/.venv/bin:${PATH}"
. /vol/cuda/12.4.0/setup.sh

cd "${REPO}"

# ─── Background sync to Google Drive ──────────────────────────────────────
LOCAL_OUTPUTS="${REPO}/experiment_outputs"
DRIVE_OUTPUTS="gdrive:EquitableEdu/Furkan Pala/experiment_outputs"
RCLONE_LOG="${REPO}/slurm_logs/fed_eval_compare_${SLURM_JOB_ID}.rclone.log"
SYNC_PID=""

if command -v rclone >/dev/null 2>&1 \
        && rclone listremotes 2>/dev/null | grep -q '^gdrive:'; then
    (
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
echo "  FED EVAL + COMPARE   job=${SLURM_JOB_ID}  node=$(hostname)"
echo "  $(date)"
echo "================================================================"
nvidia-smi
python -V
echo "torch: $(python -c 'import torch; print(torch.__version__, torch.cuda.is_available())')"
echo "----------------------------------------------------------------"

# ─── Step 2: federated back-fill (val + global_test, 3 folds) ─────────────
echo
echo "=================================================="
echo " STEP 2  federated back-fill   (3 folds × {best,final})"
echo "=================================================="
for k in 1 2 3; do
    echo
    echo "--- fed  fold=${k} ---"
    python experiments/13_eval_federated_post_hoc.py \
        --fed-output-dir   "${FED_BASE}_fold${k}" \
        --splits-dir       "${SPLITS}" \
        --fold             "${k}" \
        --conditioning     topic \
        --no-heavy \
        --eval-on          val global_test \
        --global-test-file "${SPLITS}/global_test.json"
done

# ─── Step 3: comparison ────────────────────────────────────────────────────
# Note: client_2/fold3 has no individual checkpoint (Colab training was
# interrupted there). 14_compare_indiv_vs_fed.py reads each metrics_*.json
# defensively and renders "--" for missing cells, so the run completes and
# the table just shows n=2 for that one slice.
echo
echo "=================================================="
echo " STEP 3  comparison"
echo "=================================================="
SAVE_DIR="${REPO}/experiment_outputs"   # FED_BASE itself isn't a directory
mkdir -p "${SAVE_DIR}"
python experiments/14_compare_indiv_vs_fed.py \
    --indiv-dir    "${INDIV}" \
    --fed-dir      "${FED_BASE}" \
    --conditioning topic \
    --save         "${SAVE_DIR}/comparison_indiv_vs_fed.json" \
    | tee "${SAVE_DIR}/comparison_summary.txt"

echo "================================================================"
echo "  DONE   $(date)"
echo "================================================================"
