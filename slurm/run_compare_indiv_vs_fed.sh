#!/bin/bash
# Slurm sbatch script for the individual-vs-federated comparison pipeline.
#
# Submit:
#     sbatch slurm/run_compare_indiv_vs_fed.sh
#     sbatch --partition=a100 slurm/run_compare_indiv_vs_fed.sh
#
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=compare_indiv_vs_fed
#SBATCH --output=/vol/bitbucket/fp223/EquitableEdu/slurm_logs/compare_indiv_vs_fed_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=f.pala23@imperial.ac.uk

set -euo pipefail

REPO=/vol/bitbucket/fp223/EquitableEdu
SPLITS="${REPO}/data/splits"
INDIV="${REPO}/experiment_outputs/unifiedfl_fp_individual_experiment/unifiedfl_fp_individual_experiment/unbalanced"
FED_BASE="${REPO}/experiment_outputs/unifiedfl_fp_federated_experiment"

# venv + ~/.local/bin (rclone) + cuda toolkit
export PATH="${HOME}/.local/bin:${REPO}/.venv/bin:${PATH}"
. /vol/cuda/12.4.0/setup.sh

cd "${REPO}"

# ─── Background sync to Google Drive ──────────────────────────────────────
# Same pattern as slurm/run_federated_fold.sh: 60s incremental sync of the
# whole experiment_outputs/ tree, plus a final flush in the EXIT trap.
LOCAL_OUTPUTS="${REPO}/experiment_outputs"
DRIVE_OUTPUTS="gdrive:EquitableEdu/Furkan Pala/experiment_outputs"
RCLONE_LOG="${REPO}/slurm_logs/compare_indiv_vs_fed_${SLURM_JOB_ID}.rclone.log"
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
echo "  COMPARE indiv vs fed   job=${SLURM_JOB_ID}  node=$(hostname)"
echo "  $(date)"
echo "================================================================"
nvidia-smi
python -V
echo "torch: $(python -c 'import torch; print(torch.__version__, torch.cuda.is_available())')"
echo "----------------------------------------------------------------"

# ─── Step 1: back-fill individual checkpoints with local_test + global_test ─
declare -A SPECS=(
    [0]='facebook/bart-base bart q_proj v_proj'
    [1]='google/flan-t5-base t5 q v'
    [2]='allenai/led-base-16384 led q_proj v_proj'
)

echo
echo "=================================================="
echo " STEP 1  individual back-fill   (3 clients × 3 folds)"
echo "=================================================="
for cid in 0 1 2; do
    read -r MODEL FAMILY T1 T2 <<< "${SPECS[$cid]}"
    for k in 1 2 3; do
        echo
        echo "--- indiv  client=${cid}  fold=${k}  model=${MODEL} ---"
        python experiments/11_recover_eval_only.py \
            --output-dir   "${INDIV}" \
            --splits-dir   "${SPLITS}" \
            --client-id    "${cid}" \
            --fold         "${k}" \
            --conditioning topic \
            --model        "${MODEL}" \
            --family       "${FAMILY}" \
            --targets      "${T1}" "${T2}" \
            --no-heavy --eval-local-test --skip-val-eval \
            --global-test-file "${SPLITS}/global_test.json"
    done
done

# ─── Step 2: back-fill federated checkpoints with val + global_test ────────
echo
echo "=================================================="
echo " STEP 2  federated back-fill   (3 folds × {best,final})"
echo "=================================================="
for k in 1 2 3; do
    echo
    echo "--- fed  fold=${k} ---"
    python experiments/13_eval_federated_post_hoc.py \
        --fed-output-dir "${FED_BASE}_fold${k}" \
        --splits-dir     "${SPLITS}" \
        --fold           "${k}" \
        --conditioning   topic \
        --no-heavy \
        --eval-on        val global_test \
        --global-test-file "${SPLITS}/global_test.json"
done

# ─── Step 3: comparison ────────────────────────────────────────────────────
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
