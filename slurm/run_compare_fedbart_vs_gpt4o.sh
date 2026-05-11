#!/bin/bash
# Compare federated-BART vs GPT-4o on QA generation over the global mixed
# test set (data/splits/global_test.json).
#
# Submit (full run, all 916 rows × 3 folds + GPT-4o):
#     sbatch slurm/run_compare_fedbart_vs_gpt4o.sh
#
# Submit (pilot, 5 unique contexts, BART fold1 only, no GPT-4o):
#     sbatch --export=ALL,LIMIT_CONTEXTS=5,FOLDS=1,SKIP_GPT4O=1 \
#            --time=00:30:00 slurm/run_compare_fedbart_vs_gpt4o.sh
#
# Submit (pilot, 5 contexts, fold1 + GPT-4o):
#     sbatch --export=ALL,LIMIT_CONTEXTS=5,FOLDS=1 \
#            --time=00:30:00 slurm/run_compare_fedbart_vs_gpt4o.sh
#
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --job-name=cmp_fedbart_gpt4o
#SBATCH --output=/vol/bitbucket/fp223/EquitableEdu/slurm_logs/cmp_fedbart_gpt4o_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=f.pala23@imperial.ac.uk

set -euo pipefail

REPO=/vol/bitbucket/fp223/EquitableEdu
OUT_DIR="${REPO}/experiment_outputs/slm_vs_llm"

# Knobs (override via --export=ALL,VAR=value at sbatch time)
LIMIT_CONTEXTS="${LIMIT_CONTEXTS:-0}"        # 0 = use all 179 unique contexts
FOLDS="${FOLDS:-1 2 3}"                       # space-separated list
SKIP_GPT4O="${SKIP_GPT4O:-0}"                 # 1 to skip
SKIP_BART="${SKIP_BART:-0}"
FORCE_REGEN="${FORCE_REGEN:-0}"

export PATH="${HOME}/.local/bin:${REPO}/.venv/bin:${PATH}"
. /vol/cuda/12.4.0/setup.sh

cd "${REPO}"
mkdir -p "${OUT_DIR}"

# ─── Background sync to Google Drive ──────────────────────────────────────
LOCAL_OUTPUTS="${REPO}/experiment_outputs"
DRIVE_OUTPUTS="gdrive:EquitableEdu/Furkan Pala/experiment_outputs"
RCLONE_LOG="${REPO}/slurm_logs/cmp_fedbart_gpt4o_${SLURM_JOB_ID}.rclone.log"
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
    echo "  rclone sync loop pid=${SYNC_PID}"

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
echo "  COMPARE fed-BART vs GPT-4o   job=${SLURM_JOB_ID}  node=$(hostname)"
echo "  $(date)"
echo "  LIMIT_CONTEXTS=${LIMIT_CONTEXTS}  FOLDS='${FOLDS}'  "
echo "  SKIP_GPT4O=${SKIP_GPT4O}  SKIP_BART=${SKIP_BART}  FORCE_REGEN=${FORCE_REGEN}"
echo "================================================================"
nvidia-smi
python -V
echo "torch: $(python -c 'import torch; print(torch.__version__, torch.cuda.is_available())')"
echo "----------------------------------------------------------------"

EXTRA_FLAGS=()
[[ "${SKIP_GPT4O}"   == "1" ]] && EXTRA_FLAGS+=("--skip-gpt4o")
[[ "${SKIP_BART}"    == "1" ]] && EXTRA_FLAGS+=("--skip-bart")
[[ "${FORCE_REGEN}"  == "1" ]] && EXTRA_FLAGS+=("--force-regen")

python -u experiments/15_compare_fedbart_vs_gpt4o.py \
    --global-test-file "${REPO}/data/splits/global_test.json" \
    --fed-base-dir     "${REPO}/experiment_outputs/unifiedfl_fp_federated_experiment" \
    --splits-dir       "${REPO}/data/splits" \
    --out-dir          "${OUT_DIR}" \
    --folds            ${FOLDS} \
    --limit-contexts   "${LIMIT_CONTEXTS}" \
    --conditioning     topic \
    "${EXTRA_FLAGS[@]}"

echo "================================================================"
echo "  DONE   $(date)"
echo "================================================================"
