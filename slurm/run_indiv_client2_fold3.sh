#!/bin/bash
# One-off Slurm wrapper to fill the missing individual-training cell:
#   client_2 (allenai/led-base-16384), fold 3, conditioning topic.
#
# Output lands at:
#   experiment_outputs/unifiedfl_fp_individual_experiment/
#                      unifiedfl_fp_individual_experiment/
#                      unbalanced/topic/client_2/fold3/{best,final,results}/
# (same doubly-nested layout the Colab runs produced for fold1/fold2).
#
# Submit:
#     sbatch slurm/run_indiv_client2_fold3.sh
#
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --job-name=indiv_c2_f3
#SBATCH --output=/vol/bitbucket/fp223/EquitableEdu/slurm_logs/indiv_c2_f3_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=f.pala23@imperial.ac.uk

set -euo pipefail

REPO=/vol/bitbucket/fp223/EquitableEdu
SPLITS="${REPO}/data/splits"
# Match the existing Colab tree: --output-dir is the doubly-nested base + 'unbalanced'.
INDIV_OUT="${REPO}/experiment_outputs/unifiedfl_fp_individual_experiment/unifiedfl_fp_individual_experiment/unbalanced"

# venv + ~/.local/bin (rclone) + cuda
export PATH="${HOME}/.local/bin:${REPO}/.venv/bin:${PATH}"
. /vol/cuda/12.4.0/setup.sh

cd "${REPO}"
mkdir -p "${INDIV_OUT}"

# ─── Background sync to Drive ─────────────────────────────────────────────
LOCAL_OUTPUTS="${REPO}/experiment_outputs"
DRIVE_OUTPUTS="gdrive:EquitableEdu/Furkan Pala/experiment_outputs"
RCLONE_LOG="${REPO}/slurm_logs/indiv_c2_f3_${SLURM_JOB_ID}.rclone.log"
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
echo "  INDIV TRAIN  client=2  fold=3  LED-base-16384"
echo "  job=${SLURM_JOB_ID}  node=$(hostname)   $(date)"
echo "================================================================"
nvidia-smi
python -V
echo "torch: $(python -c 'import torch; print(torch.__version__, torch.cuda.is_available())')"
echo "----------------------------------------------------------------"

# Hyperparameters: take train_client.py defaults for everything except the
# required identity args (client/fold/model/family/targets), I/O paths, the
# conditioning mode, fast-eval-only flag, and the global test set. Defaults
# in train_client.py: num-epochs=100, patience=10, batch-size=4, lr=3e-4,
# warmup-ratio=0.1, grad-clip=1.0, lora-r=16, lora-alpha=32, lora-dropout=0.1,
# max-input-len=512, max-target-len=128, early-stop-metric=rouge_l.
python -u unifiedfl/train_client.py \
    --client-id        2 \
    --fold             3 \
    --model            allenai/led-base-16384 \
    --family           led \
    --targets          q_proj v_proj \
    --splits-dir       "${SPLITS}" \
    --output-dir       "${INDIV_OUT}" \
    --conditioning     topic \
    --no-heavy \
    --global-test-file "${SPLITS}/global_test.json"

echo "================================================================"
echo "  DONE   $(date)"
echo "================================================================"
