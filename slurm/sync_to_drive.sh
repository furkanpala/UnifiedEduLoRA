#!/usr/bin/env bash
# Push experiment outputs from /vol/bitbucket to Google Drive.
# Canonical Drive layout (note the space in "Furkan Pala"):
#   gdrive:EquitableEdu/Furkan Pala/experiment_outputs/<dir-name>/
# Local layout mirrors it under:
#   /vol/bitbucket/fp223/EquitableEdu/experiment_outputs/<dir-name>/
#
# Usage:
#   slurm/sync_to_drive.sh fold2                    # push fold2 dir
#   slurm/sync_to_drive.sh fold2 fold3              # multiple folds
#   slurm/sync_to_drive.sh --dry-run fold2          # preview only
#   slurm/sync_to_drive.sh --pull fold2             # Drive -> local instead
#   slurm/sync_to_drive.sh --all                    # push entire experiment_outputs/
#   slurm/sync_to_drive.sh --path experiment_outputs/foo  # arbitrary repo-relative dir

set -euo pipefail

REPO=/vol/bitbucket/fp223/EquitableEdu
LOCAL_OUTPUTS_REL="experiment_outputs"
DRIVE_OUTPUTS="gdrive:EquitableEdu/Furkan Pala/experiment_outputs"
RCLONE="${HOME}/.local/bin/rclone"

if [[ ! -x "$RCLONE" ]]; then
    echo "ERROR: rclone not found at $RCLONE" >&2
    exit 1
fi
if ! "$RCLONE" listremotes 2>/dev/null | grep -q '^gdrive:'; then
    echo "ERROR: rclone remote 'gdrive' is not configured." >&2
    echo "Run: rclone config   (see README/notes for the headless OAuth flow)" >&2
    exit 1
fi

DRY_RUN=()
DIRECTION="push"
# Each entry is "<local-rel-path>::<drive-rel-under-experiment_outputs-or-empty>"
# Empty drive-rel means: sync local_dir directly to ${DRIVE_OUTPUTS}/<basename>.
JOBS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run|-n) DRY_RUN=(--dry-run); shift ;;
        --pull)       DIRECTION="pull"; shift ;;
        --all)
            JOBS+=("${LOCAL_OUTPUTS_REL}::__ROOT__")
            shift
            ;;
        --path)
            # Caller-supplied relative path; mirrored 1:1 on Drive (under experiment_outputs/)
            JOBS+=("$2::$(basename "$2")")
            shift 2
            ;;
        fold[0-9]*)
            name="unifiedfl_fp_federated_experiment_${1}"
            JOBS+=("${LOCAL_OUTPUTS_REL}/${name}::${name}")
            shift
            ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ ${#JOBS[@]} -eq 0 ]]; then
    echo "No paths given. Try: $0 fold2 fold3   |   $0 --all" >&2
    exit 2
fi

for spec in "${JOBS[@]}"; do
    local_rel="${spec%%::*}"
    drive_sub="${spec##*::}"
    local_dir="${REPO}/${local_rel}"
    if [[ "${drive_sub}" == "__ROOT__" ]]; then
        remote_dir="${DRIVE_OUTPUTS}"
    else
        remote_dir="${DRIVE_OUTPUTS}/${drive_sub}"
    fi

    if [[ "$DIRECTION" == "push" ]]; then
        if [[ ! -d "$local_dir" ]]; then
            echo "SKIP (no local dir): $local_dir" >&2
            continue
        fi
        echo "PUSH  $local_dir  ->  $remote_dir"
        "$RCLONE" copy "$local_dir" "$remote_dir" \
            --transfers 4 --checkers 8 \
            --progress \
            "${DRY_RUN[@]}"
    else
        mkdir -p "$local_dir"
        echo "PULL  $remote_dir  ->  $local_dir"
        "$RCLONE" copy "$remote_dir" "$local_dir" \
            --transfers 4 --checkers 8 \
            --progress \
            "${DRY_RUN[@]}"
    fi
done
