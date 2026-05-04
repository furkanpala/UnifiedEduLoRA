"""
Step 7 — Run individual (no-federation) LoRA training for all three clients.

Each client trains independently on its own data. This is the baseline that the
federated run (step 8) is compared against.

Client assignments:
  0 → MIT notes    → facebook/bart-base   (bart,  q_proj v_proj, d_model=768)
  1 → Stanford     → google/flan-t5-base  (t5,    q v,           d_model=768)
  2 → Papers       → allenai/led-base-16384 (led, q_proj v_proj, d_model=768)

All clients use topic conditioning and are evaluated on:
  - their local validation set  → metrics_val.json
  - the global test set         → metrics_global_test.json

Usage (Colab terminal):
    python experiments/07_run_individual_baselines_fed_exp.py \\
        --splits-dir /content/drive/MyDrive/unifiedfl_fed_experiment/splits \\
        --output-dir /content/drive/MyDrive/unifiedfl_fed_experiment/individual

Re-running skips clients whose metrics_val.json already exists (pass --force
to override).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_DIR = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, str(Path(__file__).parent))

from _common import load_openai_key, default_key_search_paths

# ── Client specs ──────────────────────────────────────────────────────────────
# (client_id, model_hf_id, family, lora_targets)
CLIENT_SPECS = [
    (0, "facebook/bart-base",        "bart", ["q_proj", "v_proj"]),
    (1, "google/flan-t5-base",        "t5",   ["q", "v"]),
    (2, "allenai/led-base-16384",     "led",  ["q_proj", "v_proj"]),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--splits-dir",  required=True,
                   help="Splits directory from step 6.")
    p.add_argument("--output-dir",  required=True,
                   help="Root output directory for individual training runs.")
    p.add_argument("--drive-dir",   default=None)
    p.add_argument("--clients",     nargs="+", type=int, default=[0, 1, 2],
                   help="Which clients to train (default: all three).")
    p.add_argument("--force",       action="store_true",
                   help="Re-train even if metrics_val.json already exists.")

    # Training hyperparams — kept consistent with federated run (step 8)
    p.add_argument("--num-epochs",    type=int,   default=100)
    p.add_argument("--batch-size",    type=int,   default=4)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--warmup-ratio",  type=float, default=0.1)
    p.add_argument("--grad-clip",     type=float, default=1.0)
    p.add_argument("--patience",      type=int,   default=15)
    p.add_argument("--lora-r",        type=int,   default=16)
    p.add_argument("--lora-alpha",    type=int,   default=32)
    p.add_argument("--lora-dropout",  type=float, default=0.1)
    p.add_argument("--max-input-len", type=int,   default=512)
    p.add_argument("--max-target-len",type=int,   default=128)

    p.add_argument("--openai-api-key", default=None,
                   help="Not passed to train_client.py (fast eval used by default).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve OpenAI key — only used if full eval is wanted; fast eval is default
    # (no --openai-api-key passed to child, consistent with step 3 default)
    api_key, _ = load_openai_key(*default_key_search_paths(REPO_DIR, args.drive_dir))

    train_script = str(Path(REPO_DIR) / "unifiedfl" / "train_client.py")
    global_test_file = str(Path(args.splits_dir) / "global_test.json")

    summary: dict[int, int] = {}

    for cid, model, family, targets in CLIENT_SPECS:
        if cid not in args.clients:
            continue

        # train_client.py outputs to {output_dir}/{conditioning}/client_{id}/fold{fold}/
        metrics_path = (
            Path(args.output_dir) / "topic"
            / f"client_{cid}" / "fold1" / "metrics_val.json"
        )
        if metrics_path.exists() and not args.force:
            print(f"\n[skip] client {cid} — already has metrics_val.json. "
                  f"Pass --force to rerun.")
            summary[cid] = 0
            continue

        cmd = [
            "python", "-u", train_script,
            "--client-id",        str(cid),
            "--fold",             "1",
            "--model",            model,
            "--family",           family,
            "--targets",          *targets,
            "--splits-dir",       args.splits_dir,
            "--output-dir",       args.output_dir,
            "--conditioning",     "topic",
            "--num-epochs",       str(args.num_epochs),
            "--batch-size",       str(args.batch_size),
            "--lr",               str(args.lr),
            "--warmup-ratio",     str(args.warmup_ratio),
            "--grad-clip",        str(args.grad_clip),
            "--patience",         str(args.patience),
            "--lora-r",           str(args.lora_r),
            "--lora-alpha",       str(args.lora_alpha),
            "--lora-dropout",     str(args.lora_dropout),
            "--max-input-len",    str(args.max_input_len),
            "--max-target-len",   str(args.max_target_len),
            "--no-heavy",         # fast eval only
            "--global-test-file", global_test_file,
        ]

        print(f"\n{'─'*70}")
        print(f"  INDIVIDUAL  client={cid}  model={model}")
        print(f"{'─'*70}")
        print("  " + " ".join(cmd), flush=True)

        t0 = time.time()
        rc = subprocess.call(cmd)
        elapsed = time.time() - t0
        summary[cid] = rc
        print(f"\n  → finished client {cid} in {elapsed:.1f}s (exit {rc})", flush=True)
        if rc != 0:
            print(f"  WARNING: training failed for client {cid}")

    print("\n" + "=" * 70)
    print("  INDIVIDUAL TRAINING SUMMARY")
    print("=" * 70)
    for cid, rc in summary.items():
        status = "OK" if rc == 0 else f"FAILED ({rc})"
        print(f"  client {cid}: {status}")


if __name__ == "__main__":
    main()
