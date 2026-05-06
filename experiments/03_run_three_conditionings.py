"""
Step 3 — Train BART-base on the MIT data under three conditioning regimes,
each across 3 CV folds (9 total runs).

train_client.py is invoked as a child process WITHOUT capturing stdout/stderr,
so its log lines appear live in the terminal. Re-running this script skips
folds that already have a metrics_val.json (resume-friendly).

Usage (Colab terminal):
    python experiments/03_run_three_conditionings.py \\
        --splits-dir /content/drive/MyDrive/unifiedfl_mit_experiment/splits \\
        --output-dir /content/drive/MyDrive/unifiedfl_mit_experiment/outputs \\
        --client-id 0

Override defaults with --conditionings, --folds, --num-epochs, etc.
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--splits-dir",  required=True)
    p.add_argument("--output-dir",  required=True)
    p.add_argument("--client-id",   type=int, default=0)
    p.add_argument("--drive-dir",   default=None,
                   help="Drive root (used to find an openai_api_key file there)")

    # Selection
    p.add_argument("--conditionings", nargs="+",
                   default=["baseline", "topic", "bloom"],
                   choices=["baseline", "topic", "bloom"])
    p.add_argument("--folds",          nargs="+", type=int,
                   default=[1, 2, 3])
    p.add_argument("--force",          action="store_true",
                   help="Run even if metrics_val.json already exists")

    # Model
    p.add_argument("--model",   default="facebook/bart-base")
    p.add_argument("--family",  default="bart")
    p.add_argument("--targets", nargs="+", default=["q_proj", "v_proj"])

    # Training hyperparams (forwarded to train_client.py)
    p.add_argument("--num-epochs",       type=int,   default=100)
    p.add_argument("--batch-size",       type=int,   default=4)
    p.add_argument("--lr",               type=float, default=3e-4)
    p.add_argument("--patience",         type=int,   default=10)
    p.add_argument("--preview-every",    type=int,   default=5,
                   help="Generate a sample QA from the val set every N epochs "
                        "(matches train_client.py default). Pass 0 to disable.")
    p.add_argument("--checkpoint-every", type=int,   default=5,
                   help="Save a full LoRA + optimizer + scheduler checkpoint "
                        "every N epochs (matches train_client.py default). "
                        "Pass 0 to disable periodic checkpointing.")
    p.add_argument("--fast-eval",        action="store_true",
                   help="Skip heavy (UnifiedQA/DeBERTa) and LLM-based metrics. "
                        "Default: comprehensive evaluation (requires OpenAI key for "
                        "LLM-based metrics; passes --no-heavy to train_client.py "
                        "when this flag is set).")
    p.add_argument("--openai-api-key",   default=None,
                   help="Override; otherwise loaded from env / repo / drive.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve the API key — used for comprehensive eval (default)
    api_key = None
    if not args.fast_eval:
        if args.openai_api_key and args.openai_api_key.startswith("sk-"):
            api_key, source = args.openai_api_key, "<--openai-api-key arg>"
        else:
            api_key, source = load_openai_key(*default_key_search_paths(REPO_DIR, args.drive_dir))
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            print(f"OpenAI key loaded from: {source}")
        else:
            print("WARNING: no OpenAI API key found — LLM-based metrics will be skipped. "
                  "Place an 'openai_api_key' file at the repo root, set OPENAI_API_KEY, "
                  "or pass --openai-api-key sk-... to enable them.")
    else:
        print("Fast eval mode: running ROUGE-L, BLEU-4, BERTScore + local Bloom's "
              "classifier only. Omit --fast-eval to enable comprehensive metrics.")

    train_script = str(Path(REPO_DIR) / "unifiedfl" / "train_client.py")

    summary: dict[tuple[str, int], int] = {}

    for conditioning in args.conditionings:
        for fold in args.folds:
            # train_client.py writes to {output_dir}/{conditioning}/client_{id}/fold{N}/
            # and saves per-checkpoint metrics under results/{best,final}/.
            fold_dir = (Path(args.output_dir) / conditioning
                        / f"client_{args.client_id}" / f"fold{fold}")
            new_metrics = fold_dir / "results" / "best" / "metrics_val.json"
            old_metrics = fold_dir / "metrics_val.json"   # legacy layout
            if (new_metrics.exists() or old_metrics.exists()) and not args.force:
                print(f"\n[skip] {conditioning} fold{fold} — already has metrics_val.json. "
                      f"Pass --force to rerun.")
                summary[(conditioning, fold)] = 0
                continue

            cmd = [
                "python", "-u", train_script,
                "--client-id",        str(args.client_id),
                "--fold",             str(fold),
                "--model",            args.model,
                "--family",           args.family,
                "--targets",          *args.targets,
                "--splits-dir",       args.splits_dir,
                "--output-dir",       args.output_dir,
                "--num-epochs",       str(args.num_epochs),
                "--batch-size",       str(args.batch_size),
                "--lr",               str(args.lr),
                "--patience",         str(args.patience),
                "--preview-every",    str(args.preview_every),
                "--checkpoint-every", str(args.checkpoint_every),
                "--conditioning",     conditioning,
            ]
            if args.fast_eval:
                cmd.append("--no-heavy")
            if api_key:
                cmd += ["--openai-api-key", api_key]

            print(f"\n{'─' * 70}")
            print(f"  RUNNING  conditioning={conditioning}  fold={fold}")
            print(f"{'─' * 70}")
            # Don't echo the API key
            safe = ["sk-***" if c == api_key else c for c in cmd]
            print("  " + " ".join(safe), flush=True)

            t0 = time.time()
            # No capture: child stdout/stderr go directly to our terminal.
            # python -u + flush=True keep the output unbuffered.
            rc = subprocess.call(cmd)
            elapsed = time.time() - t0
            summary[(conditioning, fold)] = rc
            print(f"\n  → finished {conditioning} fold{fold} in {elapsed:.1f}s (exit {rc})",
                  flush=True)
            if rc != 0:
                print(f"  WARNING: training failed for {conditioning} fold{fold}")

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for (cond, fold), rc in summary.items():
        status = "OK" if rc == 0 else f"FAILED ({rc})"
        print(f"  {cond:<10} fold{fold}: {status}")


if __name__ == "__main__":
    main()
