"""
Recover an interrupted train_client.py run by re-running only the
post-training evaluation step on adapters already saved to disk.

Use case: training finished and best/lora_model + final/lora_model were
saved, but the Colab session died during eval, so the
results/{best,final}/metrics_val.json files are missing. Without those
files, experiments/03_run_three_conditionings.py's resume-skip won't kick
in and the orchestrator will re-train the fold from scratch.

This script:
  1. Loads val (and optionally global_test) samples from the splits dir.
  2. Builds a fresh ClientModel for the requested family/model.
  3. Loads the FINAL adapter from {fold_dir}/final/lora_model and runs
     _run_full_eval -> {fold_dir}/results/final/metrics_val.json.
  4. Loads the BEST adapter from {fold_dir}/best/lora_model and runs
     _run_full_eval -> {fold_dir}/results/best/metrics_val.json.

Once both metrics_val.json files are written, the orchestrator's skip-check
treats the fold as complete on the next run.

Usage (Colab):
    python experiments/11_recover_eval_only.py \\
        --output-dir   /content/drive/MyDrive/unifiedfl_mit_experiment/outputs \\
        --splits-dir   /content/drive/MyDrive/unifiedfl_mit_experiment/splits \\
        --client-id    2 \\
        --fold         2 \\
        --conditioning topic \\
        --model        allenai/led-base-16384 \\
        --family       led \\
        --targets      q_proj v_proj
        # Add --skip-final or --skip-best to run only one side.
        # Add --no-heavy / --openai-api-key to control metric depth
        # (defaults to comprehensive eval, matching 03's default).
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR / "unifiedfl"))
sys.path.insert(0, str(Path(__file__).parent))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

# Re-use the exact same eval helpers + checkpoint loader the original training
# script uses, so the recovered metrics are bit-for-bit comparable to the
# fold1 metrics that were written during the un-interrupted run.
from train_client import (
    _ensure_nltk_punkt,
    _evaluate,             # noqa: F401  (used inside _run_full_eval)
    _load_best,
    _load_split,
    _run_full_eval,
    set_seeds,
)
from models.client_model import ClientModel

from _common import default_key_search_paths, load_openai_key


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir",    required=True)
    p.add_argument("--splits-dir",    required=True)
    p.add_argument("--client-id",     type=int, required=True)
    p.add_argument("--fold",          type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--conditioning",  required=True,
                   choices=["baseline", "topic", "bloom"])

    # Model spec — must match what was used at training time.
    p.add_argument("--model",   required=True, help="HuggingFace model ID")
    p.add_argument("--family",  required=True,
                   choices=["t5", "bart", "led", "pegasus_x", "marian", "prophetnet"])
    p.add_argument("--targets", nargs="+", required=True)

    # LoRA — must also match training-time config (same r/alpha/dropout)
    p.add_argument("--lora-r",       type=int,   default=16)
    p.add_argument("--lora-alpha",   type=int,   default=32)
    p.add_argument("--lora-dropout", type=float, default=0.1)

    # Generation / eval — keep defaults aligned with train_client.py
    p.add_argument("--batch-size",                type=int,   default=4,
                   help="Batch size used by _evaluate during generation. "
                        "Same default as train_client.py.")
    p.add_argument("--max-input-len",             type=int,   default=512)
    p.add_argument("--max-target-len",            type=int,   default=128)
    p.add_argument("--eval-num-return-sequences", type=int,   default=5)
    p.add_argument("--eval-num-beams",            type=int,   default=10)
    p.add_argument("--eval-num-beam-groups",      type=int,   default=5)
    p.add_argument("--eval-diversity-penalty",    type=float, default=1.0)
    p.add_argument("--seed",                      type=int,   default=42)
    p.add_argument("--device",                    default="cuda")

    # Comprehensive-eval knobs (mirror train_client.py)
    p.add_argument("--no-heavy", action="store_true")
    p.add_argument("--blooms-model", default="cip29/bert-blooms-taxonomy-classifier")
    p.add_argument("--openai-api-key", default=None)
    p.add_argument("--drive-dir",      default=None,
                   help="Drive root (used to find an openai_api_key file there)")

    # Optional global-test eval (only if a global_test.json exists in splits/)
    p.add_argument("--global-test-file", default=None,
                   help="Path to a JSON file of flat QA samples; if provided, "
                        "metrics_global_test.json is also written for each "
                        "checkpoint.")
    p.add_argument("--eval-local-test", action="store_true",
                   help="Also evaluate on the per-client local test set "
                        "(client_{ID}_test.json from --splits-dir) and write "
                        "metrics_test.json next to metrics_val.json. Useful "
                        "for the indiv-vs-fed comparison where the individual "
                        "training never saw the test set.")
    p.add_argument("--skip-val-eval", action="store_true",
                   help="Don't re-run the val eval (e.g., metrics_val.json "
                        "already exists from training). Combined with "
                        "--eval-local-test and --global-test-file, this lets "
                        "you back-fill only the held-out splits.")

    # Toggles for skipping one side of the work
    p.add_argument("--skip-final", action="store_true",
                   help="Don't re-run the FINAL-checkpoint eval "
                        "(metrics_val.json already present).")
    p.add_argument("--skip-best",  action="store_true",
                   help="Don't re-run the BEST-checkpoint eval "
                        "(metrics_val.json already present).")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if results/{best,final}/metrics_val.json "
                        "already exist (default: skip those automatically).")

    return p.parse_args()


def _resolve_openai_key(args: argparse.Namespace) -> str | None:
    if args.no_heavy:
        return None
    if args.openai_api_key and args.openai_api_key.startswith("sk-"):
        os.environ["OPENAI_API_KEY"] = args.openai_api_key
        print("OpenAI key from --openai-api-key arg")
        return args.openai_api_key
    api_key, source = load_openai_key(
        *default_key_search_paths(str(REPO_DIR), args.drive_dir)
    )
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        print(f"OpenAI key loaded from: {source}")
    else:
        print("WARNING: no OpenAI key — Answer Relevancy and LLM-judge metrics "
              "will be skipped.")
    return api_key


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)

    args.openai_api_key = _resolve_openai_key(args)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()

    output_dir  = Path(args.output_dir)
    fold_dir    = (output_dir / args.conditioning
                   / f"client_{args.client_id}" / f"fold{args.fold}")
    best_dir    = fold_dir / "best"
    final_dir   = fold_dir / "final"
    results_dir = fold_dir / "results"

    if not best_dir.exists() or not (best_dir / "lora_model").exists():
        raise FileNotFoundError(
            f"BEST adapter not found at {best_dir}/lora_model. "
            "Was training interrupted before the best checkpoint was saved? "
            "If so, you must re-train the fold from scratch — the recovery "
            "script can only re-run the eval step."
        )
    if not final_dir.exists() or not (final_dir / "lora_model").exists():
        raise FileNotFoundError(
            f"FINAL adapter not found at {final_dir}/lora_model. "
            "Training did not finish — re-run the fold instead."
        )

    # Comprehensive metrics use NLTK sent_tokenize (faithfulness, qafacteval).
    # train_client.py ensures punkt_tab is downloaded before eval; mirror that.
    if not args.no_heavy:
        _ensure_nltk_punkt()

    splits_dir   = Path(args.splits_dir)
    val_samples  = _load_split(splits_dir, args.client_id, args.fold, "val")

    # Optional: per-client local test (single test file per client, no fold suffix).
    local_test_samples = None
    if args.eval_local_test:
        import json as _json
        test_path = splits_dir / f"client_{args.client_id}_test.json"
        if test_path.exists():
            local_test_samples = _json.loads(test_path.read_text(encoding="utf-8"))
            print(f"Local test set: {len(local_test_samples)} samples ({test_path})")
        else:
            print(f"WARNING: --eval-local-test set but {test_path} not found — skipping.")

    global_test_samples = None
    if args.global_test_file:
        gt_path = Path(args.global_test_file)
        if gt_path.exists():
            import json
            global_test_samples = json.loads(gt_path.read_text(encoding="utf-8"))
            print(f"Global test set: {len(global_test_samples)} samples")
        else:
            print(f"WARNING: --global-test-file not found: {gt_path}")

    print(f"\nRecovering eval for {args.conditioning}/client_{args.client_id}/fold{args.fold}")
    print(f"  fold_dir   = {fold_dir}")
    print(f"  val        = {len(val_samples)} samples")
    print(f"  device     = {device}  use_amp={use_amp}")

    # Build a fresh model — same architecture as at training time.
    print(f"\nBuilding ClientModel({args.model}) …")
    client_model = ClientModel(
        model_name=args.model,
        model_family=args.family,
        lora_target_modules=args.targets,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        device=device,
    )

    def _eval_one_ckpt(ckpt_label: str, ckpt_dir: Path):
        out_dir = results_dir / ckpt_label
        # Per-split skip-logic: a side is run only if it's both requested AND
        # missing (or --force). That makes back-fill calls (e.g. --eval-local-test
        # --skip-val-eval) cheap when val was already produced during training.
        plan: list[tuple[str, list]] = []
        if not args.skip_val_eval:
            plan.append(("val", val_samples))
        if local_test_samples is not None:
            plan.append(("test", local_test_samples))
        if global_test_samples is not None:
            plan.append(("global_test", global_test_samples))
        # Drop splits whose metrics file is already present unless --force
        plan = [(n, s) for (n, s) in plan
                if args.force or not (out_dir / f"metrics_{n}.json").exists()]
        if not plan:
            print(f"\n[skip] {ckpt_label.upper()} — all requested splits already evaluated.")
            return
        print(f"\n{'=' * 60}\n  Loading {ckpt_label.upper()} checkpoint\n{'=' * 60}")
        if device.type == "cuda":
            client_model.model.to(device)
            gc.collect()
            torch.cuda.empty_cache()
        _load_best(client_model, ckpt_dir)
        print(f"\n{'=' * 60}\n  Evaluating {ckpt_label.upper()} on {[n for n,_ in plan]}\n{'=' * 60}")
        for split_name, samples in plan:
            _run_full_eval(
                client_model, samples, args, device, use_amp,
                out_dir=out_dir, split_name=split_name,
            )

    # ── FINAL ────────────────────────────────────────────────────────────────
    if args.skip_final:
        print(f"\n[skip] FINAL eval — requested via --skip-final")
    else:
        _eval_one_ckpt("final", final_dir)

    # ── BEST ─────────────────────────────────────────────────────────────────
    if args.skip_best:
        print(f"\n[skip] BEST eval — requested via --skip-best")
    else:
        _eval_one_ckpt("best", best_dir)

    print(f"\nDone. Per-checkpoint metrics → {results_dir}/{{best,final}}/metrics_*.json")


if __name__ == "__main__":
    main()
