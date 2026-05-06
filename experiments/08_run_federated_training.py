"""
Step 8 — Run graph-based federated training across all three clients.

33 rounds × 3 local epochs ≈ 99 effective training passes, matching the
individual baseline (step 7, 100 epochs with patience).

Client assignments:
  0 → MIT notes    → facebook/bart-base      (bart, q_proj v_proj, d_model=768)
  1 → Stanford     → google/flan-t5-base     (t5,   q v,           d_model=768)
  2 → Papers       → allenai/led-base-16384  (led,  q_proj v_proj, d_model=768)

All clients use topic conditioning. Quantitative evaluation runs every
eval_every_n rounds on each client's local test set AND the global test set.
Final per-client metrics are written to final_metrics_per_client.json.

Usage (Colab terminal):
    python experiments/08_run_federated_training.py \\
        --splits-dir /content/drive/MyDrive/unifiedfl_fed_experiment/splits \\
        --output-dir /content/drive/MyDrive/unifiedfl_fed_experiment/federated
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

from _common import default_key_search_paths, load_openai_key


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--splits-dir",  required=True)
    p.add_argument("--output-dir",  required=True)
    p.add_argument("--fold", type=int, default=0, choices=[0, 1, 2, 3],
                   help="If >=1, use the fold-specific splits produced by "
                        "experiments/02_split_3fold.py (matches the individual "
                        "experiment's split layout). 0 = legacy flat layout.")
    p.add_argument("--conditioning", default="topic",
                   choices=["baseline", "topic", "bloom"],
                   help="Prompt conditioning used for both training and eval.")
    p.add_argument("--snapshot-metric", choices=["rouge_l", "val_loss"],
                   default="rouge_l",
                   help="Metric used to pick the best snapshot. rouge_l matches "
                        "train_client.py's early-stop metric.")

    # Federation
    p.add_argument("--num-rounds",   type=int, default=33)
    p.add_argument("--local-epochs", type=int, default=3)
    p.add_argument("--eval-every-n", type=int, default=5,
                   help="Evaluate every N rounds (default 5). "
                        "The final round is always evaluated.")

    # Consistent hyperparams (match individual training in step 7)
    p.add_argument("--batch-size",    type=int,   default=4)
    p.add_argument("--lr-lora",       type=float, default=3e-4)
    p.add_argument("--lr-gnn",        type=float, default=1e-3)
    p.add_argument("--lr-film",       type=float, default=1e-3)
    p.add_argument("--warmup-ratio",  type=float, default=0.1)
    p.add_argument("--grad-clip",     type=float, default=1.0)
    p.add_argument("--lora-r",        type=int,   default=16)
    p.add_argument("--lora-alpha",    type=int,   default=32)
    p.add_argument("--lora-dropout",  type=float, default=0.1)
    p.add_argument("--max-input-len", type=int,   default=512)
    p.add_argument("--max-target-len",type=int,   default=128)

    # GNN / FiLM
    p.add_argument("--gnn-hidden",  type=int,   default=64)
    p.add_argument("--gnn-heads",   type=int,   default=4)
    p.add_argument("--gnn-layers",  type=int,   default=3)
    p.add_argument("--gnn-dropout", type=float, default=0.1)
    p.add_argument("--film-hidden", type=int,   default=128)
    p.add_argument("--film-alpha",  type=float, default=0.0)

    p.add_argument("--checkpoint-every-round", type=int, default=5)
    p.add_argument("--resume-from-round",      type=int, default=0)
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--device", default="cuda")

    # Post-training comprehensive eval (mirrors 03_run_three_conditionings.py).
    # Default: comprehensive runs on best snapshot AND final-round state.
    # Pass --fast-eval to skip everything except the lightweight 3-metric
    # combo + local Bloom's classifier.
    p.add_argument("--fast-eval", action="store_true",
                   help="Skip heavy (UnifiedQA / DeBERTa NLI) and LLM-based "
                        "metrics in the post-training eval. Passes --no-heavy "
                        "to train_federated.py and skips OpenAI key loading.")
    p.add_argument("--no-comprehensive-eval", action="store_true",
                   help="Disable the post-training per-client comprehensive "
                        "eval block entirely (only the existing per-round "
                        "monitoring eval runs).")
    p.add_argument("--openai-api-key", default=None,
                   help="Override; otherwise loaded from env / repo / drive.")
    p.add_argument("--drive-dir", default=None,
                   help="Drive root (used to find an openai_api_key file there).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve the OpenAI key for comprehensive eval (mirrors 03's logic).
    api_key: str | None = None
    if not args.fast_eval and not args.no_comprehensive_eval:
        if args.openai_api_key and args.openai_api_key.startswith("sk-"):
            api_key, source = args.openai_api_key, "<--openai-api-key arg>"
        else:
            api_key, source = load_openai_key(
                *default_key_search_paths(REPO_DIR, args.drive_dir)
            )
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            print(f"OpenAI key loaded from: {source}")
        else:
            print("WARNING: no OpenAI API key found — Answer Relevancy + LLM "
                  "judges will be skipped. Place an 'openai_api_key' file at "
                  "the repo root, set OPENAI_API_KEY, or pass --openai-api-key sk-... "
                  "to enable them.")
    elif args.fast_eval:
        print("Fast eval mode: skipping heavy + LLM-based metrics.")
    else:
        print("Comprehensive eval disabled (--no-comprehensive-eval).")

    train_script = str(Path(REPO_DIR) / "unifiedfl" / "train_federated.py")

    # Client specs: "id:model:family:targets:d_model"
    client_specs = [
        "0:facebook/bart-base:bart:q_proj,v_proj:768",
        "1:google/flan-t5-base:t5:q,v:768",
        "2:allenai/led-base-16384:led:q_proj,v_proj:768",
    ]

    cmd = [
        "python", "-u", train_script,
        "--splits-dir",             args.splits_dir,
        "--output-dir",             args.output_dir,
        "--fold",                   str(args.fold),
        "--conditioning",           args.conditioning,
        "--snapshot-metric",        args.snapshot_metric,
        "--no-qual-eval",           # qualitative eval less meaningful with topic conditioning
        "--num-rounds",             str(args.num_rounds),
        "--local-epochs",           str(args.local_epochs),
        "--eval-every-n",           str(args.eval_every_n),
        "--batch-size",             str(args.batch_size),
        "--lr-lora",                str(args.lr_lora),
        "--lr-gnn",                 str(args.lr_gnn),
        "--lr-film",                str(args.lr_film),
        "--warmup-ratio",           str(args.warmup_ratio),
        "--grad-clip",              str(args.grad_clip),
        "--max-input-len",          str(args.max_input_len),
        "--max-target-len",         str(args.max_target_len),
        "--lora-r",                 str(args.lora_r),
        "--lora-alpha",             str(args.lora_alpha),
        "--lora-dropout",           str(args.lora_dropout),
        "--gnn-hidden",             str(args.gnn_hidden),
        "--gnn-heads",              str(args.gnn_heads),
        "--gnn-layers",             str(args.gnn_layers),
        "--gnn-dropout",            str(args.gnn_dropout),
        "--film-hidden",            str(args.film_hidden),
        "--film-alpha",             str(args.film_alpha),
        "--checkpoint-every-round", str(args.checkpoint_every_round),
        "--resume-from-round",      str(args.resume_from_round),
        "--seed",                   str(args.seed),
        "--device",                 args.device,
    ]
    if args.fast_eval:
        cmd.append("--no-heavy")
    if args.no_comprehensive_eval:
        cmd.append("--no-comprehensive-eval")
    if api_key:
        cmd += ["--openai-api-key", api_key]
    for spec in client_specs:
        cmd += ["--client", spec]

    print(f"\n{'='*70}")
    print("  FEDERATED TRAINING")
    print(f"  {len(client_specs)} clients × {args.num_rounds} rounds × {args.local_epochs} local epochs")
    print(f"  ≈ {args.num_rounds * args.local_epochs} effective training passes per client")
    print(f"{'='*70}")
    # Don't echo the API key
    safe = ["sk-***" if (api_key and c == api_key) else c for c in cmd]
    print("  " + " ".join(safe), flush=True)

    t0 = time.time()
    rc = subprocess.call(cmd)
    elapsed = time.time() - t0

    print(f"\n{'='*70}")
    print(f"  Federated training finished in {elapsed:.1f}s (exit {rc})")
    if rc != 0:
        print("  WARNING: training process exited with non-zero code.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
