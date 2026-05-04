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
import subprocess
import sys
import time
from pathlib import Path

REPO_DIR = str(Path(__file__).resolve().parent.parent)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--splits-dir",  required=True)
    p.add_argument("--output-dir",  required=True)

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
    return p.parse_args()


def main() -> None:
    args = parse_args()

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
        "--conditioning",           "topic",
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
    for spec in client_specs:
        cmd += ["--client", spec]

    print(f"\n{'='*70}")
    print("  FEDERATED TRAINING")
    print(f"  {len(client_specs)} clients × {args.num_rounds} rounds × {args.local_epochs} local epochs")
    print(f"  ≈ {args.num_rounds * args.local_epochs} effective training passes per client")
    print(f"{'='*70}")
    print("  " + " ".join(cmd), flush=True)

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
