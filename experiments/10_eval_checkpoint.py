"""
Step 10 — Evaluate any federated checkpoint and save metrics for comparison.

Loads a saved snapshot of all 3 clients (LoRA + FiLM + GNN), runs quantitative
evaluation on every client's local test set AND the global test set, and writes
the results in the same format as step 8's final_metrics_per_client.json — so
step 9 can compare it against the individual baseline.

Use it to:
  • evaluate the ACTUAL final-round state           → --checkpoint-dir .../fed_final_round
  • evaluate a specific intermediate round          → --checkpoint-dir .../fed_checkpoints/round_15
  • re-evaluate the best snapshot (sanity check)    → --checkpoint-dir .../fed_final

The checkpoint directory must contain client_0/, client_1/, client_2/ subdirs
(each with a lora_model/ folder and film.pt) and a top-level gnn.pt.

Usage:
    python experiments/10_eval_checkpoint.py \\
        --checkpoint-dir /content/drive/MyDrive/unifiedfl_fed_experiment/federated/fed_final_round \\
        --splits-dir     /content/drive/MyDrive/unifiedfl_fed_experiment/splits \\
        --output-dir     /content/drive/MyDrive/unifiedfl_fed_experiment/federated_finalround \\
        --label          "final_round"

    # Then compare:
    python experiments/09_compare_results.py \\
        --indiv-dir /content/drive/MyDrive/unifiedfl_fed_experiment/individual \\
        --fed-dir   /content/drive/MyDrive/unifiedfl_fed_experiment/federated_finalround
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR / "unifiedfl"))

from config.config import Config, ClientConfig, LoRAConfig, GNNConfig, FiLMConfig
from evaluation.evaluator import Evaluator
from federation.client import FederatedClient
from models.client_model import ClientModel
from models.film_adapter import FiLMAdapter
from models.gnn import ArchitectureGNN
from models.graph_constructor import build_graph


# ── Default client assignments — match steps 7 and 8 ─────────────────────────
DEFAULT_CLIENT_SPECS = [
    "0:facebook/bart-base:bart:q_proj,v_proj:768",
    "1:google/flan-t5-base:t5:q,v:768",
    "2:allenai/led-base-16384:led:q_proj,v_proj:768",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", required=True,
                   help="Directory containing client_0/, client_1/, client_2/ and gnn.pt")
    p.add_argument("--splits-dir",     required=True,
                   help="Splits directory (must contain client_*_test.json + global_test.json)")
    p.add_argument("--output-dir",     required=True,
                   help="Where to write final_metrics_per_client.json")
    p.add_argument("--label",          default=None,
                   help="Optional label saved in the metrics file for clarity.")

    # Client specs — default to the federated experiment's fixed assignments
    p.add_argument("--clients", nargs="+", default=DEFAULT_CLIENT_SPECS,
                   help="id:model:family:targets:d_model. Defaults match step 8.")

    # Must match training
    p.add_argument("--conditioning",   default="topic",
                   choices=["baseline", "topic", "bloom"])
    p.add_argument("--max-input-len",  type=int,   default=512)
    p.add_argument("--max-target-len", type=int,   default=128)
    p.add_argument("--batch-size",     type=int,   default=4)

    # LoRA hyperparams (must match training so the adapter shapes line up)
    p.add_argument("--lora-r",         type=int,   default=16)
    p.add_argument("--lora-alpha",     type=int,   default=32)
    p.add_argument("--lora-dropout",   type=float, default=0.1)

    # GNN / FiLM (must match training)
    p.add_argument("--gnn-hidden",     type=int,   default=64)
    p.add_argument("--gnn-heads",      type=int,   default=4)
    p.add_argument("--gnn-dropout",    type=float, default=0.1)
    p.add_argument("--film-hidden",    type=int,   default=128)
    p.add_argument("--film-alpha",     type=float, default=0.0)

    p.add_argument("--device",         default="cuda")
    return p.parse_args()


def parse_client_spec(spec: str) -> ClientConfig:
    parts = spec.split(":", 4)
    if len(parts) != 5:
        sys.exit(f"Bad client spec: {spec!r}")
    cid, model, family, targets_str, d_model = parts
    return ClientConfig(
        client_id=int(cid),
        model_name=model,
        model_family=family,
        lora_target_modules=targets_str.split(","),
        d_model=int(d_model),
        data_path="",
    )


def load_split(splits_dir: Path, client_id: int, name: str) -> list:
    path = splits_dir / f"client_{client_id}_{name}.json"
    if not path.exists():
        sys.exit(f"Split file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_state_into_clients(clients: list, checkpoint_dir: Path, device: torch.device) -> None:
    """
    Restore LoRA + FiLM + GNN for every client from a snapshot directory.
    Uses set_peft_model_state_dict to overwrite the existing 'default' adapter
    (PEFT's load_adapter would create a new one).
    """
    from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict

    gnn_path = checkpoint_dir / "gnn.pt"
    if not gnn_path.exists():
        sys.exit(f"Missing gnn.pt in checkpoint: {checkpoint_dir}")
    gnn_state = torch.load(gnn_path, map_location=device)

    for client in clients:
        cdir = checkpoint_dir / f"client_{client.client_id}"
        if not cdir.exists():
            sys.exit(f"Missing client subdir: {cdir}")
        lora_state = load_peft_weights(str(cdir / "lora_model"), device=str(device))
        set_peft_model_state_dict(client.client_model.model, lora_state)
        client.film_adapter.load_state_dict(
            torch.load(cdir / "film.pt", map_location=device)
        )
        client.gnn.load_state_dict(gnn_state)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        sys.exit(f"Checkpoint directory not found: {checkpoint_dir}")

    splits_dir = Path(args.splits_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse client specs
    client_cfgs = [parse_client_spec(s) for s in args.clients]
    client_cfgs.sort(key=lambda c: c.client_id)

    cfg = Config(
        max_input_len=args.max_input_len,
        max_target_len=args.max_target_len,
        batch_size=args.batch_size,
        device=str(device),
        lora=LoRAConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout),
        gnn=GNNConfig(hidden=args.gnn_hidden, heads=args.gnn_heads, dropout=args.gnn_dropout),
        film=FiLMConfig(hidden=args.film_hidden, alpha_init=args.film_alpha),
        clients=client_cfgs,
        conditioning=args.conditioning,
    )

    # Load splits
    print("\nLoading splits ...")
    client_splits = {}
    for client_cfg in client_cfgs:
        cid = client_cfg.client_id
        client_splits[cid] = {
            "train": load_split(splits_dir, cid, "train"),
            "val":   load_split(splits_dir, cid, "val"),
            "test":  load_split(splits_dir, cid, "test"),
        }
        print(f"  Client {cid}: test={len(client_splits[cid]['test'])} samples")

    global_test_path = splits_dir / "global_test.json"
    if not global_test_path.exists():
        sys.exit(f"global_test.json not found: {global_test_path}")
    global_test = json.loads(global_test_path.read_text(encoding="utf-8"))
    print(f"  Global test: {len(global_test)} samples")

    # Build clients
    print("\nBuilding clients (initial weights — will be overwritten by checkpoint) ...")
    clients = []
    for client_cfg in client_cfgs:
        cid = client_cfg.client_id

        client_model = ClientModel(
            model_name=client_cfg.model_name,
            model_family=client_cfg.model_family,
            lora_target_modules=client_cfg.lora_target_modules,
            lora_r=cfg.lora.r,
            lora_alpha=cfg.lora.lora_alpha,
            lora_dropout=cfg.lora.lora_dropout,
            device=device,
        )
        graph_data = build_graph(
            peft_model=client_model.model,
            model_name=client_cfg.model_name,
            lora_alpha=cfg.lora.lora_alpha,
            lora_r=cfg.lora.r,
            device=device,
        )
        gnn = ArchitectureGNN(
            in_channels=cfg.gnn.in_channels,
            hidden=cfg.gnn.hidden,
            heads=cfg.gnn.heads,
            dropout=cfg.gnn.dropout,
        ).to(device)
        film_adapter = FiLMAdapter(
            d_model=client_cfg.d_model,
            film_hidden=cfg.film.hidden,
            alpha_init=cfg.film.alpha_init,
            model_family=client_cfg.model_family,
        ).to(device)

        fc = FederatedClient(
            client_id=cid,
            client_model=client_model,
            gnn=gnn,
            film_adapter=film_adapter,
            graph_data=graph_data,
            train_samples=client_splits[cid]["train"],
            val_samples=client_splits[cid]["val"],
            test_samples=client_splits[cid]["test"],
            device=device,
        )
        clients.append(fc)
        print(f"  Client {cid} ready ({client_cfg.model_name})")

    # Restore checkpoint
    print(f"\nLoading checkpoint from: {checkpoint_dir}")
    load_state_into_clients(clients, checkpoint_dir, device)
    print("  Checkpoint loaded.")

    # Evaluate
    print(f"\n{'=' * 60}")
    print(f"  Evaluating checkpoint: {args.label or checkpoint_dir.name}")
    print(f"{'=' * 60}")
    evaluator = Evaluator(clients, cfg, device)
    eval_results = evaluator.quantitative_eval(round_idx=0, global_test_samples=global_test)

    # Save in the same format as train_federated.py's final_metrics_per_client.json
    out_path = output_dir / "final_metrics_per_client.json"
    out_path.write_text(json.dumps({
        "checkpoint":         str(checkpoint_dir),
        "label":              args.label,
        "per_client_metrics": {str(cid): v for cid, v in eval_results.items()},
    }, indent=2))
    print(f"\nMetrics saved → {out_path}")
    print(f"\nNow compare against the individual baseline with:")
    print(f"  python experiments/09_compare_results.py \\")
    print(f"      --indiv-dir <individual_output_dir> \\")
    print(f"      --fed-dir   {output_dir}")


if __name__ == "__main__":
    main()
