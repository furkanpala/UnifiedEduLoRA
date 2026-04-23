"""
Run federated training across configurable clients.
Supports checkpointing per round and resuming, designed for Colab + Drive.

Client spec format:  "id:model_name:family:lora_targets:d_model"
  - lora_targets are comma-separated (no spaces)

Usage (Colab):
    # Mount Drive first:
    #   from google.colab import drive; drive.mount('/content/drive')

    python train_federated.py \
        --splits-dir /content/drive/MyDrive/federated_qa_gnn/outputs/splits \
        --client "0:google/flan-t5-small:t5:q,v:512" \
        --client "1:facebook/bart-base:bart:q_proj,v_proj:768" \
        --client "2:allenai/led-base-16384:led:q_proj,v_proj:768" \
        --output-dir /content/drive/MyDrive/federated_qa_gnn/outputs \
        --num-rounds 20 \
        --local-epochs 3

    # Resume from round 10 checkpoint:
    python train_federated.py ... --resume-from-round 10
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from config.config import Config, ClientConfig, LoRAConfig, GNNConfig, FiLMConfig
from evaluation.evaluator import Evaluator
from evaluation.metrics import compute_all_metrics
from federation.client import FederatedClient
from federation.server import FederatedServer
from models.client_model import ClientModel
from models.film_adapter import FiLMAdapter
from models.gnn import ArchitectureGNN
from models.graph_constructor import build_graph, refresh_graph_features
from training.trainer import LocalTrainer
from utils.logging_utils import JSONLogger, setup_logging


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Federated training with configurable clients")

    # Client specs
    p.add_argument(
        "--client", action="append", required=True,
        metavar="ID:MODEL:FAMILY:TARGETS:D_MODEL",
        help=(
            "Client spec, e.g. '0:google/flan-t5-small:t5:q,v:512'. "
            "Repeat for each client."
        ),
    )

    # Data
    p.add_argument("--splits-dir", default="outputs/splits",
                   help="Directory with split JSON files from split.py")

    # Federation / training
    p.add_argument("--num-rounds",     type=int,   default=20)
    p.add_argument("--local-epochs",   type=int,   default=3)
    p.add_argument("--batch-size",     type=int,   default=4)
    p.add_argument("--lr-lora",        type=float, default=3e-4)
    p.add_argument("--lr-gnn",         type=float, default=1e-3)
    p.add_argument("--lr-film",        type=float, default=1e-3)
    p.add_argument("--warmup-ratio",   type=float, default=0.1)
    p.add_argument("--grad-clip",      type=float, default=1.0)
    p.add_argument("--max-input-len",  type=int,   default=512)
    p.add_argument("--max-target-len", type=int,   default=128)
    p.add_argument("--eval-every-n",   type=int,   default=5)

    # LoRA
    p.add_argument("--lora-r",       type=int,   default=16)
    p.add_argument("--lora-alpha",   type=int,   default=32)
    p.add_argument("--lora-dropout", type=float, default=0.1)

    # GNN
    p.add_argument("--gnn-hidden",  type=int,   default=64)
    p.add_argument("--gnn-heads",   type=int,   default=4)
    p.add_argument("--gnn-layers",  type=int,   default=3)
    p.add_argument("--gnn-dropout", type=float, default=0.1)

    # FiLM
    p.add_argument("--film-hidden", type=int,   default=128)
    p.add_argument("--film-alpha",  type=float, default=0.0)

    # Checkpointing / resuming
    p.add_argument("--checkpoint-every-round", type=int, default=5,
                   help="Save checkpoint after every N rounds (0 = only at end)")
    p.add_argument("--resume-from-round", type=int, default=0,
                   help="Resume from this round's checkpoint (0 = start fresh)")

    # Misc
    p.add_argument("--output-dir", default="outputs/",
                   help="Root output dir (can be a Drive path on Colab)")
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--device", default="cuda")

    return p.parse_args()


# ── client spec parsing ───────────────────────────────────────────────────────

def _parse_client_spec(spec: str) -> ClientConfig:
    """Parse 'id:model:family:targets:d_model' into a ClientConfig."""
    parts = spec.split(":", 4)
    if len(parts) != 5:
        raise ValueError(
            f"--client must be 'id:model:family:targets:d_model', got: {spec!r}"
        )
    cid, model, family, targets_str, d_model = parts
    return ClientConfig(
        client_id=int(cid),
        model_name=model,
        model_family=family,
        lora_target_modules=targets_str.split(","),
        d_model=int(d_model),
        data_path="",  # not used here — splits loaded from disk
    )


# ── split loading ─────────────────────────────────────────────────────────────

def _load_split(splits_dir: Path, client_id: int, name: str) -> list:
    path = splits_dir / f"client_{client_id}_{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found: {path}\n"
            "Run split.py first to generate splits."
        )
    return json.loads(path.read_text(encoding="utf-8"))


# ── checkpoint helpers ────────────────────────────────────────────────────────

def _save_round_checkpoint(
    clients: list,
    round_idx: int,
    ckpt_base: Path,
) -> None:
    ckpt_dir = ckpt_base / f"round_{round_idx + 1}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for client in clients:
        cdir = ckpt_dir / f"client_{client.client_id}"
        cdir.mkdir(exist_ok=True)
        client.client_model.model.save_pretrained(str(cdir / "lora_model"))
        torch.save(client.film_adapter.state_dict(), cdir / "film.pt")

    # GNN is shared after aggregation — save once
    torch.save(clients[0].gnn.state_dict(), ckpt_dir / "gnn.pt")
    print(f"  [checkpoint] round {round_idx + 1} saved → {ckpt_dir}")


def _load_round_checkpoint(
    clients: list,
    round_idx: int,
    ckpt_base: Path,
    device: torch.device,
) -> None:
    ckpt_dir = ckpt_base / f"round_{round_idx}"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Round checkpoint not found: {ckpt_dir}")

    gnn_state = torch.load(ckpt_dir / "gnn.pt", map_location=device)
    for client in clients:
        cdir = ckpt_dir / f"client_{client.client_id}"
        client.client_model.model.load_adapter(str(cdir / "lora_model"), adapter_name="default")
        client.film_adapter.load_state_dict(
            torch.load(cdir / "film.pt", map_location=device)
        )
        client.gnn.load_state_dict(gnn_state)
    print(f"  [checkpoint] resumed from round {round_idx} ← {ckpt_dir}")


# ── build federated clients ───────────────────────────────────────────────────

def _build_clients(
    cfg: Config,
    client_splits: dict,
    device: torch.device,
    gnn_shared_state: dict | None = None,
) -> list:
    clients = []
    for client_cfg in cfg.clients:
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
        if gnn_shared_state is not None:
            gnn.load_state_dict({k: v.to(device) for k, v in gnn_shared_state.items()})

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
        print(
            f"  Client {cid} ({client_cfg.model_name}) ready — "
            f"{len(fc.train_samples)} train / {len(fc.val_samples)} val / "
            f"{len(fc.test_samples)} test"
        )
    return clients


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    set_seeds(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    ckpt_base  = output_dir / "fed_checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(str(output_dir))

    json_logger = JSONLogger(str(output_dir))

    # Parse client specs
    client_cfgs = [_parse_client_spec(s) for s in args.client]
    client_cfgs.sort(key=lambda c: c.client_id)

    # Build Config from args
    cfg = Config(
        seed=args.seed,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        max_input_len=args.max_input_len,
        max_target_len=args.max_target_len,
        lr_lora=args.lr_lora,
        lr_gnn=args.lr_gnn,
        lr_film=args.lr_film,
        warmup_ratio=args.warmup_ratio,
        grad_clip=args.grad_clip,
        eval_every_n=args.eval_every_n,
        device=args.device,
        output_dir=str(output_dir),
        lora=LoRAConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout),
        gnn=GNNConfig(hidden=args.gnn_hidden, heads=args.gnn_heads,
                      layers=args.gnn_layers, dropout=args.gnn_dropout),
        film=FiLMConfig(hidden=args.film_hidden, alpha_init=args.film_alpha),
        clients=client_cfgs,
    )

    # Load splits from disk
    splits_dir = Path(args.splits_dir)
    client_splits: dict = {}
    for client_cfg in client_cfgs:
        cid = client_cfg.client_id
        client_splits[cid] = {
            "train": _load_split(splits_dir, cid, "train"),
            "val":   _load_split(splits_dir, cid, "val"),
            "test":  _load_split(splits_dir, cid, "test"),
        }

    global_test_path = splits_dir / "global_test.json"
    global_test = json.loads(global_test_path.read_text()) if global_test_path.exists() else []

    # All clients start with the same GNN weights
    ref_gnn = ArchitectureGNN(
        in_channels=cfg.gnn.in_channels, hidden=cfg.gnn.hidden,
        heads=cfg.gnn.heads, dropout=cfg.gnn.dropout,
    )
    gnn_init_state = {k: v.cpu() for k, v in ref_gnn.state_dict().items()}
    del ref_gnn

    print("\nBuilding clients …")
    clients = _build_clients(cfg, client_splits, device, gnn_shared_state=gnn_init_state)

    # Resume from checkpoint if requested
    start_round = 0
    if args.resume_from_round > 0:
        _load_round_checkpoint(clients, args.resume_from_round, ckpt_base, device)
        start_round = args.resume_from_round

    # Federation
    server  = FederatedServer()
    trainer = LocalTrainer(cfg)
    weights = [c.n_train_samples for c in clients]

    if global_test:
        evaluator = Evaluator(clients, cfg, device)

    final_metrics: dict = {}

    print(f"\n{'=' * 60}")
    print("  FEDERATED TRAINING")
    print(f"  {len(clients)} clients × {cfg.num_rounds} rounds × {cfg.local_epochs} local epochs")
    print(f"{'=' * 60}")

    for round_idx in range(start_round, cfg.num_rounds):
        print(f"\n{'─' * 50}")
        print(f"  Round {round_idx + 1} / {cfg.num_rounds}")
        print(f"{'─' * 50}")

        round_record: dict = {"round": round_idx, "clients": [], "eval_metrics": None}

        for client in clients:
            refresh_graph_features(
                client.graph_data, client.client_model.model,
                cfg.lora.lora_alpha, cfg.lora.r,
            )
            print(f"  Training client {client.client_id} ({client.client_model.model_name}) …")
            result = trainer.train_round(client, round_idx)
            alpha_val = client.film_adapter.get_alpha()
            print(
                f"    train_loss={result['avg_train_loss']:.4f}  "
                f"val_loss={result['val_loss']:.4f}  "
                f"alpha={alpha_val:.4f}"
            )
            round_record["clients"].append({
                "client_id":   client.client_id,
                "train_loss":  result["avg_train_loss"],
                "val_loss":    result["val_loss"],
                "alpha":       alpha_val,
            })

        # GNN aggregation
        client_states = [c.get_gnn_state_dict() for c in clients]
        global_state  = server.aggregate(client_states, weights)
        for client in clients:
            client.load_gnn_state_dict(global_state)
        round_record["gnn_global_norm"] = server.global_param_norm()

        # Evaluation
        if global_test and (round_idx + 1) % cfg.eval_every_n == 0:
            print(f"\n  [Evaluation — Round {round_idx + 1}]")
            eval_results = evaluator.quantitative_eval(round_idx + 1, global_test)
            round_record["eval_metrics"] = eval_results
            if round_idx + 1 == cfg.num_rounds:
                # Average global metrics across clients
                sums = {"rouge_l": 0.0, "bleu_4": 0.0, "bertscore_f1": 0.0}
                for cid, splits in eval_results.items():
                    for k in sums:
                        sums[k] += splits.get("global", {}).get(k, 0.0)
                n = max(len(eval_results), 1)
                final_metrics = {k: v / n for k, v in sums.items()}

        json_logger.append_round(round_record)

        # Checkpoint
        if args.checkpoint_every_round > 0 and (round_idx + 1) % args.checkpoint_every_round == 0:
            _save_round_checkpoint(clients, round_idx, ckpt_base)

    # Final evaluation if not already run
    if global_test and not final_metrics:
        eval_results = evaluator.quantitative_eval(cfg.num_rounds, global_test)
        sums = {"rouge_l": 0.0, "bleu_4": 0.0, "bertscore_f1": 0.0}
        for cid, splits in eval_results.items():
            for k in sums:
                sums[k] += splits.get("global", {}).get(k, 0.0)
        n = max(len(eval_results), 1)
        final_metrics = {k: v / n for k, v in sums.items()}

    if final_metrics:
        json_logger.set_federated_final(final_metrics)
        print(
            f"\nFinal federated metrics — "
            f"ROUGE-L={final_metrics['rouge_l']:.3f}  "
            f"BLEU-4={final_metrics['bleu_4']:.3f}  "
            f"BERTScore={final_metrics['bertscore_f1']:.3f}"
        )

    # Save final models
    final_dir = output_dir / "fed_final"
    for client in clients:
        cdir = final_dir / f"client_{client.client_id}"
        cdir.mkdir(parents=True, exist_ok=True)
        client.client_model.model.save_pretrained(str(cdir / "lora_model"))
        torch.save(client.film_adapter.state_dict(), cdir / "film.pt")
    torch.save(clients[0].gnn.state_dict(), final_dir / "gnn.pt")
    print(f"\nFinal models saved → {final_dir}")


if __name__ == "__main__":
    main()
