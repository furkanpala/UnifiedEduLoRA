"""
Run federated training across configurable clients.
Supports checkpointing per round and resuming, designed for Colab + Drive.

Client spec format:  "id:model_name:family:lora_targets:d_model"
  - lora_targets are comma-separated (no spaces)

Usage (Colab):
    # Mount Drive first:
    #   from google.colab import drive; drive.mount('/content/drive')

    python train_federated.py \
        --splits-dir /content/drive/MyDrive/unifiedfl/outputs/splits \
        --client "0:google/flan-t5-small:t5:q,v:512" \
        --client "1:facebook/bart-base:bart:q_proj,v_proj:768" \
        --client "2:allenai/led-base-16384:led:q_proj,v_proj:768" \
        --output-dir /content/drive/MyDrive/unifiedfl/outputs \
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
from data.dataset import render_prompt
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
    p.add_argument("--fold", type=int, default=0, choices=[0, 1, 2, 3],
                   help="If non-zero, load fold-specific splits "
                        "(client_{cid}_fold{N}_train.json) — matches the "
                        "individual-experiment split layout. 0 = legacy flat "
                        "client_{cid}_train.json layout.")

    # Federation / training
    p.add_argument("--num-rounds",     type=int,   default=20)
    p.add_argument("--local-epochs",   type=int,   default=3)
    p.add_argument("--snapshot-metric", choices=["rouge_l", "val_loss"],
                   default="rouge_l",
                   help="Metric used for best-snapshot selection. rouge_l "
                        "(default) = greedy-decoded ROUGE-L on a fixed val "
                        "subsample, averaged across clients (matches "
                        "train_client.py's early-stop metric). val_loss = "
                        "teacher-forced cross-entropy averaged across clients.")
    p.add_argument("--fast-eval-samples", type=int, default=50,
                   help="Per-client val samples used for the fast ROUGE-L "
                        "snapshot metric (drawn once with seed=args.seed).")
    p.add_argument("--batch-size",     type=int,   default=4)
    p.add_argument("--lr-lora",        type=float, default=3e-4)
    p.add_argument("--lr-gnn",         type=float, default=1e-3)
    p.add_argument("--lr-film",        type=float, default=1e-3)
    p.add_argument("--warmup-ratio",   type=float, default=0.1)
    p.add_argument("--grad-clip",      type=float, default=1.0)
    p.add_argument("--max-input-len",  type=int,   default=512)
    p.add_argument("--max-target-len", type=int,   default=128)
    p.add_argument("--eval-every-n",   type=int,   default=5)

    # Conditioning
    p.add_argument("--conditioning", default="baseline",
                   choices=["baseline", "topic", "bloom"],
                   help="Prompt-conditioning mode used for training AND evaluation.")
    p.add_argument("--no-qual-eval", action="store_true",
                   help="Skip qualitative evaluation (useful for topic/bloom conditioning "
                        "when hardcoded contexts are less meaningful).")

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

def _load_split(splits_dir: Path, client_id: int, name: str,
                fold: int = 0) -> list:
    """
    fold == 0: legacy flat layout produced by split.py:
                 client_{cid}_{train,val,test}.json
    fold >= 1: per-fold layout produced by experiments/02_split_3fold.py:
                 client_{cid}_fold{N}_{train,val}.json     (train + val only)
                 client_{cid}_test.json                    (single test set)
    """
    if fold > 0 and name in ("train", "val"):
        path = splits_dir / f"client_{client_id}_fold{fold}_{name}.json"
    else:
        path = splits_dir / f"client_{client_id}_{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found: {path}\n"
            "Run split.py / 02_split_3fold.py first to generate splits."
        )
    return json.loads(path.read_text(encoding="utf-8"))


# ── fast ROUGE-L for federated best-snapshot selection ──────────────────────

@torch.no_grad()
def _fast_rouge_l_clients(
    clients: list,
    es_subsamples: dict,
    cfg: Config,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, dict]:
    """
    Greedy-decoded ROUGE-L on a fixed val subsample per client. Mirrors
    train_client._fast_rouge_l so snapshot selection behaves identically to
    individual-training early-stop. Returns (avg_across_clients, per_client_dict).
    """
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    per_client: dict = {}
    for client in clients:
        samples = es_subsamples.get(client.client_id, [])
        if not samples:
            per_client[client.client_id] = 0.0
            continue
        tokenizer = client.client_model.tokenizer
        client.client_model.model.eval()
        total = 0.0
        for sample in samples:
            prompt = render_prompt(sample, cfg.conditioning)
            enc = tokenizer(
                prompt, max_length=cfg.max_input_len,
                truncation=True, padding=False, return_tensors="pt",
            ).to(device)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
                out_ids = client.client_model.generate(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    num_beams=1,
                    max_new_tokens=cfg.max_target_len,
                    no_repeat_ngram_size=3,
                )
            gen = tokenizer.decode(out_ids[0], skip_special_tokens=True)
            ref = f"Question: {sample['question']}\nAnswer: {sample['answer']}"
            total += scorer.score(ref, gen)["rougeL"].fmeasure
        client.client_model.model.train()
        per_client[client.client_id] = total / len(samples)

    avg = sum(per_client.values()) / max(len(per_client), 1)
    return avg, per_client


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
    _load_state_from_dir(clients, ckpt_dir, device)
    print(f"  [checkpoint] resumed from round {round_idx} ← {ckpt_dir}")


def _save_snapshot(clients: list, snapshot_dir: Path) -> None:
    """Save a named snapshot of all client states (LoRA + FiLM + shared GNN)."""
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for client in clients:
        cdir = snapshot_dir / f"client_{client.client_id}"
        cdir.mkdir(exist_ok=True)
        client.client_model.model.save_pretrained(str(cdir / "lora_model"))
        torch.save(client.film_adapter.state_dict(), cdir / "film.pt")
    torch.save(clients[0].gnn.state_dict(), snapshot_dir / "gnn.pt")


def _load_state_from_dir(
    clients: list,
    snapshot_dir: Path,
    device: torch.device,
) -> None:
    """
    Restore all client states from a snapshot directory.

    Uses set_peft_model_state_dict() to overwrite the existing 'default' LoRA
    adapter. PEFT's load_adapter() would create a NEW adapter rather than
    overwrite — that is the latent bug in older code paths.
    """
    from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict

    gnn_state = torch.load(snapshot_dir / "gnn.pt", map_location=device)
    for client in clients:
        cdir = snapshot_dir / f"client_{client.client_id}"
        lora_state = load_peft_weights(str(cdir / "lora_model"), device=str(device))
        set_peft_model_state_dict(client.client_model.model, lora_state)
        client.film_adapter.load_state_dict(
            torch.load(cdir / "film.pt", map_location=device)
        )
        client.gnn.load_state_dict(gnn_state)


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
        conditioning=args.conditioning,
    )

    # Load splits from disk
    splits_dir = Path(args.splits_dir)
    client_splits: dict = {}
    for client_cfg in client_cfgs:
        cid = client_cfg.client_id
        client_splits[cid] = {
            "train": _load_split(splits_dir, cid, "train", fold=args.fold),
            "val":   _load_split(splits_dir, cid, "val",   fold=args.fold),
            "test":  _load_split(splits_dir, cid, "test"),  # single test/client
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

    # Best-snapshot tracking. Two metrics supported:
    #   - rouge_l (default):  greedy-decoded ROUGE-L on a fixed val subsample,
    #                          averaged across clients. Higher is better.
    #                          Matches train_client.py's early-stop metric.
    #   - val_loss:            teacher-forced cross-entropy averaged across
    #                          clients. Lower is better.
    higher_is_better = (args.snapshot_metric == "rouge_l")
    best_metric = float("-inf") if higher_is_better else float("inf")
    best_round = 0
    best_snapshot_dir = ckpt_base / "best_snapshot"
    min_delta = 1e-4

    # Fixed per-client val subsamples used by the fast ROUGE-L metric.
    # Drawn once with seed=args.seed so the metric is comparable across rounds
    # and across resumed runs.
    es_subsamples: dict = {}
    if args.snapshot_metric == "rouge_l":
        es_rng = random.Random(args.seed)
        for client in clients:
            n = min(args.fast_eval_samples, len(client.val_samples))
            es_subsamples[client.client_id] = es_rng.sample(client.val_samples, n)
        sizes = ", ".join(f"c{cid}={len(s)}"
                          for cid, s in es_subsamples.items())
        print(f"  Best-snapshot metric: greedy ROUGE-L on fixed val subsamples ({sizes}).")
    else:
        print("  Best-snapshot metric: avg val_loss (teacher-forced CE).")

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

        # ── Best-snapshot tracking ──────────────────────────────────────────
        # Always log avg val_loss for monitoring even when we select on ROUGE-L.
        val_losses = [c["val_loss"] for c in round_record["clients"]]
        avg_val_loss = sum(val_losses) / max(len(val_losses), 1)
        round_record["avg_val_loss"] = avg_val_loss

        if args.snapshot_metric == "rouge_l":
            # Compute fast greedy ROUGE-L AFTER aggregation so the metric
            # reflects what the saved snapshot will produce at inference time.
            # use_amp matches the trainer's convention (BF16 on capable GPUs).
            use_amp_eval = device.type == "cuda" and torch.cuda.is_bf16_supported()
            avg_rouge, per_client_rouge = _fast_rouge_l_clients(
                clients, es_subsamples, cfg, device, use_amp_eval,
            )
            round_record["avg_val_rouge_l"] = avg_rouge
            round_record["per_client_val_rouge_l"] = per_client_rouge
            cur_metric, label = avg_rouge, "rouge_l"
        else:
            cur_metric, label = avg_val_loss, "val_loss"

        improved = (cur_metric > best_metric + min_delta) if higher_is_better \
                   else (cur_metric < best_metric - min_delta)
        if improved:
            best_metric = cur_metric
            best_round = round_idx + 1
            _save_snapshot(clients, best_snapshot_dir)
            print(f"  → new best avg {label}: {cur_metric:.4f} "
                  f"(round {best_round}) — snapshot saved   "
                  f"[avg val_loss={avg_val_loss:.4f}]")
        else:
            print(f"  avg {label}: {cur_metric:.4f} "
                  f"(best={best_metric:.4f} @ round {best_round})   "
                  f"[avg val_loss={avg_val_loss:.4f}]")

        # Periodic per-round monitoring eval (NOT used for final metrics)
        if global_test and (round_idx + 1) % cfg.eval_every_n == 0:
            print(f"\n  [Monitoring eval — Round {round_idx + 1}]")
            eval_results = evaluator.quantitative_eval(round_idx + 1, global_test)
            round_record["eval_metrics"] = eval_results

        json_logger.append_round(round_record)

        # Checkpoint
        if args.checkpoint_every_round > 0 and (round_idx + 1) % args.checkpoint_every_round == 0:
            _save_round_checkpoint(clients, round_idx, ckpt_base)

    # ── Save the final-round state BEFORE restoring the best snapshot ────────
    # This preserves the round-N model so it can be loaded later for comparison.
    final_round_dir = output_dir / "fed_final_round"
    print(f"\nSaving final-round (round {cfg.num_rounds}) state → {final_round_dir}")
    _save_snapshot(clients, final_round_dir)

    # ── Restore best snapshot before final evaluation ────────────────────────
    if best_round > 0 and best_snapshot_dir.exists():
        print(f"\n{'=' * 60}")
        print(f"  Restoring best snapshot: round {best_round} "
              f"(avg {args.snapshot_metric} = {best_metric:.4f})")
        print(f"{'=' * 60}")
        _load_state_from_dir(clients, best_snapshot_dir, device)
    else:
        print("\nNo best snapshot to restore — using final-round state.")

    # ── Final evaluation on the (possibly restored) best snapshot ────────────
    if global_test:
        print(f"\n{'=' * 60}")
        print("  Final evaluation on best snapshot")
        print(f"{'=' * 60}")
        eval_results = evaluator.quantitative_eval(cfg.num_rounds, global_test)
        sums = {"rouge_l": 0.0, "bleu_4": 0.0, "bertscore_f1": 0.0}
        for cid, splits in eval_results.items():
            for k in sums:
                sums[k] += splits.get("global", {}).get(k, 0.0)
        n = max(len(eval_results), 1)
        final_metrics = {k: v / n for k, v in sums.items()}

        per_client_path = output_dir / "final_metrics_per_client.json"
        per_client_path.write_text(
            json.dumps({
                "best_round":              best_round,
                "best_snapshot_metric":    args.snapshot_metric,
                "best_snapshot_value":     best_metric,
                "per_client_metrics":      {str(cid): v for cid, v in eval_results.items()},
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  Per-client metrics saved → {per_client_path}")
        print(f"  Best-snapshot round: {best_round}")

    if final_metrics:
        json_logger.set_federated_final(final_metrics)
        print(
            f"\nFinal federated metrics — "
            f"ROUGE-L={final_metrics['rouge_l']:.3f}  "
            f"BLEU-4={final_metrics['bleu_4']:.3f}  "
            f"BERTScore={final_metrics['bertscore_f1']:.3f}"
        )

    # Save the (now-restored best) state to fed_final/ — this is the model
    # whose metrics are reported in final_metrics_per_client.json.
    final_dir = output_dir / "fed_final"
    for client in clients:
        cdir = final_dir / f"client_{client.client_id}"
        cdir.mkdir(parents=True, exist_ok=True)
        client.client_model.model.save_pretrained(str(cdir / "lora_model"))
        torch.save(client.film_adapter.state_dict(), cdir / "film.pt")
    torch.save(clients[0].gnn.state_dict(), final_dir / "gnn.pt")

    print(f"\n{'=' * 60}")
    print("  Saved model artifacts:")
    print(f"  • {final_dir}/  — best snapshot (round {best_round}); "
          f"used for reported metrics")
    print(f"  • {final_round_dir}/  — actual final-round state (round {cfg.num_rounds})")
    print(f"  • {best_snapshot_dir}/  — same as fed_final/, kept inside fed_checkpoints/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
