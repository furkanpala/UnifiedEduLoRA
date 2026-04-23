# ── COLAB SETUP (run this cell first) ──────────────────────────────────────
# !pip install transformers==4.40.0
# !pip install peft==0.10.0
# !pip install torch-geometric
# !pip install pyg-lib torch-scatter torch-sparse -f \
#     https://data.pyg.org/whl/torch-2.3.0+cu121.html
# !pip install rouge_score bert_score nltk accelerate
# !pip install sentencepiece
# import nltk; nltk.download('punkt')
# ───────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

# ── Project imports ────────────────────────────────────────────────────────
from config.config import Config
from data.dataset import QADataset
from data.preprocessing import prepare_all_data
from evaluation.evaluator import Evaluator
from evaluation.metrics import compute_all_metrics
from federation.client import FederatedClient
from federation.server import FederatedServer
from models.client_model import ClientModel
from models.film_adapter import FiLMAdapter
from models.gnn import ArchitectureGNN
from models.graph_constructor import build_graph, refresh_graph_features
from training.trainer import LocalTrainer
from utils.logging_utils import (
    JSONLogger,
    print_balancing_table,
    print_comparison_table,
    setup_logging,
)


# ─────────────────────────────────────────────────────────────────────────────
# Seed & device helpers
# ─────────────────────────────────────────────────────────────────────────────

def set_seeds(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_device(preferred: str) -> torch.device:
    """Return the requested device, falling back to CPU with a warning."""
    if preferred == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available — falling back to CPU.")
        return torch.device("cpu")
    return torch.device(preferred)


# ─────────────────────────────────────────────────────────────────────────────
# Client factory
# ─────────────────────────────────────────────────────────────────────────────

def build_clients(
    cfg: Config,
    client_splits: Dict[int, Dict[str, List[Dict[str, str]]]],
    device: torch.device,
    gnn_shared_state: Dict[str, Any] | None = None,
) -> List[FederatedClient]:
    """
    Instantiate ClientModel, GNN, FiLM adapter, and graph for each client.

    If `gnn_shared_state` is provided all GNNs are initialised from it
    (used when re-initialising after the baseline run).
    """
    clients: List[FederatedClient] = []

    for client_cfg in cfg.clients:
        cid = client_cfg.client_id

        # Language model + LoRA
        client_model = ClientModel(
            model_name=client_cfg.model_name,
            model_family=client_cfg.model_family,
            lora_target_modules=client_cfg.lora_target_modules,
            lora_r=cfg.lora.r,
            lora_alpha=cfg.lora.lora_alpha,
            lora_dropout=cfg.lora.lora_dropout,
            device=device,
        )

        # Architecture graph (built from the PEFT-wrapped model)
        graph_data = build_graph(
            peft_model=client_model.model,
            model_name=client_cfg.model_name,
            lora_alpha=cfg.lora.lora_alpha,
            lora_r=cfg.lora.r,
            device=device,
        )

        # GNN — always FP32, one per client (weights will be aggregated)
        gnn = ArchitectureGNN(
            in_channels=cfg.gnn.in_channels,
            hidden=cfg.gnn.hidden,
            heads=cfg.gnn.heads,
            dropout=cfg.gnn.dropout,
        ).to(device)

        if gnn_shared_state is not None:
            gnn.load_state_dict({k: v.to(device) for k, v in gnn_shared_state.items()})

        # FiLM adapter — client-local, FP32
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
        print(f"  Client {cid} ({client_cfg.model_name}) ready — "
              f"{len(fc.train_samples)} train / {len(fc.val_samples)} val / "
              f"{len(fc.test_samples)} test qa-pairs")

    return clients


# ─────────────────────────────────────────────────────────────────────────────
# Baseline: individual training without federation
# ─────────────────────────────────────────────────────────────────────────────

def run_individual_baseline(
    cfg: Config,
    client_splits: Dict[int, Dict[str, List[Dict[str, str]]]],
    global_test_samples: List[Dict[str, str]],
    device: torch.device,
    json_logger: JSONLogger,
) -> Dict[str, float]:
    """
    Train each client independently with the same LoRA setup but no GNN or FiLM.

    Total training epochs = NUM_ROUNDS * LOCAL_EPOCHS to match the federated run.
    Returns aggregated metrics on the global test set.
    """
    print("\n" + "=" * 60)
    print("  BASELINE: Individual training (no federation, no GNN)")
    print("=" * 60)

    total_epochs = cfg.num_rounds * cfg.local_epochs
    use_amp = device.type == "cuda"
    all_predictions: List[str] = []
    all_references: List[str] = []

    for client_cfg in cfg.clients:
        cid = client_cfg.client_id
        print(f"\nBaseline — Client {cid} ({client_cfg.model_name})")

        client_model = ClientModel(
            model_name=client_cfg.model_name,
            model_family=client_cfg.model_family,
            lora_target_modules=client_cfg.lora_target_modules,
            lora_r=cfg.lora.r,
            lora_alpha=cfg.lora.lora_alpha,
            lora_dropout=cfg.lora.lora_dropout,
            device=device,
        )

        train_loader = DataLoader(
            QADataset(
                client_splits[cid]["train"],
                client_model.tokenizer,
                cfg.max_input_len,
                cfg.max_target_len,
            ),
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=use_amp,
        )

        optimizer = torch.optim.AdamW(
            client_model.get_lora_params(), lr=cfg.lr_lora, weight_decay=0.01
        )
        total_steps = len(train_loader) * total_epochs
        warmup_steps = max(1, int(cfg.warmup_ratio * total_steps))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        client_model.model.train()
        for epoch in range(total_epochs):
            losses: List[float] = []
            for batch in train_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = client_model.forward(input_ids, attention_mask, labels)
                    loss = outputs.loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    client_model.get_lora_params(), cfg.grad_clip
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                losses.append(loss.item())
            avg = sum(losses) / max(len(losses), 1)
            print(f"  Epoch {epoch + 1}/{total_epochs} — avg_loss={avg:.4f}")

        # Evaluate on global test set
        preds, refs = _generate_on_samples(
            client_model, global_test_samples, cfg, device, use_amp
        )
        all_predictions.extend(preds)
        all_references.extend(refs)

        torch.cuda.empty_cache()
        del client_model

    metrics = compute_all_metrics(all_predictions, all_references, device)
    json_logger.set_baseline_metrics(metrics)
    print(
        f"\nBaseline metrics (global) — "
        f"ROUGE-L={metrics['rouge_l']:.3f}  "
        f"BLEU-4={metrics['bleu_4']:.3f}  "
        f"BERTScore={metrics['bertscore_f1']:.3f}"
    )
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Federated training
# ─────────────────────────────────────────────────────────────────────────────

def run_federated_training(
    cfg: Config,
    clients: List[FederatedClient],
    global_test_samples: List[Dict[str, str]],
    device: torch.device,
    json_logger: JSONLogger,
) -> Dict[str, float]:
    """
    Main federated training loop: 20 rounds × 3 local epochs.

    Federation mechanism:
      - GNN weights (φ) are averaged via weighted FedAvg each round.
      - LoRA and FiLM weights remain client-local.
    """
    print("\n" + "=" * 60)
    print("  FEDERATED TRAINING (GNN-based cross-architecture federation)")
    print("=" * 60)

    server = FederatedServer()
    trainer = LocalTrainer(cfg)
    evaluator = Evaluator(clients, cfg, device)
    weights = [c.n_train_samples for c in clients]

    final_metrics: Dict[str, float] = {}

    for round_idx in range(cfg.num_rounds):
        print(f"\n{'─' * 50}")
        print(f"  Federation Round {round_idx + 1} / {cfg.num_rounds}")
        print(f"{'─' * 50}")

        round_record: Dict[str, Any] = {
            "round": round_idx,
            "clients": [],
            "gnn_global_norm": 0.0,
            "eval_metrics": None,
        }

        # Steps 1–5: refresh graph, train (GNN run per-batch inside trainer)
        for client in clients:
            # Step 1: refresh effective-weight node features
            refresh_graph_features(
                client.graph_data,
                client.client_model.model,
                cfg.lora.lora_alpha,
                cfg.lora.r,
            )

            print(f"  Training client {client.client_id} "
                  f"({client.client_model.model_name}) …")
            result = trainer.train_round(client, round_idx)

            alpha_val = client.film_adapter.get_alpha()
            with torch.no_grad():
                _, graph_emb = client.gnn(client.graph_data.data)
            graph_emb_list = graph_emb[0].tolist()

            round_record["clients"].append({
                "client_id": client.client_id,
                "train_loss": result["avg_train_loss"],
                "val_loss": result["val_loss"],
                "kd_loss": 0,
                "graph_embedding": graph_emb_list,
                "alpha": alpha_val,
            })
            print(
                f"    client={client.client_id}  "
                f"train_loss={result['avg_train_loss']:.4f}  "
                f"val_loss={result['val_loss']:.4f}  "
                f"alpha={alpha_val:.4f}"
            )

        # Steps 6–8: GNN aggregation and broadcast
        client_states = [c.get_gnn_state_dict() for c in clients]
        global_state = server.aggregate(client_states, weights)
        for client in clients:
            client.load_gnn_state_dict(global_state)

        round_record["gnn_global_norm"] = server.global_param_norm()

        # Steps 9–10: evaluation every EVAL_EVERY_N rounds
        if (round_idx + 1) % cfg.eval_every_n == 0:
            print(f"\n  [Qualitative Evaluation — Round {round_idx + 1}]")
            evaluator.qualitative_eval(round_idx + 1)

            print(f"\n  [Quantitative Evaluation — Round {round_idx + 1}]")
            eval_results = evaluator.quantitative_eval(
                round_idx + 1, global_test_samples
            )
            round_record["eval_metrics"] = eval_results

            # Track final metrics from the last eval round
            if round_idx + 1 == cfg.num_rounds:
                final_metrics = _average_global_metrics(eval_results)

        json_logger.append_round(round_record)

    # Final evaluation if not already done
    if not final_metrics:
        eval_results = evaluator.quantitative_eval(cfg.num_rounds, global_test_samples)
        final_metrics = _average_global_metrics(eval_results)

    json_logger.set_federated_final(final_metrics)

    # Save all clients
    output_dir = cfg.output_dir
    for client in clients:
        client.save(output_dir)
    torch.save(
        clients[0].gnn.state_dict(),  # all gnns share same weights post-aggregation
        Path(output_dir) / "gnn_final.pt",
    )

    return final_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _generate_on_samples(
    client_model: ClientModel,
    samples: List[Dict[str, str]],
    cfg: Config,
    device: torch.device,
    use_amp: bool,
) -> tuple[List[str], List[str]]:
    """Generate predictions and collect references for a list of samples."""
    client_model.model.eval()
    dataset = QADataset(
        samples, client_model.tokenizer, cfg.max_input_len, cfg.max_target_len
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=use_amp,
    )
    predictions: List[str] = []
    references: List[str] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        generated = client_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=4,
            max_new_tokens=cfg.max_target_len,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
        for ids in generated:
            predictions.append(
                client_model.tokenizer.decode(ids, skip_special_tokens=True)
            )
        for lab in batch["labels"]:
            lab = lab.masked_fill(lab == -100, client_model.tokenizer.pad_token_id)
            references.append(
                client_model.tokenizer.decode(lab, skip_special_tokens=True)
            )

    return predictions, references


def _average_global_metrics(
    eval_results: Dict[int, Dict[str, Dict[str, float]]]
) -> Dict[str, float]:
    """Average the 'global' split metrics across all clients."""
    sums: Dict[str, float] = {"rouge_l": 0.0, "bleu_4": 0.0, "bertscore_f1": 0.0}
    n = len(eval_results)
    for cid, splits in eval_results.items():
        for key in sums:
            sums[key] += splits["global"].get(key, 0.0)
    return {k: v / max(n, 1) for k, v in sums.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 60)
    print("  COLAB SETUP REMINDER")
    print("  Run the install block at the top of main.py before starting.")
    print("=" * 60 + "\n")

    cfg = Config()
    set_seeds(cfg.seed)
    device = get_device(cfg.device)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    logger = setup_logging(cfg.output_dir)
    logger.info(f"Using device: {device}")

    json_logger = JSONLogger(cfg.output_dir)

    # ── Step 3: Load, balance, and split all datasets ─────────────────────
    print("\nLoading and balancing datasets …")
    data_info = prepare_all_data(cfg.clients, cfg.output_dir, cfg.seed)

    # ── Step 4: Print balancing summary ──────────────────────────────────
    client_names = ["C0 MIT/T5", "C1 Stan/BART", "C2 Papers/LED"]
    print_balancing_table(
        raw_counts=data_info["raw_counts"],
        balanced_counts=data_info["balanced_counts"],
        train_qa_counts=data_info["train_qa_counts"],
        client_names=client_names,
    )

    client_splits = data_info["client_splits"]

    # Global test set = union of all three clients' test splits
    global_test_samples: List[Dict[str, str]] = []
    for cid in sorted(client_splits.keys()):
        global_test_samples.extend(client_splits[cid]["test"])

    # ── Step 8: Baseline run ──────────────────────────────────────────────
    print("\nInitialising clients for baseline …")
    baseline_metrics = run_individual_baseline(
        cfg, client_splits, global_test_samples, device, json_logger
    )

    # ── Step 9: Re-initialise all models with same seed ───────────────────
    set_seeds(cfg.seed)
    print("\nRe-initialising clients for federated training …")

    # Build a single GNN and copy its state for each client so all start identically
    reference_gnn = ArchitectureGNN(
        in_channels=cfg.gnn.in_channels,
        hidden=cfg.gnn.hidden,
        heads=cfg.gnn.heads,
        dropout=cfg.gnn.dropout,
    )
    gnn_init_state = {k: v.cpu() for k, v in reference_gnn.state_dict().items()}
    del reference_gnn

    clients = build_clients(cfg, client_splits, device, gnn_shared_state=gnn_init_state)

    # ── Step 10: Federated training ───────────────────────────────────────
    federated_metrics = run_federated_training(
        cfg, clients, global_test_samples, device, json_logger
    )

    # ── Step 13: Final comparison table ──────────────────────────────────
    print_comparison_table(baseline_metrics, federated_metrics)


if __name__ == "__main__":
    main()
