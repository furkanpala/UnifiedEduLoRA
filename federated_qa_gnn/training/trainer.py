from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from data.dataset import QADataset
from models.graph_constructor import refresh_graph_features

logger = logging.getLogger("federated_qa")


class LocalTrainer:
    """
    Executes one federation round of local training for a single client.

    Training order per batch:
      1. Run GNN in FP32 (no autocast) → fresh node_embeddings with grad
      2. Update FiLM hook references with new embeddings
      3. Run LM forward in FP16 (autocast) — FiLM hooks apply modulation
      4. Backward → clip → step

    This ordering is required for gradient flow:
      ∂L/∂φ_k is non-zero because the loss flows back through FiLM → GNN.
    """

    def __init__(self, config: Any) -> None:
        self.config = config

    def train_round(
        self,
        client: Any,       # FederatedClient
        round_idx: int,
    ) -> Dict[str, float]:
        """
        Run LOCAL_EPOCHS of training for `client` in one federation round.

        Returns:
            dict with 'avg_train_loss' and 'final_val_loss'
        """
        cfg = self.config
        device = client.device
        # Use BF16 autocast on GPUs that support it (A100/H100).
        # FP16 causes T5 attention logits to overflow; BF16 has the same
        # exponent range as FP32 and does not. On T4 (no BF16), use_amp=False
        # and training runs in stable FP32. GradScaler is only for FP16, so
        # it is always disabled here.
        use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
        pin_mem = device.type == "cuda"

        train_loader = DataLoader(
            QADataset(
                client.train_samples,
                client.client_model.tokenizer,
                cfg.max_input_len,
                cfg.max_target_len,
            ),
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=pin_mem,
        )
        val_loader = DataLoader(
            QADataset(
                client.val_samples,
                client.client_model.tokenizer,
                cfg.max_input_len,
                cfg.max_target_len,
            ),
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=pin_mem,
        )

        # Build parameter groups
        lora_params = client.client_model.get_lora_params()
        gnn_params = list(client.gnn.parameters())
        film_params = list(client.film_adapter.parameters())

        optimizer = torch.optim.AdamW(
            [
                {"params": lora_params, "lr": cfg.lr_lora},
                {"params": gnn_params, "lr": cfg.lr_gnn},
                {"params": film_params, "lr": cfg.lr_film},
            ],
            weight_decay=0.01,
        )

        total_steps = len(train_loader) * cfg.local_epochs
        warmup_steps = max(1, int(cfg.warmup_ratio * total_steps))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        scaler = torch.amp.GradScaler("cuda", enabled=False)

        # Step 1 — refresh effective weight stats before training starts
        refresh_graph_features(
            client.graph_data,
            client.client_model.model,
            cfg.lora.lora_alpha,
            cfg.lora.r,
        )

        # Step 3 — register FiLM hooks (hooks reference mutable embeddings updated per batch)
        with torch.no_grad():
            init_node_emb, init_graph_emb = client.gnn(client.graph_data.data)
        client.film_adapter.register_hooks(
            client.client_model.model,
            init_node_emb,
            client.graph_data.layer_to_node_idx,
            init_graph_emb,
        )

        all_train_losses: List[float] = []
        val_loss = 0.0
        global_step = 0

        try:
            for epoch in range(cfg.local_epochs):
                client.client_model.model.train()
                epoch_losses: List[float] = []

                for batch in train_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    optimizer.zero_grad()

                    # Step 2 — run GNN in FP32, get fresh node_embeddings with grad
                    node_emb, graph_emb = client.gnn(client.graph_data.data)
                    client.film_adapter.update_embeddings(
                        node_emb, graph_emb, client.graph_data.layer_to_node_idx
                    )

                    # Step 4 — LM forward under autocast (FiLM hooks fire here)
                    with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
                        outputs = client.client_model.forward(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )
                        loss = outputs.loss

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    all_params = lora_params + gnn_params + film_params
                    torch.nn.utils.clip_grad_norm_(all_params, cfg.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()

                    batch_loss = loss.item()
                    epoch_losses.append(batch_loss)
                    all_train_losses.append(batch_loss)
                    global_step += 1

                    logger.debug(
                        f"round={round_idx} epoch={epoch} step={global_step} "
                        f"client={client.client_id} loss={batch_loss:.4f} "
                        f"lr={scheduler.get_last_lr()[0]:.2e}"
                    )

                avg_epoch_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
                val_loss = self._compute_val_loss(client, val_loader, use_amp, device)

                print(
                    f"    epoch {epoch + 1}/{cfg.local_epochs} — "
                    f"train={avg_epoch_loss:.4f}  val={val_loss:.4f}"
                )

        finally:
            # Step 5 — always remove hooks after training
            client.film_adapter.remove_hooks()

        # Free GPU memory before next client
        torch.cuda.empty_cache()

        avg_train_loss = sum(all_train_losses) / max(len(all_train_losses), 1)
        return {"avg_train_loss": avg_train_loss, "val_loss": val_loss}

    @torch.no_grad()
    def _compute_val_loss(
        self,
        client: Any,
        val_loader: DataLoader,
        use_amp: bool,
        device: torch.device,
    ) -> float:
        """Compute mean cross-entropy loss on the validation set."""
        client.client_model.model.eval()
        losses: List[float] = []
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
                outputs = client.client_model.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
            losses.append(outputs.loss.item())
        client.client_model.model.train()
        return sum(losses) / max(len(losses), 1)
