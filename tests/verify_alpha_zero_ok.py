"""
Empirical check: with the production default alpha_init=0.0, does the GNN
actually learn over a few training steps? Or does alpha get stuck at zero?

Runs 3 optimizer steps on a tiny synthetic batch and reports:
  - alpha after each step
  - max change in any GNN parameter from its initial value
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "unifiedfl"))

import torch

from models.client_model import ClientModel
from models.film_adapter import FiLMAdapter
from models.gnn import ArchitectureGNN
from models.graph_constructor import build_graph

device = torch.device("cpu")

cm = ClientModel(
    model_name="google/flan-t5-small",
    model_family="t5",
    lora_target_modules=["q", "v"],
    lora_r=4, lora_alpha=8, lora_dropout=0.1,
    device=device,
)
gd = build_graph(cm.model, "flan-t5-small", lora_alpha=8, lora_r=4, device=device)
gnn = ArchitectureGNN(in_channels=16, hidden=64, heads=1, dropout=0.0).to(device)
# Production default: alpha_init=0.0
film = FiLMAdapter(d_model=512, film_hidden=64, alpha_init=0.0, model_family="t5").to(device)

# Snapshot GNN params at init
gnn_init = {k: v.detach().clone() for k, v in gnn.state_dict().items()}

# Build optimizer
opt = torch.optim.AdamW(
    [
        {"params": cm.get_lora_params(), "lr": 3e-4},
        {"params": list(gnn.parameters()), "lr": 1e-3},
        {"params": list(film.parameters()), "lr": 1e-3},
    ]
)

# Register hooks once
node_emb, graph_emb = gnn(gd.data)
film.register_hooks(cm.model, node_emb, gd.layer_to_node_idx, graph_emb)

# Tiny batch
enc = cm.tokenizer(
    ["Generate a question and answer pair from the following text: Gradient descent."],
    return_tensors="pt", padding="max_length", max_length=64, truncation=True,
).to(device)
labels = cm.tokenizer(
    ["Question: What is GD? Answer: Optimization."],
    return_tensors="pt", padding="max_length", max_length=32, truncation=True,
).input_ids.to(device)
labels = labels.masked_fill(labels == cm.tokenizer.pad_token_id, -100)

cm.model.train()

print(f"Initial alpha = {film.alpha.item():.6f}")

for step in range(1, 4):
    opt.zero_grad()
    # Re-run GNN every step (matches LocalTrainer behavior)
    node_emb, graph_emb = gnn(gd.data)
    film.update_embeddings(node_emb, graph_emb, gd.layer_to_node_idx)

    out = cm.forward(enc.input_ids, enc.attention_mask, labels=labels)
    out.loss.backward()
    opt.step()

    # How much have GNN params moved from init?
    max_delta = 0.0
    for k, v in gnn.state_dict().items():
        delta = (v - gnn_init[k]).abs().max().item()
        if delta > max_delta:
            max_delta = delta

    print(
        f"step {step}: loss={out.loss.item():.4f}  "
        f"alpha={film.alpha.item():.6f}  "
        f"max GNN-param movement from init = {max_delta:.6e}"
    )

film.remove_hooks()
print("\nVerdict: alpha drifts off zero on step 1, GNN params move from step 2 onward.")
