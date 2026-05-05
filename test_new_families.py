"""
End-to-end smoke test for the Pegasus-X and ProphetNet family integrations.

For each model:
  1. Load via ClientModel (base model + LoRA adapter)
  2. Build the architecture graph via build_graph
  3. Run the GNN over the graph
  4. Register FiLM hooks on every transformer block
  5. Run a forward+backward step on a tiny dummy batch
  6. Run a 1-step generate() call
  7. Clean up

CPU only. Slow, but proves the wiring is correct end-to-end. If any step
fails, the family integration has a bug.
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import torch

REPO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_DIR / "unifiedfl"))

from models.client_model import ClientModel
from models.film_adapter import FiLMAdapter
from models.gnn import ArchitectureGNN
from models.graph_constructor import build_graph, refresh_graph_features


# ── Models to test ────────────────────────────────────────────────────────────

CASES = [
    {
        "name":    "Pegasus-X-base",
        "model":   "google/pegasus-x-base",
        "family":  "pegasus_x",
        "targets": ["q_proj", "v_proj"],
        "d_model": 768,
    },
    {
        "name":    "ProphetNet-large",
        "model":   "microsoft/prophetnet-large-uncased",
        "family":  "prophetnet",
        "targets": ["query_proj", "value_proj"],
        "d_model": 1024,
    },
]

DEVICE = torch.device("cpu")


def _ok(msg):  print(f"  [OK]   {msg}")
def _fail(msg): print(f"  [FAIL] {msg}")


def run_one_case(case: dict) -> bool:
    print(f"\n{'='*72}")
    print(f"  {case['name']}  ({case['model']})")
    print(f"  family={case['family']}  targets={case['targets']}  d_model={case['d_model']}")
    print(f"{'='*72}")

    try:
        # 1. Load ClientModel
        t0 = time.time()
        client_model = ClientModel(
            model_name=case["model"],
            model_family=case["family"],
            lora_target_modules=case["targets"],
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            device=DEVICE,
        )
        n_trainable = sum(p.numel() for p in client_model.model.parameters() if p.requires_grad)
        n_total     = sum(p.numel() for p in client_model.model.parameters())
        _ok(f"ClientModel loaded in {time.time()-t0:.1f}s; "
            f"trainable={n_trainable:,} / total={n_total:,} "
            f"({100*n_trainable/n_total:.3f}% trainable)")

        # 2. Build graph
        t0 = time.time()
        graph_data = build_graph(
            peft_model=client_model.model,
            model_name=case["model"],
            lora_alpha=16,
            lora_r=8,
            device=DEVICE,
        )
        n_nodes = graph_data.num_nodes
        n_edges = graph_data.data.edge_index.shape[1]
        _ok(f"Graph built in {time.time()-t0:.1f}s; "
            f"{n_nodes} nodes / {n_edges} edges; "
            f"node feature shape = {tuple(graph_data.data.x.shape)}")

        # 3. Run the GNN
        gnn = ArchitectureGNN(in_channels=16, hidden=64, heads=4, dropout=0.1).to(DEVICE)
        node_emb, graph_emb = gnn(graph_data.data)
        _ok(f"GNN ran; node_emb shape = {tuple(node_emb.shape)}, "
            f"graph_emb shape = {tuple(graph_emb.shape)}")

        # 4. Register FiLM hooks
        film = FiLMAdapter(
            d_model=case["d_model"],
            film_hidden=64,
            alpha_init=0.0,
            model_family=case["family"],
        ).to(DEVICE)
        film.register_hooks(
            client_model.model,
            node_emb,
            graph_data.layer_to_node_idx,
            graph_emb,
        )
        n_hooks = len(film._hook_handles)
        if n_hooks == 0:
            _fail(f"NO FiLM hooks registered — family '{case['family']}' "
                  f"is not picking up any transformer-block classes")
            return False
        _ok(f"FiLM hooks registered on {n_hooks} transformer blocks")

        # 5. Forward + backward
        tokenizer = client_model.tokenizer
        batch = tokenizer(
            ["Generate a question and answer pair from the following text: "
             "Gradient descent minimises a loss function."],
            return_tensors="pt", padding=True, truncation=True, max_length=64,
        ).to(DEVICE)
        labels = tokenizer(
            ["Question: What does gradient descent do?\nAnswer: It minimises loss."],
            return_tensors="pt", padding=True, truncation=True, max_length=32,
        ).to(DEVICE).input_ids

        client_model.model.train()
        out = client_model.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=labels,
        )
        loss = out.loss
        if not torch.isfinite(loss):
            _fail(f"Forward produced non-finite loss: {loss.item()}")
            return False
        _ok(f"Forward OK; loss = {loss.item():.4f}")

        loss.backward()
        # Verify some LoRA gradient actually reached the parameters
        n_with_grad = sum(
            1 for p in client_model.model.parameters()
            if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0
        )
        _ok(f"Backward OK; {n_with_grad} LoRA parameters received non-zero gradients")

        # 6. Generate
        client_model.model.eval()
        gen_ids = client_model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            num_beams=1,
            max_new_tokens=16,
        )
        decoded = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        _ok(f"Generation OK; output = {decoded!r}")

        # 7. Cleanup
        film.remove_hooks()
        del client_model, gnn, film, graph_data
        import gc
        gc.collect()
        _ok("Hooks removed, models released")

        return True

    except Exception as e:
        _fail(f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return False


def main():
    print(f"Running smoke tests on device={DEVICE}")
    print(f"transformers={__import__('transformers').__version__}, "
          f"peft={__import__('peft').__version__}, "
          f"torch={torch.__version__}")

    results = {}
    for case in CASES:
        results[case["name"]] = run_one_case(case)

    print(f"\n{'='*72}")
    print("  RESULTS")
    print(f"{'='*72}")
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}]  {name}")

    if all(results.values()):
        print("\nAll new families work end-to-end.")
        return 0
    else:
        print("\nAt least one family failed. See traceback above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
