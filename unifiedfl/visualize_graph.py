"""
Visualize the layer-level architecture graphs produced by build_graph().

Three panels per model:
  1. Architecture graph  — nodes colored by type, sized by param count,
                           encoder (left column) vs decoder (right column)
  2. Node-feature heatmap — 16-dim feature matrix across all nodes
  3. Layer-type distribution — stacked bar of node categories

Usage (Colab):
    python visualize_graph.py \
        --model google/flan-t5-small --family t5 --targets q v --d-model 512

    # All three default clients side by side:
    python visualize_graph.py --all-clients

    # Save without displaying:
    python visualize_graph.py --all-clients --save outputs/graphs --no-show
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from models.client_model import ClientModel
from models.graph_constructor import build_graph, GraphData

# ── constants ─────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "type_id", "n_params(M)", "in_size", "out_size",
    "depth_ratio", "is_encoder", "w_mean", "w_std", "w_norm",
    "is_attn", "is_ff", "is_norm", "is_embed", "has_bias",
    "rel_size", "reserved",
]

# feat indices for category detection
_F_ATTN  = 9
_F_FF    = 10
_F_NORM  = 11
_F_EMBED = 12

CATEGORY_COLORS = {
    "attention":   "#4C72B0",
    "feedforward": "#DD8452",
    "norm":        "#55A868",
    "embedding":   "#C44E52",
    "other":       "#8172B2",
}


def _node_category(feat_row: np.ndarray) -> str:
    if feat_row[_F_ATTN]  > 0.5: return "attention"
    if feat_row[_F_FF]    > 0.5: return "feedforward"
    if feat_row[_F_NORM]  > 0.5: return "norm"
    if feat_row[_F_EMBED] > 0.5: return "embedding"
    return "other"


# ── graph layout ──────────────────────────────────────────────────────────────

def _build_layout(graph_data: GraphData, feats: np.ndarray) -> dict:
    """
    Two-column layout: encoder nodes on x=0, decoder nodes on x=1.
    Within each column, nodes are ordered top-to-bottom by their traversal index.
    Small x-jitter separates nodes that overlap.
    """
    enc_indices = [i for i in range(graph_data.num_nodes) if feats[i, 5] > 0.5]
    dec_indices = [i for i in range(graph_data.num_nodes) if feats[i, 5] <= 0.5]

    pos = {}
    for col_x, indices in [(0.0, enc_indices), (1.0, dec_indices)]:
        n = max(len(indices), 1)
        for rank, idx in enumerate(indices):
            # spread vertically; add tiny x-jitter to separate overlapping nodes
            y = 1.0 - rank / n
            jitter = (rank % 3 - 1) * 0.04
            pos[idx] = (col_x + jitter, y)
    return pos


# ── per-model figure ──────────────────────────────────────────────────────────

def plot_model(graph_data: GraphData, ax_graph, ax_heatmap, ax_bar) -> None:
    feats = graph_data.data.x.cpu().numpy()          # [N, 16]
    edge_index = graph_data.data.edge_index.cpu().numpy()  # [2, E]
    N = graph_data.num_nodes

    # ── 1. Architecture graph ─────────────────────────────────────────────
    G = nx.DiGraph()
    G.add_nodes_from(range(N))

    # Add edges, skipping self-loops
    src, dst = edge_index
    for s, d in zip(src.tolist(), dst.tolist()):
        if s != d:
            G.add_edge(s, d)

    categories = [_node_category(feats[i]) for i in range(N)]
    node_colors = [CATEGORY_COLORS[c] for c in categories]

    # Node size proportional to log(n_params+1), clamped for readability
    raw_sizes = np.log1p(feats[:, 1] * 1e6)
    node_sizes = np.clip(raw_sizes * 8, 4, 120).tolist()

    pos = _build_layout(graph_data, feats)

    nx.draw_networkx_nodes(
        G, pos, ax=ax_graph,
        node_color=node_colors, node_size=node_sizes, alpha=0.85,
    )
    nx.draw_networkx_edges(
        G, pos, ax=ax_graph,
        edge_color="#aaaaaa", width=0.3, alpha=0.4, arrows=False,
    )
    ax_graph.set_title(
        f"{graph_data.model_name}\n{N} nodes  {edge_index.shape[1]} edges (incl. self-loops)",
        fontsize=9, pad=6,
    )
    ax_graph.set_xlabel("← encoder    decoder →", fontsize=7)
    ax_graph.axis("off")

    # Legend
    legend_patches = [
        mpatches.Patch(color=c, label=k) for k, c in CATEGORY_COLORS.items()
    ]
    ax_graph.legend(handles=legend_patches, fontsize=6, loc="lower right",
                    framealpha=0.7, handlelength=1)

    # ── 2. Node-feature heatmap ───────────────────────────────────────────
    # Show every 4th node if there are many, so the heatmap is readable
    step = max(1, N // 80)
    feat_subset = feats[::step]                       # [M, 16]

    # Normalise each feature column to [0, 1] for visual comparison
    col_min = feat_subset.min(axis=0, keepdims=True)
    col_max = feat_subset.max(axis=0, keepdims=True)
    feat_norm = (feat_subset - col_min) / np.clip(col_max - col_min, 1e-8, None)

    im = ax_heatmap.imshow(feat_norm.T, aspect="auto", cmap="viridis",
                           interpolation="nearest")
    ax_heatmap.set_yticks(range(16))
    ax_heatmap.set_yticklabels(FEATURE_NAMES, fontsize=6)
    ax_heatmap.set_xlabel(f"Node index (every {step}th)", fontsize=7)
    ax_heatmap.set_title("Node feature heatmap (normalised)", fontsize=9)
    plt.colorbar(im, ax=ax_heatmap, fraction=0.03, pad=0.02)

    # ── 3. Layer-type distribution bar ───────────────────────────────────
    from collections import Counter
    counts = Counter(categories)
    labels = list(CATEGORY_COLORS.keys())
    values = [counts.get(k, 0) for k in labels]
    colors = [CATEGORY_COLORS[k] for k in labels]

    bars = ax_bar.barh(labels, values, color=colors, edgecolor="white", height=0.6)
    for bar, val in zip(bars, values):
        if val > 0:
            ax_bar.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                        str(val), va="center", fontsize=8)
    ax_bar.set_xlabel("Node count", fontsize=7)
    ax_bar.set_title("Layer-type distribution", fontsize=9)
    ax_bar.set_xlim(0, max(values) * 1.18)
    ax_bar.invert_yaxis()


# ── CLI & entry point ─────────────────────────────────────────────────────────

DEFAULT_CLIENTS = [
    ("google/flan-t5-small",    "t5",   ["q", "v"],             512),
    ("facebook/bart-base",      "bart", ["q_proj", "v_proj"],   768),
    ("allenai/led-base-16384",  "led",  ["q_proj", "v_proj"],   768),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize LM architecture graphs")
    p.add_argument("--model",   help="HuggingFace model ID (single-model mode)")
    p.add_argument("--family",  choices=["t5", "bart", "led"],
                   help="Model family (single-model mode)")
    p.add_argument("--targets", nargs="+", help="LoRA target modules")
    p.add_argument("--d-model", type=int,  help="Model hidden dim")
    p.add_argument("--lora-r",       type=int,   default=16)
    p.add_argument("--lora-alpha",   type=int,   default=32)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--all-clients",  action="store_true",
                   help="Visualize all three default clients")
    p.add_argument("--save",  default="", metavar="DIR",
                   help="Directory to save PNG files (empty = don't save)")
    p.add_argument("--no-show", action="store_true",
                   help="Do not call plt.show() (useful on Colab without display)")
    p.add_argument("--device", default="cpu",
                   help="Device for model loading (cpu is fine for graph building)")
    return p.parse_args()


def _build_graph_for(
    model_name: str,
    family: str,
    targets: list,
    d_model: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    device: torch.device,
) -> GraphData:
    print(f"  Loading {model_name} …")
    client_model = ClientModel(
        model_name=model_name,
        model_family=family,
        lora_target_modules=targets,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        device=device,
    )
    graph_data = build_graph(
        peft_model=client_model.model,
        model_name=model_name,
        lora_alpha=lora_alpha,
        lora_r=lora_r,
        device=device,
    )
    del client_model
    torch.cuda.empty_cache()
    return graph_data


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    # Determine which models to visualize
    if args.all_clients:
        specs = DEFAULT_CLIENTS
    elif args.model:
        if not all([args.family, args.targets, args.d_model]):
            raise ValueError("--model requires --family, --targets, and --d-model")
        specs = [(args.model, args.family, args.targets, args.d_model)]
    else:
        raise ValueError("Provide --model ... or --all-clients")

    n_models = len(specs)
    fig, axes = plt.subplots(
        3, n_models,
        figsize=(6 * n_models, 16),
        gridspec_kw={"height_ratios": [3, 1.2, 0.8]},
    )
    # Normalise axes shape to always be 2-D
    if n_models == 1:
        axes = axes.reshape(3, 1)

    fig.suptitle("Architecture Graph Analysis", fontsize=13, y=0.99)

    for col, (model_name, family, targets, d_model) in enumerate(specs):
        graph_data = _build_graph_for(
            model_name, family, targets, d_model,
            args.lora_r, args.lora_alpha, args.lora_dropout, device,
        )
        plot_model(
            graph_data,
            ax_graph=axes[0, col],
            ax_heatmap=axes[1, col],
            ax_bar=axes[2, col],
        )

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    if args.save:
        save_dir = Path(args.save)
        save_dir.mkdir(parents=True, exist_ok=True)
        suffix = "all_clients" if args.all_clients else args.model.replace("/", "_")
        out_path = save_dir / f"architecture_graph_{suffix}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {out_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
