from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops

# ── safe imports of architecture-specific layer types ────────────────────────
try:
    from transformers.models.t5.modeling_t5 import (
        T5Attention,
        T5LayerFF,
        T5LayerNorm,
    )
except ImportError:  # pragma: no cover
    T5Attention = T5LayerFF = T5LayerNorm = None  # type: ignore[assignment,misc]

try:
    from transformers.models.bart.modeling_bart import (
        BartAttention,
        BartEncoderLayer,
        BartDecoderLayer,
    )
except ImportError:  # pragma: no cover
    BartAttention = BartEncoderLayer = BartDecoderLayer = None  # type: ignore[assignment,misc]

try:
    from transformers.models.led.modeling_led import (
        LEDEncoderLayer,
        LEDDecoderLayer,
    )
    try:
        from transformers.models.led.modeling_led import LEDAttention
    except ImportError:
        LEDAttention = None  # type: ignore[assignment,misc]
    try:
        from transformers.models.led.modeling_led import LEDEncoderAttention
    except ImportError:
        LEDEncoderAttention = None  # type: ignore[assignment,misc]
    try:
        from transformers.models.led.modeling_led import LEDDecoderAttention
    except ImportError:
        LEDDecoderAttention = None  # type: ignore[assignment,misc]
except ImportError:  # pragma: no cover
    LEDEncoderLayer = LEDDecoderLayer = LEDAttention = None  # type: ignore[assignment,misc]
    LEDEncoderAttention = LEDDecoderAttention = None  # type: ignore[assignment,misc]

try:
    from transformers.models.longformer.modeling_longformer import LongformerAttention
except ImportError:  # pragma: no cover
    LongformerAttention = None  # type: ignore[assignment,misc]

logger = logging.getLogger("federated_qa")

# ── Layer-type ID mapping ────────────────────────────────────────────────────

def _build_type_map() -> Dict[type, int]:
    mapping: Dict[type, int] = {nn.Linear: 1, nn.LayerNorm: 2, nn.Embedding: 3, nn.MultiheadAttention: 4}
    for cls, tid in [
        (T5Attention, 5), (T5LayerFF, 6), (T5LayerNorm, 7),
        (BartAttention, 8), (BartEncoderLayer, 9), (BartDecoderLayer, 10),
        (LEDEncoderLayer, 11), (LEDDecoderLayer, 12),
        (LongformerAttention, 13), (LEDAttention, 14),
        (LEDEncoderAttention, 15), (LEDDecoderAttention, 16),
    ]:
        if cls is not None:
            mapping[cls] = tid
    return mapping


LAYER_TYPE_MAP: Dict[type, int] = _build_type_map()

_MEANINGFUL = tuple(t for t in LAYER_TYPE_MAP if t is not None)


def _is_meaningful(module: nn.Module) -> bool:
    return isinstance(module, _MEANINGFUL)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _layer_type_id(module: nn.Module) -> int:
    for cls, tid in LAYER_TYPE_MAP.items():
        if type(module) is cls:
            return tid
    if isinstance(module, _MEANINGFUL):
        return LAYER_TYPE_MAP.get(type(module), 0)
    return 0


def _size_features(module: nn.Module) -> Tuple[float, float]:
    """Return (in_size/1024, out_size/1024) as best-effort for any layer type."""
    if isinstance(module, nn.Linear):
        return module.in_features / 1024.0, module.out_features / 1024.0
    if isinstance(module, nn.LayerNorm):
        sz = module.normalized_shape[0] if module.normalized_shape else 0
        return sz / 1024.0, 0.0
    if isinstance(module, nn.Embedding):
        return module.num_embeddings / 1024.0, module.embedding_dim / 1024.0
    if isinstance(module, nn.MultiheadAttention):
        return module.embed_dim / 1024.0, float(module.num_heads) / 1024.0
    # Generic fallback: scan for common attributes
    feat2 = 0.0
    for attr in ("d_model", "embed_dim", "hidden_size"):
        if hasattr(module, attr):
            feat2 = getattr(module, attr) / 1024.0
            break
    feat3 = 0.0
    for attr in ("d_ff", "intermediate_size", "num_heads", "n_heads"):
        if hasattr(module, attr):
            feat3 = getattr(module, attr) / 1024.0
            break
    return feat2, feat3


def _is_type(module: nn.Module, *types: Optional[type]) -> bool:
    valid = tuple(t for t in types if t is not None)
    return isinstance(module, valid) if valid else False


def _node_flags(module: nn.Module) -> Tuple[float, float, float, float, float]:
    """Return (is_attention, is_ff, is_norm, is_embed, has_bias)."""
    is_attn = float(
        _is_type(module, T5Attention, BartAttention, LEDAttention,
                 LEDEncoderAttention, LEDDecoderAttention,
                 LongformerAttention, nn.MultiheadAttention)
    )
    is_ff = float(_is_type(module, T5LayerFF, nn.Linear))
    is_norm = float(_is_type(module, nn.LayerNorm, T5LayerNorm))
    is_emb = float(isinstance(module, nn.Embedding))
    has_bias = float(
        (isinstance(module, nn.Linear) and module.bias is not None)
        or (isinstance(module, nn.Embedding) and False)  # embeddings have no bias
        or (isinstance(module, nn.LayerNorm) and module.bias is not None)
    )
    return is_attn, is_ff, is_norm, is_emb, has_bias


def get_effective_weight_stats(
    module_name: str,
    base_module: nn.Module,
    peft_model: nn.Module,
    lora_alpha: int,
    lora_r: int,
) -> Tuple[float, float, float]:
    """
    Compute weight statistics on W_effective = W_base + (alpha/r) * (B @ A).

    Falls back to base weight stats when no LoRA layers are attached.

    Returns:
        (mean, std, l2_norm_normalised)
    """
    if not (hasattr(base_module, "weight") and base_module.weight is not None):
        return 0.0, 0.0, 0.0

    W = base_module.weight.data.float()

    # module_name is the full PEFT-prefixed path (e.g. "base_model.model.encoder...q").
    # The LoRA sub-modules live directly underneath it, so we do NOT prepend the
    # base_model.model. prefix a second time.
    lora_key_A = f"{module_name}.lora_A.default"
    lora_key_B = f"{module_name}.lora_B.default"

    lora_A: Optional[torch.Tensor] = None
    lora_B: Optional[torch.Tensor] = None
    for name, mod in peft_model.named_modules():
        if name == lora_key_A and hasattr(mod, "weight"):
            lora_A = mod.weight.data.float()
        if name == lora_key_B and hasattr(mod, "weight"):
            lora_B = mod.weight.data.float()
        if lora_A is not None and lora_B is not None:
            break

    if lora_A is not None and lora_B is not None:
        scaling = lora_alpha / lora_r
        delta_W = scaling * (lora_B @ lora_A)
        if delta_W.shape == W.shape:
            W = W + delta_W

    num_params = W.numel()
    mean = W.mean().item()
    std = W.std().item()
    norm = (W.norm() / (num_params ** 0.5)).item()
    return mean, std, norm


# ── GraphData dataclass ───────────────────────────────────────────────────────

@dataclass
class GraphData:
    """Container for a client's architecture graph."""

    data: Data                              # torch_geometric Data (x, edge_index)
    layer_to_node_idx: Dict[str, int]       # module_name → node index
    node_modules: List[Tuple[str, nn.Module]]  # ordered (name, module) per node
    num_nodes: int
    model_name: str


# ── Graph builder ────────────────────────────────────────────────────────────

def build_graph(
    peft_model: nn.Module,
    model_name: str,
    lora_alpha: int,
    lora_r: int,
    device: torch.device,
) -> GraphData:
    """
    Convert a PEFT-wrapped LM into a layer-level module graph.

    Node features (dim=16) are described in Section 6.2 of the spec.
    Edges connect consecutive meaningful modules (sequential ordering
    from named_modules()), plus self-loops on every node.

    The edge topology is fixed after construction. Only feat[6,7,8]
    (effective-weight statistics) change per round — see refresh_graph_features().

    Args:
        peft_model: the PEFT-wrapped model (PeftModelForSeq2SeqLM)
        model_name: human-readable identifier
        lora_alpha: LoRA alpha scaling factor
        lora_r: LoRA rank
        device: torch device for tensors

    Returns:
        GraphData with immutable topology and initial node features.

    Raises:
        ValueError: if the filtered graph has zero nodes.
    """
    # Collect meaningful modules in traversal order
    node_modules: List[Tuple[str, nn.Module]] = []
    for name, module in peft_model.named_modules():
        if _is_meaningful(module):
            node_modules.append((name, module))

    num_nodes = len(node_modules)
    if num_nodes == 0:
        raise ValueError(
            f"Graph construction produced 0 nodes for model {model_name}. "
            "Check layer filtering logic."
        )

    layer_to_node_idx: Dict[str, int] = {
        name: idx for idx, (name, _) in enumerate(node_modules)
    }

    total_params = sum(p.numel() for p in peft_model.parameters())

    # Build node feature matrix
    x = torch.zeros(num_nodes, 16, dtype=torch.float32)
    for idx, (name, module) in enumerate(node_modules):
        type_id = _layer_type_id(module)
        n_params = sum(p.numel() for p in module.parameters())
        feat2, feat3 = _size_features(module)
        depth_ratio = idx / max(num_nodes - 1, 1)
        is_encoder = 0.0 if "decoder" in name else 1.0
        mean, std, norm = get_effective_weight_stats(
            name, module, peft_model, lora_alpha, lora_r
        )
        is_attn, is_ff, is_norm, is_emb, has_bias = _node_flags(module)
        rel_size = n_params / max(total_params, 1)

        x[idx, 0] = float(type_id)
        x[idx, 1] = n_params / 1e6
        x[idx, 2] = feat2
        x[idx, 3] = feat3
        x[idx, 4] = depth_ratio
        x[idx, 5] = is_encoder
        x[idx, 6] = mean
        x[idx, 7] = std
        x[idx, 8] = norm
        x[idx, 9] = is_attn
        x[idx, 10] = is_ff
        x[idx, 11] = is_norm
        x[idx, 12] = is_emb
        x[idx, 13] = has_bias
        x[idx, 14] = rel_size
        x[idx, 15] = 0.0

    # Sequential edges: i → i+1
    if num_nodes > 1:
        src = list(range(num_nodes - 1))
        dst = list(range(1, num_nodes))
        edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)

    # Self-loops
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

    data = Data(x=x.to(device), edge_index=edge_index.to(device))

    logger.info(
        f"Graph built for {model_name}: {num_nodes} nodes, "
        f"{edge_index.size(1)} edges (incl. self-loops)"
    )

    return GraphData(
        data=data,
        layer_to_node_idx=layer_to_node_idx,
        node_modules=node_modules,
        num_nodes=num_nodes,
        model_name=model_name,
    )


def refresh_graph_features(
    graph_data: GraphData,
    peft_model: nn.Module,
    lora_alpha: int,
    lora_r: int,
) -> None:
    """
    Update feat[6,7,8] (effective weight stats) in-place for all nodes.

    Call this at the start of each federation round before running the GNN.
    The edge topology is never touched.
    """
    for idx, (name, module) in enumerate(graph_data.node_modules):
        mean, std, norm = get_effective_weight_stats(
            name, module, peft_model, lora_alpha, lora_r
        )
        graph_data.data.x[idx, 6] = mean
        graph_data.data.x[idx, 7] = std
        graph_data.data.x[idx, 8] = norm
