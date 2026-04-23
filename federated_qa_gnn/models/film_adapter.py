from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger("federated_qa")

# Safe architecture-layer imports ─────────────────────────────────────────────
try:
    from transformers.models.t5.modeling_t5 import T5Block
except ImportError:
    T5Block = None  # type: ignore[assignment,misc]

try:
    from transformers.models.bart.modeling_bart import (
        BartEncoderLayer,
        BartDecoderLayer,
    )
except ImportError:
    BartEncoderLayer = BartDecoderLayer = None  # type: ignore[assignment,misc]

try:
    from transformers.models.led.modeling_led import (
        LEDEncoderLayer,
        LEDDecoderLayer,
    )
except ImportError:
    LEDEncoderLayer = LEDDecoderLayer = None  # type: ignore[assignment,misc]

_FAMILY_TARGETS: Dict[str, Tuple[type, ...]] = {}

def _build_family_targets() -> None:
    for key, types in [
        ("t5",   [T5Block]),
        ("bart", [BartEncoderLayer, BartDecoderLayer]),
        ("led",  [LEDEncoderLayer, LEDDecoderLayer]),
    ]:
        valid = tuple(t for t in types if t is not None)
        if valid:
            _FAMILY_TARGETS[key] = valid

_build_family_targets()


class FiLMAdapter(nn.Module):
    """
    FiLM conditioning adapter that modulates each transformer layer's
    hidden states using scale (γ) and shift (β) vectors derived from
    GNN node embeddings.

    Runs in FP32. Outputs are cast to match the LM's hidden state dtype
    before modulation is applied.

    Modulation formula (residual):
        h̃_l = h_l + α * (γ_l ⊙ h_l + β_l)

    where α is a learnable scalar initialized to 0 (identity at init).

    Parameters
    ----------
    d_model : int
        Hidden dimension of the target LM (512 for T5-small, 768 for BART/LED).
    film_hidden : int
        Hidden size of the 2-layer MLP.
    alpha_init : float
        Initial value of the learnable residual scale (0.0 recommended).
    model_family : str
        One of 't5', 'bart', 'led' — determines which transformer blocks
        receive hooks.
    """

    def __init__(
        self,
        d_model: int,
        film_hidden: int = 128,
        alpha_init: float = 0.0,
        model_family: str = "t5",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.model_family = model_family

        self.mlp = nn.Sequential(
            nn.Linear(64, film_hidden),
            nn.ReLU(),
            nn.Linear(film_hidden, 2 * d_model),
        )
        # Learnable residual scale — starts as identity (no modulation)
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

        self._hook_handles: List[torch.utils.hooks.RemovableHook] = []
        # Mutable containers so hooks always see the latest tensors
        self._node_embeddings: Optional[torch.Tensor] = None
        self._graph_embedding: Optional[torch.Tensor] = None
        self._layer_to_node_idx: Dict[str, int] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def update_embeddings(
        self,
        node_embeddings: torch.Tensor,
        graph_embedding: torch.Tensor,
        layer_to_node_idx: Dict[str, int],
    ) -> None:
        """Store the latest GNN outputs for use by active hooks."""
        self._node_embeddings = node_embeddings
        self._graph_embedding = graph_embedding
        self._layer_to_node_idx = layer_to_node_idx

    def register_hooks(
        self,
        model: nn.Module,
        node_embeddings: torch.Tensor,
        layer_to_node_idx: Dict[str, int],
        graph_embedding: torch.Tensor,
    ) -> None:
        """
        Register forward hooks on all transformer blocks of the client's LM.

        Hooks intercept the block output, apply FiLM modulation, and return
        the modulated hidden state while preserving any extra tuple outputs
        (e.g. attention weights).

        Args:
            model: the PEFT-wrapped LM
            node_embeddings: [N, 64] tensor from GNN (kept live for grad flow)
            layer_to_node_idx: maps module name → node index
            graph_embedding: [1, 64] fallback when a block has no node index
        """
        self.update_embeddings(node_embeddings, graph_embedding, layer_to_node_idx)

        target_types = _FAMILY_TARGETS.get(self.model_family, ())
        if not target_types:
            logger.warning(f"No hook targets found for model family '{self.model_family}'")
            return

        for name, module in model.named_modules():
            if isinstance(module, target_types):
                handle = module.register_forward_hook(self._make_hook(name))
                self._hook_handles.append(handle)

        logger.debug(
            f"FiLM registered {len(self._hook_handles)} hooks "
            f"for family '{self.model_family}'"
        )

    def remove_hooks(self) -> None:
        """Remove all registered forward hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def get_alpha(self) -> float:
        """Return the current alpha scalar value."""
        return self.alpha.item()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _make_hook(self, layer_name: str) -> Callable:
        """Create a closure that captures `layer_name` for node-index lookup."""

        def hook(
            module: nn.Module,
            inp: tuple,
            output: object,
        ) -> object:
            if self._node_embeddings is None:
                return output

            # Resolve node embedding for this layer
            if layer_name in self._layer_to_node_idx:
                node_idx = self._layer_to_node_idx[layer_name]
                z = self._node_embeddings[node_idx]  # [64], FP32, with grad
            else:
                z = self._graph_embedding[0]  # [64], fallback

            # Run FiLM MLP in FP32 (disable autocast to prevent FP16 coercion)
            with torch.amp.autocast("cuda", enabled=False):
                z_f = z.float()
                film_out = self.mlp(z_f)  # [2 * d_model], FP32
                gamma, beta = film_out.chunk(2, dim=-1)  # [d_model] each

            # Extract hidden state from (possibly tuple) output
            if isinstance(output, tuple):
                h = output[0]
                rest = output[1:]
            else:
                h = output
                rest = None

            if not isinstance(h, torch.Tensor):
                return output

            if h.dim() < 2:
                return output

            # Cast modulation tensors to match LM hidden state dtype + device
            tgt_dtype = h.dtype
            tgt_device = h.device
            gamma = gamma.to(dtype=tgt_dtype, device=tgt_device)
            beta = beta.to(dtype=tgt_dtype, device=tgt_device)
            alpha = self.alpha.to(dtype=tgt_dtype, device=tgt_device)

            # Broadcast over (batch, seq) dimensions
            gamma = gamma.view(1, 1, -1)
            beta = beta.view(1, 1, -1)

            h_mod = h + alpha * (gamma * h + beta)

            if rest is not None:
                return (h_mod,) + rest
            return h_mod

        return hook
