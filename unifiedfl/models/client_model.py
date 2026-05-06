from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn as nn

# Force-import torch.distributed.tensor so PEFT's DTensor isinstance check
# can resolve at LoRA-injection time. PEFT >= 0.13 unconditionally references
# torch.distributed.tensor.DTensor inside _get_in_out_features when its
# _torch_supports_distributed flag is True, but does NOT import the submodule
# itself — and the submodule is not auto-loaded on every torch build (notably
# CPU-only builds), so peft can crash with
#   AttributeError: module 'torch.distributed' has no attribute 'tensor'
# the first time we call get_peft_model. Importing it eagerly here populates
# the attribute path safely.
try:
    import torch.distributed.tensor  # noqa: F401
except ImportError:
    pass

from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, LEDForConditionalGeneration

logger = logging.getLogger("federated_qa")


class ClientModel(nn.Module):
    """
    Wraps a HuggingFace Seq2Seq LM with LoRA adapters via PEFT.

    Base model weights are fully frozen; only LoRA adapter weights are
    trainable. The base model is loaded in FP32 (FP16 attention overflows
    on T5 before padding masks are applied); BF16 autocast in the trainer
    handles the mixed-precision path on hardware that supports it.

    LED-specific handling:
        - global_attention_mask (attention on first token) is added
          automatically in forward() and generate().
        - decoder_start_token_id is set to tokenizer.bos_token_id.
    """

    def __init__(
        self,
        model_name: str,
        model_family: str,
        lora_target_modules: List[str],
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.model_family = model_family
        self.device = device

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Base model in FP32 with frozen parameters.
        # Loading in FP16 causes T5 attention scores to overflow (inf - inf = NaN)
        # before padding masks are applied. The standard PyTorch mixed-precision
        # pattern keeps parameters in FP32 and lets autocast cast activations to
        # FP16 only during the forward pass.
        base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        for param in base_model.parameters():
            param.requires_grad = False

        # Apply LoRA — adapters are FP32 by default, compatible with GradScaler
        lora_cfg = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=lora_target_modules,
        )
        self.model = get_peft_model(base_model, lora_cfg)
        self.model.to(device)

        # Disable gradient checkpointing if the base model has it enabled.
        # Some models (notably ProphetNet) ship with gradient_checkpointing=True
        # in their config. Checkpointing replays forward passes during backward,
        # which is incompatible with our FiLM forward-hook design — each hook
        # would fire again with stale GNN-derived state. Mixed-precision
        # (BF16) gives us the memory savings we'd otherwise need checkpointing for.
        if hasattr(self.model, "gradient_checkpointing_disable"):
            try:
                self.model.gradient_checkpointing_disable()
            except (ValueError, AttributeError):
                # Some PEFT/transformers versions raise if checkpointing was
                # never enabled in the first place — that's fine.
                pass

        # Detect LED
        try:
            underlying = self.model.base_model.model
        except AttributeError:
            underlying = self.model
        self.is_led: bool = isinstance(underlying, LEDForConditionalGeneration)

        if self.is_led and self.tokenizer.bos_token_id is not None:
            self.model.config.decoder_start_token_id = self.tokenizer.bos_token_id

        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(
            f"ClientModel({model_name}): {n_trainable:,} trainable params, "
            f"is_led={self.is_led}"
        )

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> object:
        """Run model forward pass, adding LED global attention mask if needed."""
        if self.is_led:
            global_attention_mask = self._make_global_mask(input_ids)
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
                labels=labels,
            )
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    # ── Generation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        """Generate sequences, adding LED global attention mask when required."""
        if self.is_led:
            global_attention_mask = self._make_global_mask(input_ids)
            return self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
                **kwargs,
            )
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

    # ── Parameter access ──────────────────────────────────────────────────────

    def get_lora_params(self) -> List[nn.Parameter]:
        """Return only the trainable LoRA adapter parameters."""
        return [p for p in self.model.parameters() if p.requires_grad]

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _make_global_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Build global attention mask with attention on token 0 only."""
        mask = torch.zeros_like(input_ids)
        mask[:, 0] = 1
        if mask.shape != input_ids.shape:
            raise RuntimeError(
                f"LED global_attention_mask shape {mask.shape} "
                f"mismatches input_ids shape {input_ids.shape}"
            )
        return mask
