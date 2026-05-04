from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class LoRAConfig:
    """LoRA adapter configuration shared by all clients."""

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    bias: str = "none"
    task_type: str = "SEQ_2_SEQ_LM"


@dataclass
class GNNConfig:
    """Graph Attention Network configuration."""

    in_channels: int = 16
    hidden: int = 64
    heads: int = 4
    layers: int = 3
    dropout: float = 0.1


@dataclass
class FiLMConfig:
    """FiLM conditioning adapter configuration."""

    hidden: int = 128
    alpha_init: float = 0.0


@dataclass
class ClientConfig:
    """Per-client model and data configuration."""

    client_id: int
    model_name: str
    data_path: str
    d_model: int
    lora_target_modules: List[str]
    model_family: str


@dataclass
class Config:
    """Top-level configuration for the entire federated system."""

    seed: int = 42
    num_rounds: int = 20
    local_epochs: int = 3
    batch_size: int = 4
    max_input_len: int = 512
    max_target_len: int = 128
    lr_lora: float = 3e-4
    lr_gnn: float = 1e-3
    lr_film: float = 1e-3
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    eval_every_n: int = 5
    node_feat_dim: int = 16
    device: str = "cuda"
    output_dir: str = "outputs/"
    conditioning: str = "baseline"

    lora: LoRAConfig = field(default_factory=LoRAConfig)
    gnn: GNNConfig = field(default_factory=GNNConfig)
    film: FiLMConfig = field(default_factory=FiLMConfig)

    clients: List[ClientConfig] = field(
        default_factory=lambda: [
            ClientConfig(
                client_id=0,
                model_name="google/flan-t5-small",
                data_path="data/ML_QA_LectureNotes_MIT.json",
                d_model=512,
                lora_target_modules=["q", "v"],
                model_family="t5",
            ),
            ClientConfig(
                client_id=1,
                model_name="facebook/bart-base",
                data_path="data/ML_QA_LectureNotes_StanfordCS229.json",
                d_model=768,
                lora_target_modules=["q_proj", "v_proj"],
                model_family="bart",
            ),
            ClientConfig(
                client_id=2,
                model_name="allenai/led-base-16384",
                data_path="data/ML_QA_Papers_v2.json",
                d_model=768,
                lora_target_modules=["q_proj", "v_proj"],
                model_family="led",
            ),
        ]
    )
