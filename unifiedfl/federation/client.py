from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn

from models.client_model import ClientModel
from models.film_adapter import FiLMAdapter
from models.gnn import ArchitectureGNN
from models.graph_constructor import GraphData

logger = logging.getLogger("federated_qa")


class FederatedClient:
    """
    Encapsulates all state for one client in the federated system:
      - Language model (with LoRA)
      - Architecture graph
      - GNN (shared weights, aggregated by server)
      - FiLM adapter (client-local)
      - Train / val / test split samples

    The GNN parameters are the only component sent to the server each round.
    LoRA and FiLM weights remain strictly client-local.
    """

    def __init__(
        self,
        client_id: int,
        client_model: ClientModel,
        gnn: ArchitectureGNN,
        film_adapter: FiLMAdapter,
        graph_data: GraphData,
        train_samples: List[Dict[str, str]],
        val_samples: List[Dict[str, str]],
        test_samples: List[Dict[str, str]],
        device: torch.device,
    ) -> None:
        self.client_id = client_id
        self.client_model = client_model
        self.gnn = gnn
        self.film_adapter = film_adapter
        self.graph_data = graph_data
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples
        self.device = device

    @property
    def n_train_samples(self) -> int:
        """Number of flattened qa-pair training samples."""
        return len(self.train_samples)

    # ── GNN federation interface ──────────────────────────────────────────────

    def get_gnn_state_dict(self) -> Dict[str, torch.Tensor]:
        """Return a CPU copy of GNN parameters for aggregation.

        .clone() ensures the returned tensors don't share storage with the
        live module — without it, in-place ops on the live params (e.g. in
        the next training step) would mutate the snapshot the server is
        about to aggregate.
        """
        return {k: v.detach().cpu().clone() for k, v in self.gnn.state_dict().items()}

    def load_gnn_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load aggregated GNN parameters from the server."""
        self.gnn.load_state_dict(
            {k: v.to(self.device) for k, v in state_dict.items()}
        )

    # ── Model persistence ─────────────────────────────────────────────────────

    def save(self, output_dir: str) -> None:
        """Save LoRA model, GNN, and FiLM adapter to disk."""
        base = Path(output_dir)

        lm_dir = base / f"client_{self.client_id}_final"
        lm_dir.mkdir(parents=True, exist_ok=True)
        self.client_model.model.save_pretrained(str(lm_dir))
        logger.info(f"Client {self.client_id} LoRA model saved to {lm_dir}")

        gnn_path = base / "gnn_final.pt"
        torch.save(self.gnn.state_dict(), gnn_path)

        film_path = base / f"film_adapter_{self.client_id}_final.pt"
        torch.save(self.film_adapter.state_dict(), film_path)
        logger.info(
            f"Client {self.client_id} GNN saved to {gnn_path}, "
            f"FiLM saved to {film_path}"
        )
