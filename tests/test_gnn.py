"""
Tests for unifiedfl/models/gnn.py — ArchitectureGNN.

CPU-only, no model downloads. Builds tiny synthetic graphs.
"""

from __future__ import annotations

import pytest
import torch

try:
    from torch_geometric.data import Data
except ImportError:
    pytest.skip("torch_geometric not installed", allow_module_level=True)

from models.gnn import ArchitectureGNN


def _tiny_graph(n_nodes: int = 5, in_channels: int = 16) -> Data:
    """Path graph: 0 - 1 - 2 - ... - n-1, with bidirectional edges."""
    x = torch.randn(n_nodes, in_channels)
    src = list(range(n_nodes - 1)) + list(range(1, n_nodes))
    dst = list(range(1, n_nodes)) + list(range(n_nodes - 1))
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)


class TestArchitectureGNN:
    def test_output_shapes(self):
        torch.manual_seed(0)
        gnn = ArchitectureGNN(in_channels=16, hidden=64, heads=4)
        data = _tiny_graph(n_nodes=5, in_channels=16)
        node_emb, graph_emb = gnn(data)
        assert node_emb.shape == (5, 64), f"got {node_emb.shape}"
        assert graph_emb.shape == (1, 64), f"got {graph_emb.shape}"

    def test_default_batch_vector_is_built_when_missing(self):
        # data has no .batch attr — GNN should default to zeros and not crash.
        torch.manual_seed(0)
        gnn = ArchitectureGNN()
        data = _tiny_graph()
        # Confirm batch is genuinely absent (or None).
        assert getattr(data, "batch", None) is None
        out_node, out_graph = gnn(data)
        assert out_graph.shape[0] == 1  # single graph -> 1 row

    def test_deterministic_when_eval_mode_and_seeded(self):
        torch.manual_seed(42)
        gnn = ArchitectureGNN()
        gnn.eval()
        data = _tiny_graph()
        out_a, _ = gnn(data)
        out_b, _ = gnn(data)
        # Eval mode disables dropout — outputs identical on the same input.
        assert torch.allclose(out_a, out_b)

    def test_dropout_active_in_train_mode(self):
        torch.manual_seed(42)
        gnn = ArchitectureGNN(dropout=0.5)
        gnn.train()
        data = _tiny_graph()
        out_a, _ = gnn(data)
        out_b, _ = gnn(data)
        # Train mode + nontrivial dropout -> outputs typically differ.
        assert not torch.allclose(out_a, out_b)

    def test_grad_flows_to_input(self):
        gnn = ArchitectureGNN()
        data = _tiny_graph()
        data.x.requires_grad_()
        node_emb, _ = gnn(data)
        node_emb.sum().backward()
        assert data.x.grad is not None
        assert torch.any(data.x.grad != 0)

    def test_grad_flows_to_gnn_params(self):
        gnn = ArchitectureGNN()
        data = _tiny_graph()
        node_emb, _ = gnn(data)
        node_emb.sum().backward()
        any_grad = any(
            p.grad is not None and torch.any(p.grad != 0)
            for p in gnn.parameters()
        )
        assert any_grad

    def test_n_node_independence(self):
        # Same architecture should accept varying graph sizes (within reason).
        torch.manual_seed(0)
        gnn = ArchitectureGNN()
        for n in (2, 5, 20):
            data = _tiny_graph(n_nodes=n)
            node_emb, graph_emb = gnn(data)
            assert node_emb.shape == (n, 64)
            assert graph_emb.shape == (1, 64)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
