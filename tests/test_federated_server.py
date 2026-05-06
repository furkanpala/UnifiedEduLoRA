"""
Tests for unifiedfl/federation/server.py — weighted FedAvg of GNN state dicts.

Server is GPU-free and doesn't load any models, so these run on CPU.
"""

from __future__ import annotations

import pytest
import torch

from federation.server import FederatedServer


def _state(value: float, shape=(2, 3)) -> dict:
    return {
        "layer1.weight": torch.full(shape, value, dtype=torch.float32),
        "layer1.bias":   torch.full((shape[0],), value, dtype=torch.float32),
    }


class TestFedAvg:
    def test_equal_weights_is_arithmetic_mean(self):
        s = FederatedServer()
        agg = s.aggregate(
            client_states=[_state(1.0), _state(3.0)],
            weights=[1, 1],
        )
        assert torch.allclose(agg["layer1.weight"], torch.full((2, 3), 2.0))
        assert torch.allclose(agg["layer1.bias"],   torch.full((2,),   2.0))

    def test_unequal_weights_apply_correctly(self):
        # Weighted by sample counts: client0 has 1 sample, client1 has 9.
        # Aggregate should be much closer to client1's value.
        s = FederatedServer()
        agg = s.aggregate(
            client_states=[_state(0.0), _state(10.0)],
            weights=[1, 9],
        )
        assert torch.allclose(agg["layer1.weight"], torch.full((2, 3), 9.0))

    def test_weights_normalized_internally(self):
        # Same as above, but with weights in absolute counts vs ratios — must produce
        # the same answer because the server normalizes by total internally.
        s = FederatedServer()
        agg_counts = s.aggregate([_state(0.0), _state(10.0)], weights=[1, 9])
        agg_ratios = s.aggregate([_state(0.0), _state(10.0)], weights=[0.1, 0.9])
        assert torch.allclose(agg_counts["layer1.weight"], agg_ratios["layer1.weight"])

    def test_dtype_preserved(self):
        s = FederatedServer()
        # Original tensors are float32 — output should be too.
        agg = s.aggregate([_state(1.0), _state(2.0)], weights=[1, 1])
        for k, v in agg.items():
            assert v.dtype == torch.float32, f"{k} dtype changed to {v.dtype}"

    def test_state_count_must_match_weight_count(self):
        s = FederatedServer()
        with pytest.raises(AssertionError):
            s.aggregate([_state(1.0)], weights=[1, 2])

    def test_global_state_is_persisted(self):
        s = FederatedServer()
        s.aggregate([_state(1.0), _state(3.0)], weights=[1, 1])
        cached = s.get_global_state()
        assert torch.allclose(cached["layer1.weight"], torch.full((2, 3), 2.0))

    def test_global_param_norm_before_any_aggregation(self):
        s = FederatedServer()
        assert s.global_param_norm() == 0.0

    def test_global_param_norm_matches_torch(self):
        s = FederatedServer()
        agg = s.aggregate([_state(2.0)], weights=[1])
        # Norm = sqrt(2*3*4 + 2*4) = sqrt(24+8) = sqrt(32)
        expected = torch.cat([t.flatten() for t in agg.values()]).norm().item()
        assert s.global_param_norm() == pytest.approx(expected, rel=1e-5)

    def test_three_client_aggregation(self):
        s = FederatedServer()
        # Mean of [1,2,3] weighted equally = 2.0
        agg = s.aggregate(
            [_state(1.0), _state(2.0), _state(3.0)],
            weights=[10, 10, 10],
        )
        assert torch.allclose(agg["layer1.weight"], torch.full((2, 3), 2.0))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
