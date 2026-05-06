"""
Tests for unifiedfl/utils/reproducibility.py.

We don't test that two GPU runs produce bit-identical outputs (would
require a GPU and is brittle on CI). We test the cheap invariants:
- env vars get set
- cuDNN flags get flipped
- two consecutive seedings produce identical CPU RNG draws
- train_client / train_federated / main all import the same set_seeds
"""

from __future__ import annotations

import os
import random

import numpy as np
import pytest
import torch


def test_set_seeds_makes_cpu_rng_reproducible():
    from utils.reproducibility import set_seeds

    set_seeds(42)
    py_a = [random.random() for _ in range(5)]
    np_a = np.random.rand(5).tolist()
    th_a = torch.rand(5).tolist()

    set_seeds(42)
    py_b = [random.random() for _ in range(5)]
    np_b = np.random.rand(5).tolist()
    th_b = torch.rand(5).tolist()

    assert py_a == py_b
    assert np_a == np_b
    assert th_a == th_b


def test_set_seeds_different_seeds_diverge():
    from utils.reproducibility import set_seeds

    set_seeds(1)
    a = torch.rand(5).tolist()
    set_seeds(2)
    b = torch.rand(5).tolist()
    assert a != b


def test_set_seeds_sets_pythonhashseed_env():
    from utils.reproducibility import set_seeds

    set_seeds(1234)
    assert os.environ.get("PYTHONHASHSEED") == "1234"


def test_cublas_workspace_set_at_module_load():
    # Importing the module should have already set CUBLAS_WORKSPACE_CONFIG
    # (via os.environ.setdefault) — without it, torch.use_deterministic_algorithms
    # crashes on CUDA 10.2+ the first time a matmul runs.
    import utils.reproducibility  # noqa: F401
    assert "CUBLAS_WORKSPACE_CONFIG" in os.environ
    # Either ":4096:8" (our default) or ":16:8" (some CI machines preset it).
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"].startswith(":")


def test_cudnn_flags_flipped_after_set_seeds():
    from utils.reproducibility import set_seeds

    # Reset to a known non-deterministic state first.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    set_seeds(42)

    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_train_client_reexports_set_seeds():
    # 11_recover_eval_only.py imports set_seeds from train_client; this guards
    # the re-export so a future refactor can't silently break recovery.
    import importlib.util
    from pathlib import Path

    repo = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "tc", repo / "unifiedfl" / "train_client.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    from utils.reproducibility import set_seeds as canonical
    assert m.set_seeds is canonical


def test_train_federated_reexports_set_seeds():
    import importlib.util
    from pathlib import Path

    repo = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "tf", repo / "unifiedfl" / "train_federated.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    from utils.reproducibility import set_seeds as canonical
    assert m.set_seeds is canonical


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
