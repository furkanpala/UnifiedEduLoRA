"""
Reproducibility settings for training and evaluation.

Usage:
    from utils.reproducibility import set_seeds
    set_seeds(args.seed)

Notes:
- ``CUBLAS_WORKSPACE_CONFIG`` must be set BEFORE CUDA is initialized so
  ``torch.use_deterministic_algorithms`` can run on CUDA 10.2+. We set it
  via ``os.environ.setdefault`` at module load, so importing this module
  early in main() partially configures the process before set_seeds()
  is called.
- ``cudnn.benchmark = False`` disables the cuDNN auto-tuner. This costs
  ~5-10% GPU throughput in our benchmarks but is required for
  reproducible runs across machines with the same hardware.
- ``torch.use_deterministic_algorithms(warn_only=True)`` flags
  non-deterministic ops without crashing — needed because PEFT /
  HuggingFace generation use a few CUDA ops (some scatter/gather
  kernels) that don't have deterministic implementations. With
  ``warn_only=False`` the run would die at the first such op.
- ``PYTHONHASHSEED`` only affects subprocess invocations after this
  point (Python's hash randomization is set at interpreter start) but
  we set it for the benefit of any subprocess.call() the orchestrators
  spawn.
"""

from __future__ import annotations

import os
import random

# Set CUBLAS workspace BEFORE torch imports CUDA. Required for
# torch.use_deterministic_algorithms() on CUDA 10.2+.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch


def set_seeds(seed: int = 42) -> None:
    """Seed all RNG sources and configure deterministic CUDA / cuDNN behavior.

    Call once near the top of main() — after argparse, before model creation
    or any tensor allocation. Importing this module already pre-sets
    ``CUBLAS_WORKSPACE_CONFIG``, so even if the caller forgets to invoke
    ``set_seeds()`` the most-load-bearing CUDA env var is in place.
    """
    # Process-level
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch CPU + CUDA
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN — pin algorithm choice and disable the auto-tuner.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Force deterministic algorithm choices everywhere PyTorch supports it.
    # warn_only=True so ops that lack a deterministic implementation
    # (e.g. some scatter/gather kernels on CUDA) don't crash the run —
    # they emit a UserWarning instead.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (TypeError, AttributeError):
        # warn_only kwarg was added in torch 1.11. On older builds the
        # strict version would crash on PEFT's non-deterministic ops, so
        # fall back to the looser cudnn-only setup.
        pass
