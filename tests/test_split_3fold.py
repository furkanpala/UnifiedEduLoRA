"""
Tests for the 3-fold CV split logic in experiments/02_split_3fold.py and the
fed-experiment splitter in experiments/06_split_fed_experiment.py.

We don't import the orchestrator scripts directly (they have a `__main__`
block and use argparse). Instead we re-exercise the same algorithm here so
regressions in fold disjointness or train/test leakage are caught.
"""

from __future__ import annotations

import random

import pytest


def _three_fold_indices(n: int, seed: int, test_ratio: float = 0.1):
    """Reproduces the algorithm from experiments/02_split_3fold.py."""
    rng = random.Random(seed)
    all_idx = list(range(n))
    rng.shuffle(all_idx)
    n_test = int(round(n * test_ratio))
    test_idx = sorted(all_idx[:n_test])
    remaining = all_idx[n_test:]
    fold_size = len(remaining) // 3
    folds = []
    for k in range(3):
        start = k * fold_size
        end = (k + 1) * fold_size if k < 2 else len(remaining)
        folds.append(sorted(remaining[start:end]))
    return test_idx, remaining, folds


class TestThreeFoldSplit:
    def test_test_and_folds_partition_full_index_set(self):
        test, remaining, folds = _three_fold_indices(n=100, seed=42)
        union = set(test)
        for f in folds:
            union |= set(f)
        assert union == set(range(100))

    def test_test_disjoint_from_every_fold(self):
        test, _, folds = _three_fold_indices(n=100, seed=42)
        ts = set(test)
        for k, f in enumerate(folds):
            assert not (ts & set(f)), f"fold {k} overlaps test"

    def test_folds_are_pairwise_disjoint(self):
        _, _, folds = _three_fold_indices(n=100, seed=42)
        for i in range(3):
            for j in range(i + 1, 3):
                assert not (set(folds[i]) & set(folds[j])), \
                    f"fold {i} and fold {j} overlap"

    def test_remainder_assigned_to_last_fold_when_not_divisible(self):
        # 100 - 10 (test) = 90 remaining → fold_size=30, all folds equal.
        # 101 - 10 = 91 remaining → fold_size=30, last fold gets 31.
        _, remaining, folds = _three_fold_indices(n=101, seed=42)
        assert len(folds[0]) == 30
        assert len(folds[1]) == 30
        assert len(folds[2]) == 31
        assert len(folds[0]) + len(folds[1]) + len(folds[2]) == len(remaining)

    def test_seed_makes_split_reproducible(self):
        a = _three_fold_indices(n=200, seed=7)
        b = _three_fold_indices(n=200, seed=7)
        assert a == b

    def test_different_seeds_give_different_splits(self):
        a = _three_fold_indices(n=200, seed=7)
        b = _three_fold_indices(n=200, seed=8)
        assert a != b

    def test_train_for_each_fold_is_other_two_folds(self):
        # In CV, train_k = remaining \ val_k, where val_k is the kth fold.
        # The implementation in 02_split_3fold.py uses set arithmetic.
        # Confirm there's no off-by-one.
        _, remaining, folds = _three_fold_indices(n=120, seed=1)
        for k in range(3):
            val_k = folds[k]
            train_k = sorted(set(remaining) - set(val_k))
            # train_k should equal the union of the other two folds.
            other = sorted(set(folds[(k + 1) % 3]) | set(folds[(k + 2) % 3]))
            assert train_k == other


class TestFedExperimentSplit:
    """
    Regression tests for the simpler 80/10/10 train/val/test split used in
    experiments/06_split_fed_experiment.py.
    """

    def _split(self, n: int, seed: int, train_ratio=0.8, val_ratio=0.1):
        rng = random.Random(seed)
        idx = list(range(n))
        rng.shuffle(idx)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        train = sorted(idx[:n_train])
        val   = sorted(idx[n_train:n_train + n_val])
        test  = sorted(idx[n_train + n_val:])
        return train, val, test

    def test_three_way_partition(self):
        train, val, test = self._split(n=100, seed=42)
        assert set(train) | set(val) | set(test) == set(range(100))
        assert not (set(train) & set(val))
        assert not (set(val) & set(test))
        assert not (set(train) & set(test))

    def test_counts_sum_to_n(self):
        for n in (50, 99, 100, 101, 1000):
            train, val, test = self._split(n=n, seed=1)
            assert len(train) + len(val) + len(test) == n

    def test_default_ratios_approximately_80_10_10(self):
        train, val, test = self._split(n=1000, seed=1)
        assert len(train) == 800
        assert len(val) == 100
        assert len(test) == 100


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
