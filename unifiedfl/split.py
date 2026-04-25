"""
Create and persist train/val/test splits with 3-fold cross-validation.

PROTOCOL - every collaborator runs this with the same seed (default 42).

Split structure
───────────────
  Step 1  Fix a test set (15% of entries) - identical across all folds.
          never used for training or validation

  Step 2  Divide the remaining 85% into 3 equal folds.

  Step 3  For each fold k ∈ {1, 2, 3}:
            train = the other two folds  (flattened QA pairs)
            val   = fold k               (flattened QA pairs)

All splits are at the entry level (not QA-pair level) to prevent context leakage.

Output (inside --output-dir/splits/)
─────────────────────────────────────
  client_<N>_test.json            ← fixed, same for all folds
  client_<N>_fold1_train.json
  client_<N>_fold1_val.json
  client_<N>_fold2_train.json
  client_<N>_fold2_val.json
  client_<N>_fold3_train.json
  client_<N>_fold3_val.json
  checksums.txt                   ← MD5 of every file; send this to Furkan Pala

Usage
─────
  python split.py \\
      --client 0:data/client0_data.json \\
      --seed   42                        # DO NOT CHANGE
      --output-dir outputs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Core splitting logic
# ─────────────────────────────────────────────────────────────────────────────

def _load(path: Path) -> List[Dict[str, Any]]:
    """Load a JSON file that may be a proper array or streaming objects."""
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    if raw.startswith("["):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    import json as _json
    decoder = _json.JSONDecoder()
    records, idx = [], 0
    while idx < len(raw):
        while idx < len(raw) and raw[idx] in " \t\n\r":
            idx += 1
        if idx >= len(raw):
            break
        obj, idx = decoder.raw_decode(raw, idx)
        records.append(obj)
    return records


def _flatten(data: List[Dict[str, Any]], indices: List[int]) -> List[Dict[str, str]]:
    """Return one sample per (context, qa_pair) from the selected entries."""
    samples: List[Dict[str, str]] = []
    for idx in indices:
        entry = data[idx]
        ctx = entry.get("clean_context", "")
        for qa in entry.get("qa_pairs", []):
            samples.append({
                "context":        ctx,
                "question":       qa.get("question", ""),
                "answer":         qa.get("answer", ""),
                "question_topic": qa.get("question_topic", ""),
                "bloom_level":    qa.get("bloom_level", ""),
                "difficulty":     qa.get("difficulty", ""),
            })
    return samples


def make_splits(
    data: List[Dict[str, Any]],
    seed: int = 42,
    n_folds: int = 3,
    test_ratio: float = 0.15,
) -> Dict[str, Any]:
    """
    1. Hold out a fixed test set (test_ratio of entries).
    2. Divide the rest into n_folds equal folds.
    3. For each fold k: train = other folds, val = fold k.

    Returns a dict with keys:
        test_indices          : List[int]
        folds                 : List[List[int]]   (n_folds lists of entry indices)
        fold_splits           : {fold_k: {train: [...], val: [...]}}  (QA-pair samples)
        test_samples          : List[Dict]  (QA-pair samples from test set)
    """
    rng = random.Random(seed)
    n = len(data)
    indices = list(range(n))
    rng.shuffle(indices)

    # Step 1 — fixed test set
    n_test = max(1, int(test_ratio * n))
    test_indices = sorted(indices[:n_test])
    dev_indices  = indices[n_test:]               # remaining for CV

    # Step 2 — divide dev into n_folds equal parts
    rng.shuffle(dev_indices)                      # shuffle again for fold assignment
    fold_size = len(dev_indices) // n_folds
    folds: List[List[int]] = []
    for k in range(n_folds):
        start = k * fold_size
        end   = start + fold_size if k < n_folds - 1 else len(dev_indices)
        folds.append(sorted(dev_indices[start:end]))

    # Step 3 — build per-fold train/val samples
    test_samples = _flatten(data, test_indices)

    fold_splits: Dict[int, Dict[str, List]] = {}
    for k in range(n_folds):
        val_idx   = folds[k]
        train_idx = []
        for j, f in enumerate(folds):
            if j != k:
                train_idx.extend(f)
        fold_splits[k + 1] = {
            "train": _flatten(data, train_idx),
            "val":   _flatten(data, val_idx),
        }

    return {
        "test_indices": test_indices,
        "folds":        folds,
        "fold_splits":  fold_splits,
        "test_samples": test_samples,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Checksum
# ─────────────────────────────────────────────────────────────────────────────

def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate anchored 3-fold CV splits (seed=42 is mandatory)"
    )
    p.add_argument(
        "--client", action="append", required=True, metavar="ID:DATA_PATH",
        help="Client spec 'id:path/to/data.json'. Repeat for each client.",
    )
    p.add_argument("--output-dir", default="outputs/")
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed — must be 42 for all collaborators (default: 42)",
    )
    p.add_argument(
        "--n-folds", type=int, default=3,
        help="Number of CV folds (default: 3)",
    )
    p.add_argument(
        "--test-ratio", type=float, default=0.15,
        help="Fraction of entries held out as fixed test set (default: 0.15)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.seed != 42:
        print(f"WARNING: seed is {args.seed}, not 42. "
              "All collaborators must use seed=42 for comparable splits.")

    # Parse client specs
    client_cfgs = []
    for spec in args.client:
        parts = spec.split(":", 1)
        if len(parts) != 2:
            raise ValueError(f"--client must be 'id:data_path', got: {spec!r}")
        cid, data_path = int(parts[0]), parts[1]
        client_cfgs.append((cid, Path(data_path)))
    client_cfgs.sort(key=lambda x: x[0])

    splits_dir = Path(args.output_dir) / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    checksum_lines: List[str] = []

    print(f"\nSeed: {args.seed} | Folds: {args.n_folds} | Test ratio: {args.test_ratio}")
    print("=" * 60)

    for cid, data_path in client_cfgs:
        print(f"\nClient {cid} — {data_path}")
        data = _load(data_path)
        print(f"  Loaded {len(data)} entries")

        result = make_splits(
            data,
            seed=args.seed,
            n_folds=args.n_folds,
            test_ratio=args.test_ratio,
        )

        # Save fixed test set
        test_path = splits_dir / f"client_{cid}_test.json"
        test_path.write_text(
            json.dumps(result["test_samples"], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        checksum_lines.append(f"{_md5(test_path)}  {test_path.name}")
        print(f"  Test  : {len(result['test_samples']):>5} QA pairs → {test_path.name}")

        # Save per-fold train/val
        for fold_k, splits in result["fold_splits"].items():
            for split_name, samples in splits.items():
                out_path = splits_dir / f"client_{cid}_fold{fold_k}_{split_name}.json"
                out_path.write_text(
                    json.dumps(samples, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                checksum_lines.append(f"{_md5(out_path)}  {out_path.name}")
                print(f"  Fold {fold_k} {split_name:<5}: {len(samples):>5} QA pairs → {out_path.name}")

    # Write checksum file
    checksum_path = splits_dir / "checksums.txt"
    checksum_path.write_text(
        f"# seed={args.seed}  n_folds={args.n_folds}  test_ratio={args.test_ratio}\n"
        + "\n".join(checksum_lines) + "\n",
        encoding="utf-8",
    )

    print(f"\n{'=' * 60}")
    print(f"All splits saved to: {splits_dir}")
    print(f"Checksums:           {checksum_path}")
    print(f"\n>>> Send checksums.txt to Furkan Pala for verification. <<<\n")


if __name__ == "__main__":
    main()
