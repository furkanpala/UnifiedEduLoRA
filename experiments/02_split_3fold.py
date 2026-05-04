"""
Step 2 — Create 3-fold cross-validation splits at the entry level.

Reads the enhanced JSON (post step 1), holds out a fixed test set, splits the
rest into N folds, then flattens each fold into per-QA samples preserving
`question_topic`, `bloom_level`, `difficulty`.

Usage (Colab terminal):
    python experiments/02_split_3fold.py \\
        --input  /content/drive/MyDrive/unifiedfl_mit_experiment/data/ML_QA_LectureNotes_MIT_enhanced.json \\
        --output-dir /content/drive/MyDrive/unifiedfl_mit_experiment/splits \\
        --client-id 0 --seed 42 --test-ratio 0.15 --n-folds 3
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_DIR = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, str(Path(REPO_DIR) / "unifiedfl"))

from data.preprocessing import load_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input",      required=True, help="Enhanced JSON from step 1")
    p.add_argument("--output-dir", required=True, help="Where to write split JSONs")
    p.add_argument("--client-id",  type=int, default=0)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--n-folds",    type=int, default=3)
    return p.parse_args()


def flatten_with_metadata(entries, indices):
    out = []
    for i in indices:
        entry = entries[i]
        ctx = entry.get("clean_context", "")
        for qa in entry.get("qa_pairs", []):
            out.append({
                "context":         ctx,
                "question":        qa.get("question", ""),
                "answer":          qa.get("answer", ""),
                "question_topic":  qa.get("question_topic", "the main topic"),
                "bloom_level":     int(qa.get("bloom_level", 2)),
                "difficulty":      qa.get("difficulty", "medium"),
            })
    return out


def main() -> None:
    args = parse_args()

    entries = load_json(Path(args.input))
    print(f"Loaded {len(entries)} entries from {args.input}")

    rng = random.Random(args.seed)
    all_idx = list(range(len(entries)))
    rng.shuffle(all_idx)

    n_test    = int(round(len(entries) * args.test_ratio))
    test_idx  = sorted(all_idx[:n_test])
    remaining = all_idx[n_test:]
    fold_size = len(remaining) // args.n_folds

    fold_indices = []
    for k in range(args.n_folds):
        start = k * fold_size
        end   = (k + 1) * fold_size if k < args.n_folds - 1 else len(remaining)
        fold_indices.append(sorted(remaining[start:end]))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cid = args.client_id

    test_samples = flatten_with_metadata(entries, test_idx)
    (out_dir / f"client_{cid}_test.json").write_text(
        json.dumps(test_samples, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"  test               {len(test_samples):>5} samples")

    for k in range(args.n_folds):
        val_k   = fold_indices[k]
        train_k = sorted(set(remaining) - set(val_k))
        train_samples = flatten_with_metadata(entries, train_k)
        val_samples   = flatten_with_metadata(entries, val_k)
        (out_dir / f"client_{cid}_fold{k+1}_train.json").write_text(
            json.dumps(train_samples, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        (out_dir / f"client_{cid}_fold{k+1}_val.json").write_text(
            json.dumps(val_samples, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        print(f"  fold{k+1} train       {len(train_samples):>5}   val {len(val_samples):>5}")

    # Quick coverage check on fold1 train
    sample_path = out_dir / f"client_{cid}_fold1_train.json"
    sample = json.loads(sample_path.read_text())
    n_topic = sum(1 for s in sample if s["question_topic"])
    print(f"\nField coverage in fold1_train ({len(sample)} samples):")
    print(f"  question_topic populated: {n_topic}/{len(sample)}")
    print(f"  bloom_level (first 8):    {[s['bloom_level'] for s in sample[:8]]}")
    print(f"\nSplits saved to {out_dir}")


if __name__ == "__main__":
    main()
