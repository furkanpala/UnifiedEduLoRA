"""
Step 6 — Create train/val/test splits for the individual-vs-federated experiment.

Each of the three clients gets an 80/10/10 split (by entry, then flattened to QA
samples). Splits are written in two compatible formats:

  • Federated (train_federated.py):
      client_{id}_train.json / client_{id}_val.json / client_{id}_test.json

  • Individual (train_client.py):
      client_{id}_fold1_train.json = same as client_{id}_train.json
      client_{id}_fold1_val.json   = same as client_{id}_val.json

  • Global test (evaluator.py):
      global_test.json = union of all three clients' test sets

Client assignments:
  0 → MIT lecture notes     → facebook/bart-base
  1 → Stanford CS229 notes  → google/flan-t5-base
  2 → ML papers             → allenai/led-base-16384

Usage (Colab terminal):
    python experiments/06_split_fed_experiment.py \\
        --data-dir    /content/drive/MyDrive/unifiedfl_fed_experiment/data \\
        --output-dir  /content/drive/MyDrive/unifiedfl_fed_experiment/splits \\
        --seed 42 --train-ratio 0.8 --val-ratio 0.1
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR / "unifiedfl"))

CLIENT_DATASETS = [
    (0, "ML_QA_LectureNotes_MIT_enhanced.json"),
    (1, "ML_QA_LectureNotes_StanfordCS229_enhanced.json"),
    (2, "ML_QA_Papers_v2_enhanced.json"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",    required=True,
                   help="Directory containing enhanced JSON files from step 5.")
    p.add_argument("--output-dir",  required=True,
                   help="Directory where split JSONs will be written.")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--train-ratio", type=float, default=0.80)
    p.add_argument("--val-ratio",   type=float, default=0.10)
    return p.parse_args()


def flatten_samples(entries: list, indices: list) -> list:
    """Flatten selected entries to a list of per-QA samples."""
    out = []
    for i in indices:
        entry = entries[i]
        ctx = entry.get("clean_context", "")
        for qa in entry.get("qa_pairs", []):
            out.append({
                "context":        ctx,
                "question":       qa.get("question", ""),
                "answer":         qa.get("answer", ""),
                "question_topic": qa.get("question_topic", "the main topic"),
                "bloom_level":    int(qa.get("bloom_level", 2)),
                "difficulty":     qa.get("difficulty", "medium"),
            })
    return out


def _load_json_or_jsonl(path: Path) -> list:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    decoder = json.JSONDecoder()
    entries, idx = [], 0
    while idx < len(text):
        while idx < len(text) and text[idx] in " \t\n\r":
            idx += 1
        if idx >= len(text):
            break
        obj, idx = decoder.raw_decode(text, idx)
        entries.append(obj)
    return entries


def write_json(path: Path, data: list) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    test_ratio = 1.0 - args.train_ratio - args.val_ratio

    all_test_samples: list = []

    for cid, filename in CLIENT_DATASETS:
        src = data_dir / filename
        if not src.exists():
            sys.exit(f"Enhanced file not found: {src}\nRun step 5 first.")

        entries = _load_json_or_jsonl(src)
        n = len(entries)
        print(f"\nClient {cid} — {filename}: {n} entries")

        idx = list(range(n))
        rng.shuffle(idx)

        n_train = int(round(n * args.train_ratio))
        n_val   = int(round(n * args.val_ratio))
        # test gets the remainder so the sum is exactly n
        train_idx = sorted(idx[:n_train])
        val_idx   = sorted(idx[n_train:n_train + n_val])
        test_idx  = sorted(idx[n_train + n_val:])

        train_s = flatten_samples(entries, train_idx)
        val_s   = flatten_samples(entries, val_idx)
        test_s  = flatten_samples(entries, test_idx)

        print(f"  train={len(train_s)}  val={len(val_s)}  test={len(test_s)} samples")

        # Federated format
        write_json(output_dir / f"client_{cid}_train.json", train_s)
        write_json(output_dir / f"client_{cid}_val.json",   val_s)
        write_json(output_dir / f"client_{cid}_test.json",  test_s)

        # Individual-training fold-1 aliases (same data, different filenames)
        write_json(output_dir / f"client_{cid}_fold1_train.json", train_s)
        write_json(output_dir / f"client_{cid}_fold1_val.json",   val_s)

        all_test_samples.extend(test_s)

    # Global test: union of all three clients' test sets
    rng.shuffle(all_test_samples)
    write_json(output_dir / "global_test.json", all_test_samples)
    print(f"\nglobal_test.json: {len(all_test_samples)} samples total")
    print(f"\nAll splits saved to {output_dir}")

    # Quick field-coverage check
    sample = json.loads((output_dir / "client_0_fold1_train.json").read_text())
    n_topic = sum(1 for s in sample if s["question_topic"] and s["question_topic"] != "ML concept")
    print(f"\nField coverage in client_0 train ({len(sample)} samples):")
    print(f"  question_topic populated (non-fallback): {n_topic}/{len(sample)}")
    print(f"  bloom_level sample (first 8): {[s['bloom_level'] for s in sample[:8]]}")


if __name__ == "__main__":
    main()
