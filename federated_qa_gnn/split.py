"""
Create and persist train/val/test splits for all clients.

Usage (Colab):
    # 1. Mount Drive first:
    #    from google.colab import drive; drive.mount('/content/drive')
    # 2. Run:
    python split.py \
        --client 0:data/ML_QA_LectureNotes_MIT.json \
        --client 1:data/ML_QA_LectureNotes_StanfordCS229.json \
        --client 2:data/ML_QA_Papers_v2.json \
        --output-dir /content/drive/MyDrive/federated_qa_gnn/outputs \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).parent))

from data.preprocessing import prepare_all_data
from utils.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create and save data splits")
    p.add_argument(
        "--client", action="append", required=True, metavar="ID:DATA_PATH",
        help="Client spec 'id:path/to/data.json'. Repeat for each client.",
    )
    p.add_argument("--output-dir", default="outputs/",
                   help="Directory to save splits (can be a Drive path on Colab)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Parse client specs
    client_cfgs = []
    for spec in args.client:
        parts = spec.split(":", 1)
        if len(parts) != 2:
            raise ValueError(f"--client must be 'id:data_path', got: {spec!r}")
        cid, data_path = int(parts[0]), parts[1]
        client_cfgs.append(SimpleNamespace(client_id=cid, data_path=data_path))
    client_cfgs.sort(key=lambda c: c.client_id)

    output_dir = Path(args.output_dir)
    splits_dir = output_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(str(output_dir))

    print(f"\nProcessing {len(client_cfgs)} client(s) …")
    data_info = prepare_all_data(client_cfgs, str(output_dir), seed=args.seed)
    client_splits = data_info["client_splits"]

    # Save per-client splits
    for cid, splits in sorted(client_splits.items()):
        for split_name, samples in splits.items():
            out_path = splits_dir / f"client_{cid}_{split_name}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(samples, f, indent=2)
            print(f"  Saved {len(samples):>4} samples → {out_path}")

    # Save global test set (union of all clients' test splits)
    global_test: list = []
    for cid in sorted(client_splits):
        global_test.extend(client_splits[cid]["test"])
    global_path = splits_dir / "global_test.json"
    with open(global_path, "w", encoding="utf-8") as f:
        json.dump(global_test, f, indent=2)
    print(f"  Saved {len(global_test):>4} samples → {global_path}")

    # Summary
    print(f"\nDone. n_min={data_info['n_min']}")
    for cid in sorted(client_splits):
        s = client_splits[cid]
        print(f"  Client {cid}: train={len(s['train'])}  val={len(s['val'])}  test={len(s['test'])} QA pairs")
    print(f"\nSplits written to {splits_dir}")


if __name__ == "__main__":
    main()
