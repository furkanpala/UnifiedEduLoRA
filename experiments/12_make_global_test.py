"""
Build global_test.json by concatenating per-client test files.

The split layout produced by experiments/02_split_3fold.py creates one
test file per client (e.g. ``client_0_test.json``) but no global test
pool. The fed-vs-individual comparison wants a single held-out set
that spans all clients, so we concatenate them here.

Usage (Colab):
    python experiments/12_make_global_test.py \\
        --splits-dir /content/drive/MyDrive/unifiedfl_fp_individual_experiment/unbalanced/splits

Each sample in the output gains a ``_source_client`` field (the
originating client_id) so downstream analysis can stratify if needed.
The original sample fields are otherwise unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--splits-dir", required=True,
                   help="Directory containing client_X_test.json files.")
    p.add_argument("--clients", nargs="+", type=int, default=[0, 1, 2],
                   help="Client IDs to pool. Default: 0 1 2.")
    p.add_argument("--output", default=None,
                   help="Output path. Default: <splits-dir>/global_test.json.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite an existing output file.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    splits = Path(args.splits_dir)
    out = Path(args.output) if args.output else splits / "global_test.json"

    if out.exists() and not args.force:
        print(f"[skip] {out} already exists. Pass --force to overwrite.")
        return

    pool: list = []
    for cid in args.clients:
        path = splits / f"client_{cid}_test.json"
        if not path.exists():
            raise FileNotFoundError(path)
        samples = json.loads(path.read_text(encoding="utf-8"))
        for s in samples:
            s["_source_client"] = cid
        pool.extend(samples)
        print(f"  client_{cid}: {len(samples):>4} samples")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pool, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGlobal test set: {len(pool)} samples saved to {out}")


if __name__ == "__main__":
    main()
