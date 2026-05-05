"""
Step 4 — Read every fold's metrics_val.json and print a side-by-side comparison.

Usage (Colab terminal):
    python experiments/04_aggregate_results.py \\
        --output-dir /content/drive/MyDrive/unifiedfl_mit_experiment/outputs \\
        --client-id 0
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


METRIC_KEYS = [
    "rouge_l", "bleu_4", "bertscore_f1",
    "rtc", "faithfulness", "qafacteval", "rquge", "answer_relevancy",
    "blooms_cls_evs_mean", "blooms_llm_evs_mean",
    "llm_judge_overall_mean",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir",   required=True,
                   help="Root output dir from step 3 (contains baseline/, topic/, bloom/)")
    p.add_argument("--client-id",    type=int, default=0)
    p.add_argument("--conditionings", nargs="+",
                   default=["baseline", "topic", "bloom"])
    p.add_argument("--folds",         nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--checkpoint",   choices=["best", "final"], default="best",
                   help="Which evaluated checkpoint to aggregate (default: best).")
    p.add_argument("--save-summary", default=None,
                   help="Optional path to write per-fold metrics as JSON")
    return p.parse_args()


def load_metrics(out_dir: Path, conditioning: str, client_id: int, fold: int,
                 checkpoint: str) -> dict | None:
    fold_dir = out_dir / conditioning / f"client_{client_id}" / f"fold{fold}"
    # New layout: results/{best,final}/metrics_val.json
    new_path = fold_dir / "results" / checkpoint / "metrics_val.json"
    if new_path.exists():
        return json.loads(new_path.read_text())
    # Backward compat: old layout had metrics_val.json at the fold root
    # (corresponds to the "best" checkpoint metrics).
    if checkpoint == "best":
        old_path = fold_dir / "metrics_val.json"
        if old_path.exists():
            return json.loads(old_path.read_text())
    return None


def fmt(v) -> str:
    if v is None:
        return "   --  "
    return f"{v:7.4f}"


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)

    print(f"Aggregating {args.checkpoint!r} checkpoint metrics for client {args.client_id}.\n")

    rows: list[dict] = []
    for cond in args.conditionings:
        for fold in args.folds:
            m = load_metrics(out_dir, cond, args.client_id, fold, args.checkpoint)
            if m is None:
                continue
            rows.append({
                "conditioning": cond,
                "fold": fold,
                **{k: m.get(k) for k in METRIC_KEYS},
            })

    if not rows:
        print("No metrics_val.json files found under "
              f"{out_dir} for the requested conditionings/folds.")
        return

    by_cond: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cond[r["conditioning"]].append(r)

    header = (f"{'conditioning':<10} {'fold':>4}   "
              + "  ".join(f"{k:>8}" for k in METRIC_KEYS))
    print(header)
    print("-" * len(header))
    for cond in args.conditionings:
        rs = by_cond.get(cond, [])
        for r in rs:
            print(f"{cond:<10} {r['fold']:>4}   "
                  + "  ".join(fmt(r[k]) for k in METRIC_KEYS))
        if rs:
            means = {
                k: (sum(r[k] for r in rs if r[k] is not None) /
                    max(sum(1 for r in rs if r[k] is not None), 1))
                for k in METRIC_KEYS
            }
            print(f"{cond:<10} {'mean':>4}   "
                  + "  ".join(fmt(means[k]) for k in METRIC_KEYS))
        print()

    if args.save_summary:
        Path(args.save_summary).write_text(json.dumps(rows, indent=2))
        print(f"Per-fold metrics saved to {args.save_summary}")


if __name__ == "__main__":
    main()
