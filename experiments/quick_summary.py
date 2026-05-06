"""
Quick analysis helper for the 3-conditioning x 3-fold tree produced by
experiments/03_run_three_conditionings.py.

For one client, walks the directory layout

    {output_dir}/{conditioning}/client_{cid}/fold{N}/results/{best,final}/metrics_val.json

reads every (conditioning, fold, checkpoint) cell that exists, and prints:

  1. Per-fold values plus per-conditioning mean and stdev (best checkpoint).
  2. Per-fold values plus per-conditioning mean and stdev (final checkpoint).
  3. A side-by-side mean table with the best conditioning per metric marked.
  4. A best-minus-final delta table.

Faster and more flexible than re-running 04_aggregate_results.py twice and
diffing the two output JSONs by hand. No mutation of the output tree.

Usage
-----
    python experiments/quick_summary.py \\
        --output-dir drive_outputs/unbalanced \\
        --client-id 0
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

METRIC_KEYS: List[str] = [
    "rouge_l", "bleu_4", "bertscore_f1",
    "rtc", "faithfulness", "qafacteval", "rquge", "answer_relevancy",
    "blooms_cls_evs_mean", "blooms_llm_evs_mean",
    "llm_judge_overall_mean",
]
SHORT_NAMES: Dict[str, str] = {
    "rouge_l":             "rouge_l",
    "bleu_4":              "bleu_4",
    "bertscore_f1":        "bert_f1",
    "rtc":                 "rtc",
    "faithfulness":        "faith",
    "qafacteval":          "qafe",
    "rquge":               "rquge",
    "answer_relevancy":    "ans_rel",
    "blooms_cls_evs_mean": "bloom_b",
    "blooms_llm_evs_mean": "bloom_l",
    "llm_judge_overall_mean": "llm_j",
}
# RQUGE is in [1, 5]; everything else in [0, 1]. Higher is better for all.


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir",   required=True,
                   help="Parent of baseline/, topic/, bloom/ subdirectories.")
    p.add_argument("--client-id",    type=int, default=0)
    p.add_argument("--conditionings", nargs="+",
                   default=["baseline", "topic", "bloom"])
    p.add_argument("--folds",         nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--save", default=None,
                   help="Optional path to write the full per-fold table as JSON.")
    return p.parse_args()


def _load_one(path: Path) -> Optional[Dict[str, float]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_cell(out_dir: Path, cond: str, cid: int, fold: int,
               ckpt: str) -> Optional[Dict]:
    """
    Return metrics_val.json dict if it exists. Tries the new layout first,
    falls back to the legacy flat layout (old runs only have a single
    metrics_val.json at the fold root, treated as 'best').
    """
    fold_dir = out_dir / cond / f"client_{cid}" / f"fold{fold}"
    new_path = fold_dir / "results" / ckpt / "metrics_val.json"
    if new_path.exists():
        return json.loads(new_path.read_text(encoding="utf-8"))
    if ckpt == "best":
        legacy = fold_dir / "metrics_val.json"
        if legacy.exists():
            return json.loads(legacy.read_text(encoding="utf-8"))
    return None


def _fmt(v) -> str:
    if v is None:
        return "    --"
    return f"{v:8.4f}"


def _print_table(title: str, conds: List[str], folds: List[int],
                 cells: Dict[Tuple[str, int], Dict[str, float]]) -> None:
    print(f"\n=== {title} ===")
    cols = " ".join(f"{SHORT_NAMES[k]:>8}" for k in METRIC_KEYS)
    print(f"  {'cond':<10} {'fold':>5}   {cols}")
    print("  " + "-" * (10 + 1 + 5 + 3 + 9 * len(METRIC_KEYS)))
    for cond in conds:
        per_metric: Dict[str, List[float]] = {k: [] for k in METRIC_KEYS}
        for fold in folds:
            m = cells.get((cond, fold))
            if m is None:
                continue
            row = " ".join(_fmt(m.get(k)) for k in METRIC_KEYS)
            print(f"  {cond:<10} {fold:>5}   {row}")
            for k in METRIC_KEYS:
                v = m.get(k)
                if isinstance(v, (int, float)):
                    per_metric[k].append(v)
        means = {k: (statistics.mean(v) if v else None)
                 for k, v in per_metric.items()}
        stdevs = {k: (statistics.stdev(v) if len(v) > 1 else None)
                  for k, v in per_metric.items()}
        m_row = " ".join(_fmt(means[k]) for k in METRIC_KEYS)
        s_row = " ".join(_fmt(stdevs[k]) for k in METRIC_KEYS)
        print(f"  {cond:<10} {'mean':>5}   {m_row}")
        print(f"  {cond:<10} {'std':>5}   {s_row}")
        print()


def _print_compare(label: str, conds: List[str],
                   means: Dict[Tuple[str, str], Dict[str, float]],
                   ckpt: str) -> None:
    """means[(cond, ckpt)][metric] -> value"""
    print(f"\n=== Per-conditioning means [{ckpt} checkpoint] ===")
    cols = " ".join(f"{SHORT_NAMES[k]:>8}" for k in METRIC_KEYS)
    print(f"  {'cond':<10}   {cols}")
    print("  " + "-" * (10 + 3 + 9 * len(METRIC_KEYS)))
    # Best per metric for highlighting (asterisk)
    best_per_k: Dict[str, str] = {}
    for k in METRIC_KEYS:
        best_cond, best_v = None, float("-inf")
        for cond in conds:
            v = means.get((cond, ckpt), {}).get(k)
            if isinstance(v, (int, float)) and v > best_v:
                best_v, best_cond = v, cond
        if best_cond is not None:
            best_per_k[k] = best_cond
    for cond in conds:
        m = means.get((cond, ckpt), {})
        cells = []
        for k in METRIC_KEYS:
            v = m.get(k)
            if v is None:
                cells.append("    --")
            elif best_per_k.get(k) == cond:
                cells.append(f"{v:7.4f}*")
            else:
                cells.append(f"{v:8.4f}")
        print(f"  {cond:<10}   {' '.join(cells)}")
    print("  (* = best in column)")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)

    # Read every cell up-front so the rest of the script just looks things up.
    raw: Dict[Tuple[str, str, int], Dict[str, float]] = {}
    n_found = {ckpt: 0 for ckpt in ("best", "final")}
    for cond in args.conditionings:
        for fold in args.folds:
            for ckpt in ("best", "final"):
                m = _read_cell(out_dir, cond, args.client_id, fold, ckpt)
                if m is not None:
                    raw[(cond, ckpt, fold)] = m
                    n_found[ckpt] += 1

    if not raw:
        print(f"No metrics_val.json files found under {out_dir} for "
              f"client {args.client_id}.")
        return

    print(f"Reading {out_dir}, client_{args.client_id}: "
          f"best={n_found['best']} cells, final={n_found['final']} cells")

    # 1+2: per-fold tables for each checkpoint
    for ckpt in ("best", "final"):
        cells = {(c, f): raw[(c, ckpt, f)]
                 for (c, ckpt2, f) in raw if ckpt2 == ckpt}
        if cells:
            _print_table(
                f"Per-fold metrics [{ckpt} checkpoint]",
                args.conditionings, args.folds, cells,
            )

    # 3: per-checkpoint mean compare
    means: Dict[Tuple[str, str], Dict[str, float]] = {}
    for cond in args.conditionings:
        for ckpt in ("best", "final"):
            per = {k: [] for k in METRIC_KEYS}
            for fold in args.folds:
                m = raw.get((cond, ckpt, fold))
                if m is None:
                    continue
                for k in METRIC_KEYS:
                    v = m.get(k)
                    if isinstance(v, (int, float)):
                        per[k].append(v)
            if any(per.values()):
                means[(cond, ckpt)] = {
                    k: (statistics.mean(v) if v else None)
                    for k, v in per.items()
                }
    for ckpt in ("best", "final"):
        if any((c, ckpt) in means for c in args.conditionings):
            _print_compare("means", args.conditionings, means, ckpt)

    # 4: best-minus-final delta (overfitting check)
    if all((c, "best") in means and (c, "final") in means
           for c in args.conditionings):
        print(f"\n=== Best minus Final (positive = best > final) ===")
        cols = " ".join(f"{SHORT_NAMES[k]:>8}" for k in METRIC_KEYS)
        print(f"  {'cond':<10}   {cols}")
        print("  " + "-" * (10 + 3 + 9 * len(METRIC_KEYS)))
        for cond in args.conditionings:
            b = means[(cond, "best")]
            f = means[(cond, "final")]
            cells = []
            for k in METRIC_KEYS:
                bv, fv = b.get(k), f.get(k)
                if bv is None or fv is None:
                    cells.append("    --")
                else:
                    delta = bv - fv
                    sign = "+" if delta >= 0 else ""
                    cells.append(f"{sign}{delta:7.4f}")
            print(f"  {cond:<10}   {' '.join(cells)}")
        print("  (positive => best ckpt outperforms final ckpt = overfitting)")

    # Optional save
    if args.save:
        flat = [
            {"conditioning": c, "checkpoint": k, "fold": f, **m}
            for (c, k, f), m in raw.items()
        ]
        Path(args.save).write_text(json.dumps(flat, indent=2))
        print(f"\nSaved {len(flat)} cells to {args.save}")


if __name__ == "__main__":
    main()
