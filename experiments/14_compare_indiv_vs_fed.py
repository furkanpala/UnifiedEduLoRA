"""
Compare individual training vs federated training for a single
conditioning, across folds, on three split families:

  C1  global test set        (each client's model on the pooled test set)
  C2  local val set          (each client's model on its own fold val set)
  C3  local test set         (each client's model on its own held-out test)

Prerequisites — these metrics files must exist on disk:

  Individual (per fold k, conditioning c, client X):
    {indiv_dir}/{c}/client_X/foldk/results/{best,final}/metrics_val.json
    {indiv_dir}/{c}/client_X/foldk/results/{best,final}/metrics_test.json
    {indiv_dir}/{c}/client_X/foldk/results/{best,final}/metrics_global_test.json

  Federated (per fold k):
    {fed_dir}_fold{k}/results/{best,final}/client_X_metrics_val.json
    {fed_dir}_fold{k}/results/{best,final}/client_X_metrics_test.json
    {fed_dir}_fold{k}/results/{best,final}/client_X_metrics_global_test.json

Missing files render as '   --   '. Use experiments/11 + 13 to back-fill.

Usage:
    python experiments/14_compare_indiv_vs_fed.py \\
        --indiv-dir /path/to/individual/<conditioning_root> \\
        --fed-dir   /path/to/federated/<base>     # appended with _fold1, _fold2, _fold3
        --conditioning topic
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Common metrics: present in fast-eval output of both pipelines.
METRICS = ("rouge_l", "bleu_4", "bertscore_f1", "blooms_cls_evs_mean")
SHORT = {
    "rouge_l":             "rouge_l",
    "bleu_4":              "bleu_4",
    "bertscore_f1":        "bert_f1",
    "blooms_cls_evs_mean": "bloom_evs",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--indiv-dir", required=True,
                   help="Root containing {conditioning}/client_X/foldY/results/...")
    p.add_argument("--fed-dir", required=True,
                   help="Federated run base path. The script appends _fold{k} "
                        "to look for fold runs (e.g., '/foo/fed' will look at "
                        "/foo/fed_fold1/results/...). Pass the value WITHOUT "
                        "the _fold suffix.")
    p.add_argument("--fold-suffix", default="_fold",
                   help="Connector between fed-dir and the fold number. "
                        "Default '_fold' so 'fed_dir + _fold + 1' = 'fed_fold1'.")
    p.add_argument("--conditioning", default="topic",
                   choices=["baseline", "topic", "bloom"])
    p.add_argument("--clients", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--folds",   nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--ckpts",   nargs="+", choices=["best", "final"],
                   default=["best", "final"])
    p.add_argument("--save", default=None,
                   help="Optional path to write the full per-cell table as JSON.")
    return p.parse_args()


# ── file path helpers ────────────────────────────────────────────────────────

def _indiv_path(indiv_root: Path, cond: str, cid: int, fold: int,
                ckpt: str, split: str) -> Path:
    return (indiv_root / cond / f"client_{cid}" / f"fold{fold}"
            / "results" / ckpt / f"metrics_{split}.json")


def _fed_path(fed_base: Path, suffix: str, fold: int, cid: int,
              ckpt: str, split: str) -> Path:
    fold_dir = Path(str(fed_base) + f"{suffix}{fold}")
    return fold_dir / "results" / ckpt / f"client_{cid}_metrics_{split}.json"


def _read(path: Path) -> Optional[Dict[str, float]]:
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return {k: d[k] for k in METRICS if k in d}
    except (json.JSONDecodeError, OSError):
        return None


# ── aggregation ──────────────────────────────────────────────────────────────

def _avg_across_folds(folds_data: Dict[int, Dict[str, float]]) -> Dict[str, Tuple[Optional[float], Optional[float], int]]:
    """{fold: {metric: v}} -> {metric: (mean, stdev_or_None, n_folds)}"""
    out = {}
    for m in METRICS:
        vals = [folds_data[k][m] for k in folds_data
                if folds_data[k] and m in folds_data[k]]
        out[m] = (
            statistics.mean(vals) if vals else None,
            statistics.stdev(vals) if len(vals) > 1 else None,
            len(vals),
        )
    return out


# ── printing ─────────────────────────────────────────────────────────────────

def _fmt(value: Optional[float], width: int = 8) -> str:
    if value is None:
        return f"{'--':>{width}}"
    return f"{value:{width}.4f}"


def _print_table(title: str, ckpt: str,
                 indiv_per_client: Dict[int, Dict[str, float]],
                 fed_per_client:   Dict[int, Dict[str, float]],
                 clients: List[int]) -> None:
    """One section: per-client + overall avg, both indiv and fed."""
    print(f"\n{'=' * 86}")
    print(f"  {title}  [{ckpt} checkpoint]")
    print(f"{'=' * 86}")

    cols = "  ".join(f"{SHORT[m]:>10}" for m in METRICS)
    print(f"  {'client':<10}{'pipeline':<12} {cols}    n")
    print("  " + "-" * 84)

    for cid in clients:
        for label, per in (("individual", indiv_per_client.get(cid, {})),
                           ("federated",  fed_per_client.get(cid, {}))):
            cells = "  ".join(_fmt(per.get(m, (None,))[0]) for m in METRICS)
            ns    = [per.get(m, (None, None, 0))[2] for m in METRICS]
            n_str = ns[0] if ns and len(set(ns)) == 1 else "?"
            print(f"  client_{cid}  {label:<12} {cells}   {n_str:>3}")
        # delta row (fed - indiv) on means
        delta_cells = []
        for m in METRICS:
            i = indiv_per_client.get(cid, {}).get(m, (None,))[0]
            f = fed_per_client.get(cid, {}).get(m, (None,))[0]
            if i is None or f is None:
                delta_cells.append(f"{'--':>10}")
            else:
                d = f - i
                delta_cells.append(f"{('+' if d >= 0 else ''):>2}{d:7.4f}")
        print(f"  {'':<10}{'delta':<12} {'  '.join(delta_cells)}")
        print()

    # Overall = mean across clients of each pipeline's per-client mean.
    def _overall(per_client):
        out = {}
        for m in METRICS:
            vals = [per_client[c][m][0] for c in clients
                    if c in per_client and per_client[c].get(m, (None,))[0] is not None]
            out[m] = statistics.mean(vals) if vals else None
        return out
    indiv_over = _overall(indiv_per_client)
    fed_over   = _overall(fed_per_client)

    print("  " + "-" * 84)
    cells_i = "  ".join(_fmt(indiv_over[m]) for m in METRICS)
    cells_f = "  ".join(_fmt(fed_over[m])   for m in METRICS)
    print(f"  {'OVERALL':<10}{'individual':<12} {cells_i}")
    print(f"  {'':<10}{'federated':<12} {cells_f}")
    delta_cells = []
    for m in METRICS:
        i, f = indiv_over[m], fed_over[m]
        if i is None or f is None:
            delta_cells.append(f"{'--':>10}")
        else:
            d = f - i
            delta_cells.append(f"{('+' if d >= 0 else ''):>2}{d:7.4f}")
    print(f"  {'':<10}{'delta':<12} {'  '.join(delta_cells)}")


# ── core: build per-(client, split, pipeline, ckpt) tables ──────────────────

def _gather(args, split: str, ckpt: str):
    """Returns (indiv, fed) dicts: {client_id: {metric: (mean, stdev, n)}}."""
    indiv = {cid: {} for cid in args.clients}
    fed   = {cid: {} for cid in args.clients}
    for cid in args.clients:
        ind_folds = {}
        fed_folds = {}
        for k in args.folds:
            ind_path = _indiv_path(Path(args.indiv_dir), args.conditioning, cid, k, ckpt, split)
            fed_path = _fed_path(Path(args.fed_dir), args.fold_suffix, k, cid, ckpt, split)
            ind = _read(ind_path)
            fd  = _read(fed_path)
            if ind: ind_folds[k] = ind
            if fd:  fed_folds[k] = fd
        if ind_folds: indiv[cid] = _avg_across_folds(ind_folds)
        if fed_folds: fed[cid]   = _avg_across_folds(fed_folds)
    return indiv, fed


def main() -> None:
    args = parse_args()

    print(f"\nIndividual root: {args.indiv_dir}/{args.conditioning}/")
    print(f"Federated base:  {args.fed_dir}{args.fold_suffix}<1|2|3>/")
    print(f"Folds: {args.folds}, Clients: {args.clients}, Ckpts: {args.ckpts}")

    sections = [
        ("C1: GLOBAL TEST", "global_test"),
        ("C2: LOCAL VAL",   "val"),
        ("C3: LOCAL TEST",  "test"),
    ]

    save_dump: Dict = {}

    for ckpt in args.ckpts:
        for title, split in sections:
            indiv, fed = _gather(args, split, ckpt)
            _print_table(title, ckpt, indiv, fed, args.clients)
            if args.save:
                save_dump.setdefault(ckpt, {})[split] = {
                    "individual": {str(c): {m: list(indiv[c][m]) if m in indiv[c] else None
                                            for m in METRICS} for c in indiv},
                    "federated":  {str(c): {m: list(fed[c][m])   if m in fed[c]   else None
                                            for m in METRICS} for c in fed},
                }

    if args.save:
        Path(args.save).write_text(
            json.dumps(save_dump, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        print(f"\nFull table saved to {args.save}")


if __name__ == "__main__":
    main()
