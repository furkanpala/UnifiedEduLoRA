"""
Step 9 — Compare individual vs. federated training results.

Reads:
  Individual (per client):
    {indiv_dir}/topic/client_{id}/fold1/metrics_val.json         → local val metrics
    {indiv_dir}/topic/client_{id}/fold1/metrics_global_test.json → global test metrics

  Federated (final round):
    {fed_dir}/final_metrics_per_client.json  → per-client {local, global} metrics

Prints a side-by-side table:

  client | split  | metric    | individual | federated | Δ (fed−indiv)
  -------+--------+-----------+------------+-----------+--------------
  ...

Usage (Colab terminal):
    python experiments/09_compare_results.py \\
        --indiv-dir /content/drive/MyDrive/unifiedfl_fed_experiment/individual \\
        --fed-dir   /content/drive/MyDrive/unifiedfl_fed_experiment/federated
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CLIENT_NAMES = {0: "BART (MIT)", 1: "T5 (Stanford)", 2: "LED (Papers)"}
METRICS = ["rouge_l", "bleu_4", "bertscore_f1"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--indiv-dir", required=True,
                   help="Root output dir from step 7 (individual training).")
    p.add_argument("--fed-dir",   required=True,
                   help="Root output dir from step 8 (federated training).")
    p.add_argument("--indiv-checkpoint", choices=["best", "final"], default="best",
                   help="Which individual-baseline checkpoint to load (default: best).")
    p.add_argument("--save",      default=None,
                   help="Optional path to save the comparison as JSON.")
    return p.parse_args()


def load_indiv(indiv_dir: Path, cid: int, checkpoint: str = "best") -> dict[str, dict]:
    """
    Load individual baseline metrics from either the new layout
    (results/{best,final}/metrics_*.json) or the legacy flat layout.
    """
    base = indiv_dir / "topic" / f"client_{cid}" / "fold1"
    result = {}

    def _load_first(*candidates: Path):
        for p in candidates:
            if p.exists():
                return json.loads(p.read_text())
        print(f"  [warn] missing: tried {[str(p) for p in candidates]}")
        return None

    result["local"] = _load_first(
        base / "results" / checkpoint / "metrics_val.json",
        base / "metrics_val.json",  # legacy
    )
    result["global"] = _load_first(
        base / "results" / checkpoint / "metrics_global_test.json",
        base / "metrics_global_test.json",  # legacy
    )
    return result


def load_fed(fed_dir: Path) -> dict[str, dict]:
    path = fed_dir / "final_metrics_per_client.json"
    if not path.exists():
        # Try to parse the last eval_metrics entry from training_log.json
        log_path = fed_dir / "training_log.json"
        if log_path.exists():
            try:
                log = json.loads(log_path.read_text())
                rounds = log.get("rounds", [])
                for r in reversed(rounds):
                    if r.get("eval_metrics"):
                        return {str(k): v for k, v in r["eval_metrics"].items()}
            except Exception:
                pass
        print(f"  [warn] federated metrics not found: {path}")
        return {}
    raw = json.loads(path.read_text())
    # New format: { best_round, best_avg_val_loss, per_client_metrics: {...} }
    # Old format: flat { "0": {...}, "1": {...}, ... }
    if "per_client_metrics" in raw:
        if "best_round" in raw:
            print(f"  [info] federated best snapshot was round "
                  f"{raw.get('best_round')} "
                  f"(avg val loss = {raw.get('best_avg_val_loss'):.4f})")
        return raw["per_client_metrics"]
    return raw


def fmt(v) -> str:
    if v is None:
        return "  —   "
    return f"{v:.4f}"


def delta(fed_v, indiv_v) -> str:
    if fed_v is None or indiv_v is None:
        return "  —  "
    d = fed_v - indiv_v
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.4f}"


def main() -> None:
    args = parse_args()
    indiv_dir = Path(args.indiv_dir)
    fed_dir   = Path(args.fed_dir)

    fed_data = load_fed(fed_dir)

    comparison = {}
    rows = []

    for cid in [0, 1, 2]:
        name = CLIENT_NAMES[cid]
        indiv = load_indiv(indiv_dir, cid, checkpoint=args.indiv_checkpoint)
        fed   = fed_data.get(str(cid), {})

        comparison[cid] = {"name": name, "individual": indiv, "federated": fed}

        for split in ["local", "global"]:
            indiv_split = indiv.get(split) or {}
            fed_split   = fed.get(split) or {}

            for m in METRICS:
                rows.append({
                    "client":     f"C{cid} {name}",
                    "split":      split,
                    "metric":     m,
                    "individual": indiv_split.get(m),
                    "federated":  fed_split.get(m),
                })

    # ── Print table ──────────────────────────────────────────────────────────
    col_w = {"client": 18, "split": 7, "metric": 14, "individual": 11, "federated": 10, "delta": 9}
    header = (
        f"{'client':<{col_w['client']}} {'split':<{col_w['split']}} "
        f"{'metric':<{col_w['metric']}} {'individual':>{col_w['individual']}} "
        f"{'federated':>{col_w['federated']}} {'Δ(fed−indiv)':>{col_w['delta']}}"
    )
    sep = "─" * len(header)

    print(f"\n{sep}")
    print("  Individual  vs.  Federated  (topic conditioning)")
    print(sep)
    print(header)
    print(sep)

    prev_client = None
    for r in rows:
        if r["client"] != prev_client:
            if prev_client is not None:
                print()
            prev_client = r["client"]

        d = delta(r["federated"], r["individual"])
        print(
            f"{r['client']:<{col_w['client']}} {r['split']:<{col_w['split']}} "
            f"{r['metric']:<{col_w['metric']}} {fmt(r['individual']):>{col_w['individual']}} "
            f"{fmt(r['federated']):>{col_w['federated']}} {d:>{col_w['delta']}}"
        )

    print(sep)

    # ── Per-metric averages across clients ───────────────────────────────────
    print("\n  Averages across all 3 clients:")
    for split in ["local", "global"]:
        print(f"\n  [{split} split]")
        for m in METRICS:
            subset = [r for r in rows if r["split"] == split and r["metric"] == m]
            i_vals = [r["individual"] for r in subset if r["individual"] is not None]
            f_vals = [r["federated"]  for r in subset if r["federated"]  is not None]
            i_mean = sum(i_vals) / len(i_vals) if i_vals else None
            f_mean = sum(f_vals) / len(f_vals) if f_vals else None
            print(
                f"    {m:<14} indiv={fmt(i_mean)}  fed={fmt(f_mean)}  "
                f"Δ={delta(f_mean, i_mean)}"
            )

    if args.save:
        Path(args.save).write_text(json.dumps(comparison, indent=2, default=str))
        print(f"\nComparison saved → {args.save}")


if __name__ == "__main__":
    main()
