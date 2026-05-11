"""
Aggregate per-architecture fed-SLM-vs-GPT-4o comparisons into the
UniEdu-mixture row that the manuscript's H2 table expects.

Reads:
    experiment_outputs/slm_vs_llm/{bart,t5,led}/metrics_per_run.json

For each (slice, metric) the mixture value is the mean across the 3
architectures' fold-mean values:

    mixture[slice][metric] = mean({BART_avg, T5_avg, LED_avg})

The cross-architecture std (`_amean_std`) reflects disagreement between
the 3 SLMs on that slice/metric — separate from the fold-std reported
inside each architecture's own metrics file.

Outputs:
    experiment_outputs/slm_vs_llm/mixture_summary.{json,txt}

The .txt table renders as the H2-style row order:
    fed-BART | fed-T5 | fed-LED | UniEdu (mixture) | GPT-4o

…per slice (client_0/1/2 + OVERALL), per lightweight metric.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO = Path(__file__).resolve().parent.parent

METRIC_KEYS  = ("rouge_l", "bleu_4", "bertscore_f1", "blooms_cls_evs_mean")
METRIC_LABELS = ("ROUGE-L", "BLEU-4", "BERTScore-F1", "Bloom-EVS")
SLICE_KEYS   = ("client_0", "client_1", "client_2", "overall")
SLICE_LABELS = ("client_0 (MIT)", "client_1 (Stanford)",
                "client_2 (ML papers)", "OVERALL")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir",
                   default=str(REPO / "experiment_outputs/slm_vs_llm"),
                   help="Parent dir containing {bart,t5,led}/metrics_per_run.json")
    p.add_argument("--arch-dirs", nargs="+",
                   default=["bart", "t5", "led"],
                   help="Per-architecture subdir names in --base-dir.")
    p.add_argument("--out-prefix", default="mixture_summary",
                   help="Filename stem for the two output files under --base-dir.")
    return p.parse_args()


def load_arch_summary(metrics_path: Path) -> Dict[str, Any]:
    """Load one architecture's metrics_per_run.json and pull out the
    fold-avg slice block + GPT-4o slice block + display label."""
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    meta = data.get("_meta", {})
    avg  = data.get("fedbart_avg", {})  # legacy key — same shape for any SLM
    gpt  = data.get("gpt4o", {})
    return {
        "label":   meta.get("model_label", metrics_path.parent.name),
        "n_rows":  meta.get("n_rows"),
        "folds":   meta.get("folds", []),
        "fed_avg": avg,
        "gpt4o":   gpt,
    }


def _avg(values: List[float]) -> Tuple[float, float]:
    """Mean + sample std (ddof=1) over a list of floats. Returns (NaN, NaN)
    on an empty list; (mu, 0.0) on a single value."""
    valid = [v for v in values if v is not None and not math.isnan(v)]
    if not valid:
        return (float("nan"), float("nan"))
    mu = sum(valid) / len(valid)
    if len(valid) == 1:
        return (mu, 0.0)
    var = sum((v - mu) ** 2 for v in valid) / (len(valid) - 1)
    return (mu, math.sqrt(var))


def compute_mixture(arch_summaries: List[Dict[str, Any]]
                    ) -> Dict[str, Dict[str, Dict[str, float]]]:
    """For each slice and metric, compute mean ± cross-arch std over the
    3 per-architecture fold-means."""
    mixture: Dict[str, Dict[str, Dict[str, float]]] = {}
    for slice_key in SLICE_KEYS:
        mixture[slice_key] = {}
        for mk in METRIC_KEYS:
            per_arch = [s["fed_avg"].get(slice_key, {}).get(f"{mk}_mean")
                        for s in arch_summaries]
            mu, sd = _avg([v for v in per_arch if v is not None])
            mixture[slice_key][mk] = {
                "mean":    mu,
                "std":     sd,
                "n_archs": sum(1 for v in per_arch if v is not None),
                "per_arch": dict(zip([s["label"] for s in arch_summaries],
                                     per_arch)),
            }
        # n is the same for all 3 archs on a given slice; pick from the first.
        ns = [s["fed_avg"].get(slice_key, {}).get("n") for s in arch_summaries]
        mixture[slice_key]["n"] = next((n for n in ns if n is not None), None)
    return mixture


def gpt4o_consensus(arch_summaries: List[Dict[str, Any]]) -> Dict[str, dict]:
    """Sanity-check that all 3 architectures saw the same GPT-4o numbers
    (they should — same cached predictions and same metric implementations).
    Returns the first architecture's GPT-4o block; logs disagreement."""
    blocks = [s["gpt4o"] for s in arch_summaries if s["gpt4o"]]
    if not blocks:
        return {}
    base = blocks[0]
    for slice_key in SLICE_KEYS:
        for mk in METRIC_KEYS:
            ref = base.get(slice_key, {}).get(mk)
            for other in blocks[1:]:
                v = other.get(slice_key, {}).get(mk)
                if ref is None or v is None:
                    continue
                if abs(ref - v) > 1e-6:
                    print(f"  WARNING: GPT-4o disagreement on "
                          f"{slice_key}/{mk}: {ref:.6f} vs {v:.6f} — "
                          f"using first architecture's value",
                          file=sys.stderr)
    return base


def format_table(arch_summaries: List[Dict[str, Any]],
                 mixture: Dict, gpt: Dict) -> str:
    """Render a manuscript-style table with all 3 architectures, the mixture
    row, and the GPT-4o reference row — one block per slice."""
    out: List[str] = []
    out.append("=" * 110)
    out.append("  UniEdu federated mixture vs GPT-4o  |  global mixed test  |  4 lightweight metrics")
    out.append("=" * 110)

    header = (f"  {'slice':10s}  {'model':24s}  "
              + "  ".join(f"{m:>14s}" for m in METRIC_LABELS) + "    n")
    out.append(header)
    out.append("  " + "-" * (len(header) - 2))

    for slice_key, slice_label in zip(SLICE_KEYS, SLICE_LABELS):
        # Per-architecture rows
        for s in arch_summaries:
            avg = s["fed_avg"].get(slice_key, {})
            n   = avg.get("n", "--")
            cells: List[str] = []
            for mk in METRIC_KEYS:
                mu = avg.get(f"{mk}_mean")
                sd = avg.get(f"{mk}_std", 0.0)
                cells.append(f"{mu:.4f}±{sd:.4f}" if mu is not None
                             else "      --      ")
            out.append(f"  {slice_label:10s}  {s['label']:24s}  "
                       f"{'  '.join(f'{c:>14s}' for c in cells)}    {n}")

        # Mixture row (mean across the 3 archs)
        n = mixture[slice_key].get("n", "--")
        cells = []
        for mk in METRIC_KEYS:
            mu = mixture[slice_key][mk]["mean"]
            sd = mixture[slice_key][mk]["std"]
            cells.append(f"{mu:.4f}±{sd:.4f}" if not math.isnan(mu)
                         else "      --      ")
        out.append(f"  {slice_label:10s}  {'UniEdu (mixture)':24s}  "
                   f"{'  '.join(f'{c:>14s}' for c in cells)}    {n}")

        # GPT-4o row
        gblk = gpt.get(slice_key, {})
        cells = []
        for mk in METRIC_KEYS:
            v = gblk.get(mk)
            cells.append(f"{v:.4f}      " if v is not None
                         else "      --      ")
        out.append(f"  {slice_label:10s}  {'GPT-4o':24s}  "
                   f"{'  '.join(f'{c:>14s}' for c in cells)}    {n}")

        # Delta: GPT-4o − UniEdu mixture
        cells = []
        for mk in METRIC_KEYS:
            mu = mixture[slice_key][mk]["mean"]
            v  = gblk.get(mk)
            if mu is None or v is None or math.isnan(mu):
                cells.append(f"{'--':>14s}")
            else:
                sign = "+" if v >= mu else ""
                cells.append(f"{sign}{v - mu:.4f}".rjust(14))
        out.append(f"  {'':10s}  {'  delta (GPT − mix)':24s}  "
                   + "  ".join(cells))
        out.append("")

    return "\n".join(out)


def main() -> None:
    args = parse_args()
    base = Path(args.base_dir)

    arch_summaries: List[Dict[str, Any]] = []
    for d in args.arch_dirs:
        p = base / d / "metrics_per_run.json"
        if not p.exists():
            print(f"  [skip] {p} missing", file=sys.stderr)
            continue
        arch_summaries.append(load_arch_summary(p))
        print(f"  loaded {p}  → label={arch_summaries[-1]['label']!r} "
              f"n_rows={arch_summaries[-1]['n_rows']} "
              f"folds={arch_summaries[-1]['folds']}")

    if not arch_summaries:
        sys.exit("No per-architecture metrics found. Run "
                 "experiments/15_compare_fedbart_vs_gpt4o.py for each "
                 "client_id first.")

    mixture = compute_mixture(arch_summaries)
    gpt = gpt4o_consensus(arch_summaries)

    out_json = base / f"{args.out_prefix}.json"
    out_json.write_text(json.dumps({
        "architectures": [s["label"] for s in arch_summaries],
        "per_arch":      {s["label"]: {"fed_avg": s["fed_avg"],
                                       "n_rows":  s["n_rows"],
                                       "folds":   s["folds"]}
                          for s in arch_summaries},
        "mixture":       mixture,
        "gpt4o":         gpt,
    }, indent=2), encoding="utf-8")
    print(f"\nMixture JSON → {out_json}")

    txt = format_table(arch_summaries, mixture, gpt)
    print("\n" + txt)
    out_txt = base / f"{args.out_prefix}.txt"
    out_txt.write_text(txt, encoding="utf-8")
    print(f"\nMixture table → {out_txt}")


if __name__ == "__main__":
    main()
