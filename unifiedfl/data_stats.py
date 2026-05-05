"""
Report data statistics for one or more client data files.

Two intended uses:

1. Each collaborator runs it on their own data file and reports back the
   entry count (and any other stats the coordinator asks for) — so the
   coordinator can pick the balance cap for split.py without anyone having
   to share their actual data.

2. The coordinator runs it across all collected files (locally) to compute
   the cross-client summary and the recommended --balance value.

The loader is lenient: accepts proper JSON arrays AND concatenated top-level
objects (the format used by some pre-Phase-1 data pipelines).

Usage
-----

  # one client (what each participant runs)
  python data_stats.py my_data.json

  # multiple clients (what the coordinator runs)
  python data_stats.py \
      --client 0:client0_data.json \
      --client 1:client1_data.json \
      --client 2:client2_data.json \
      --save cross_client_stats.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Lenient JSON loader (matches split.py / validate.py --lenient)
# ─────────────────────────────────────────────────────────────────────────────

def _load(path: Path) -> List[Dict[str, Any]]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    if raw.startswith("["):
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
    decoder = json.JSONDecoder()
    records, idx = [], 0
    while idx < len(raw):
        while idx < len(raw) and raw[idx] in " \t\n\r":
            idx += 1
        if idx >= len(raw):
            break
        obj, idx = decoder.raw_decode(raw, idx)
        records.append(obj)
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Stats
# ─────────────────────────────────────────────────────────────────────────────

_BLOOM_STR_MAP = {
    "remember": 1, "recall": 1,
    "understand": 2, "comprehend": 2,
    "apply": 3, "application": 3,
    "analyse": 4, "analyze": 4, "analysis": 4,
    "evaluate": 5, "evaluation": 5,
    "create": 6, "synthesis": 6, "design": 6,
}


def compute_stats(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-file statistics. Tolerant of both Phase-1 and machine-enhanced schemas."""
    qa_counts: List[int] = []
    ctx_words: List[int] = []
    bloom_dist = {k: 0 for k in range(1, 7)}
    diff_dist = {"easy": 0, "medium": 0, "hard": 0, "other_or_missing": 0}
    total_qa = 0

    for entry in data:
        if not isinstance(entry, dict):
            continue
        ctx = entry.get("clean_context") or entry.get("context") or ""
        if isinstance(ctx, str) and ctx.strip():
            ctx_words.append(len(ctx.split()))

        pairs = entry.get("qa_pairs", [])
        if not isinstance(pairs, list):
            pairs = []
        qa_counts.append(len(pairs))

        for qa in pairs:
            if not isinstance(qa, dict):
                continue
            total_qa += 1

            bl = qa.get("bloom_level")
            if isinstance(bl, int) and 1 <= bl <= 6:
                bloom_dist[bl] += 1
            elif isinstance(bl, str):
                mapped = _BLOOM_STR_MAP.get(bl.lower().strip())
                if mapped is not None:
                    bloom_dist[mapped] += 1

            d = qa.get("difficulty", "")
            if d in ("easy", "medium", "hard"):
                diff_dist[d] += 1
            else:
                diff_dist["other_or_missing"] += 1

    def _summary(xs: List) -> Dict[str, float]:
        if not xs:
            return {"mean": 0, "median": 0, "min": 0, "max": 0}
        return {
            "mean":   round(statistics.mean(xs), 2),
            "median": statistics.median(xs),
            "min":    min(xs),
            "max":    max(xs),
        }

    return {
        "entries":                 len(data),
        "total_qa_pairs":          total_qa,
        "qa_per_entry":            _summary(qa_counts),
        "context_word_count":      _summary(ctx_words),
        "bloom_distribution":      bloom_dist,
        "difficulty_distribution": diff_dist,
    }


def _print_per_file(label: str, path: Path, stats: Dict[str, Any]) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}: {path}")
    print(f"{'='*60}")
    print(f"  Entries        : {stats['entries']}")
    print(f"  Total QA pairs : {stats['total_qa_pairs']}")
    qpe = stats['qa_per_entry']
    print(f"  QA per entry   : mean={qpe['mean']}  median={qpe['median']}  "
          f"range=[{qpe['min']}, {qpe['max']}]")
    cw = stats['context_word_count']
    print(f"  Context words  : mean={cw['mean']}  median={cw['median']}  "
          f"range=[{cw['min']}, {cw['max']}]")
    bd = stats['bloom_distribution']
    bd_parts = [f"L{k}={bd[k]}" for k in range(1, 7) if bd[k] > 0]
    if bd_parts:
        print(f"  Bloom levels   : " + "  ".join(bd_parts))
    dd = stats['difficulty_distribution']
    dd_parts = [f"{k}={v}" for k, v in dd.items() if v > 0]
    if dd_parts:
        print(f"  Difficulty     : " + "  ".join(dd_parts))


def _print_cross_client(all_stats: Dict[str, Dict[str, Any]]) -> None:
    if len(all_stats) <= 1:
        return
    print(f"\n{'='*60}")
    print(f"  Cross-client summary")
    print(f"{'='*60}")
    entry_counts = [(label, s["entries"]) for label, s in all_stats.items()]
    entry_counts.sort(key=lambda x: x[1])

    width = max(len(label) for label, _ in entry_counts)
    for label, n in entry_counts:
        print(f"  {label:<{width}}  {n:>5} entries")

    min_label, min_count = entry_counts[0]
    max_label, max_count = entry_counts[-1]
    print()
    print(f"  Minimum : {min_label} ({min_count} entries)")
    print(f"  Maximum : {max_label} ({max_count} entries)")
    if min_count > 0:
        print(f"  Ratio   : {max_count / min_count:.2f}x")
    print()
    print(f"  For a balanced split, run split.py with --balance to cap")
    print(f"  every client at {min_count} entries.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Report data statistics per client (handy for picking "
                    "the --balance cap before running split.py)."
    )
    p.add_argument(
        "files", nargs="*", metavar="DATA_FILE",
        help="One or more client data JSON files.",
    )
    p.add_argument(
        "--client", action="append", metavar="ID:DATA_PATH",
        help="Alternative spec: 'id:path/to/data.json'. Repeat for multiple clients.",
    )
    p.add_argument(
        "--save", default=None,
        help="Optional path to write the combined stats as JSON.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    targets: List[Tuple[str, Path]] = []
    for f in args.files or []:
        targets.append((Path(f).name, Path(f)))
    for spec in args.client or []:
        parts = spec.split(":", 1)
        if len(parts) != 2:
            sys.exit(f"--client must be 'id:path', got {spec!r}")
        cid, p_str = parts[0], parts[1]
        targets.append((f"client_{cid}", Path(p_str)))

    if not targets:
        sys.exit("No files specified. Pass one or more files or use --client id:path.")

    all_stats: Dict[str, Dict[str, Any]] = {}

    for label, path in targets:
        if not path.exists():
            print(f"\nWARNING: file not found: {path}")
            continue
        try:
            data = _load(path)
        except Exception as e:
            print(f"\nERROR loading {path}: {e}")
            continue
        stats = compute_stats(data)
        all_stats[label] = {**stats, "path": str(path)}
        _print_per_file(label, path, stats)

    _print_cross_client(all_stats)

    if args.save:
        Path(args.save).write_text(
            json.dumps(all_stats, indent=2), encoding="utf-8",
        )
        print(f"\nStats saved -> {args.save}")


if __name__ == "__main__":
    main()
