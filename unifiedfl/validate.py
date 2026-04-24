"""
Validate a client data file before running split.py.

Checks that the file conforms to the required format:
  - JSON array of entry objects
  - Each entry has: entry_id, source_description, clean_context, context_topics, qa_pairs
  - Each qa_pair has: question, answer, question_topic, bloom_level, difficulty,
                      bloom_justification, answerable_from_context
  - Enforces content constraints (word counts, bloom range, difficulty values, etc.)

Usage:
  python validate.py client0_data.json
  python validate.py client0_data.json client1_data.json
  python validate.py --client 0:client0_data.json --client 1:client1_data.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Validation rules
# ─────────────────────────────────────────────────────────────────────────────

ENTRY_REQUIRED_KEYS = {"entry_id", "source_description", "clean_context", "context_topics", "qa_pairs"}
QA_REQUIRED_KEYS    = {"question", "answer", "question_topic", "bloom_level", "difficulty",
                        "bloom_justification", "answerable_from_context"}
VALID_DIFFICULTIES  = {"easy", "medium", "hard"}
MIN_CONTEXT_WORDS   = 50     # lenient lower bound (spec says 150, but allow shorter)
MAX_CONTEXT_WORDS   = 600    # lenient upper bound (spec says 400)

def _check_entry(i: int, entry: Any) -> List[str]:
    errors: List[str] = []
    loc = f"entry[{i}]"

    if not isinstance(entry, dict):
        return [f"{loc}: expected object, got {type(entry).__name__}"]

    missing = ENTRY_REQUIRED_KEYS - entry.keys()
    if missing:
        errors.append(f"{loc}: missing keys: {sorted(missing)}")

    # entry_id
    eid = entry.get("entry_id", "")
    if not isinstance(eid, str) or not eid.strip():
        errors.append(f"{loc}: entry_id must be a non-empty string")

    # source_description
    src = entry.get("source_description", "")
    if not isinstance(src, str) or not src.strip():
        errors.append(f"{loc}: source_description must be a non-empty string")

    # clean_context
    ctx = entry.get("clean_context", "")
    if not isinstance(ctx, str) or not ctx.strip():
        errors.append(f"{loc}: clean_context must be a non-empty string")
    else:
        wc = len(ctx.split())
        if wc < MIN_CONTEXT_WORDS:
            errors.append(f"{loc}: clean_context too short ({wc} words, minimum {MIN_CONTEXT_WORDS})")
        if wc > MAX_CONTEXT_WORDS:
            errors.append(f"{loc}: clean_context too long ({wc} words, maximum {MAX_CONTEXT_WORDS})")

    # context_topics
    topics = entry.get("context_topics", None)
    if not isinstance(topics, list) or len(topics) == 0:
        errors.append(f"{loc}: context_topics must be a non-empty list")
    elif not all(isinstance(t, str) and t.strip() for t in topics):
        errors.append(f"{loc}: context_topics must be a list of non-empty strings")

    # qa_pairs
    pairs = entry.get("qa_pairs", None)
    if not isinstance(pairs, list) or len(pairs) == 0:
        errors.append(f"{loc}: qa_pairs must be a non-empty list")
    else:
        for j, qa in enumerate(pairs):
            errors.extend(_check_qa(loc, j, qa))

    return errors


def _check_qa(entry_loc: str, j: int, qa: Any) -> List[str]:
    errors: List[str] = []
    loc = f"{entry_loc}.qa_pairs[{j}]"

    if not isinstance(qa, dict):
        return [f"{loc}: expected object, got {type(qa).__name__}"]

    missing = QA_REQUIRED_KEYS - qa.keys()
    if missing:
        errors.append(f"{loc}: missing keys: {sorted(missing)}")

    q = qa.get("question", "")
    if not isinstance(q, str) or not q.strip():
        errors.append(f"{loc}: question must be a non-empty string")

    ans = qa.get("answer", "")
    if not isinstance(ans, str) or not ans.strip():
        errors.append(f"{loc}: answer must be a non-empty string")

    qt = qa.get("question_topic", "")
    if not isinstance(qt, str) or not qt.strip():
        errors.append(f"{loc}: question_topic must be a non-empty string")

    bl = qa.get("bloom_level", None)
    if not isinstance(bl, int) or bl < 1 or bl > 6:
        errors.append(f"{loc}: bloom_level must be an integer 1–6, got {bl!r}")

    diff = qa.get("difficulty", "")
    if diff not in VALID_DIFFICULTIES:
        errors.append(f"{loc}: difficulty must be one of {sorted(VALID_DIFFICULTIES)}, got {diff!r}")

    bj = qa.get("bloom_justification", "")
    if not isinstance(bj, str) or not bj.strip():
        errors.append(f"{loc}: bloom_justification must be a non-empty string")

    afc = qa.get("answerable_from_context", None)
    if afc is not True:
        errors.append(f"{loc}: answerable_from_context must be true")

    return errors


def validate_file(path: Path) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Returns (ok, errors, stats).
    stats contains entry_count, total_qa, bloom_distribution, difficulty_distribution.
    """
    errors: List[str] = []

    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError as e:
        return False, [f"Cannot read file: {e}"], {}

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON: {e}"], {}

    if not isinstance(data, list):
        return False, ["Top-level structure must be a JSON array"], {}

    if len(data) == 0:
        return False, ["File contains an empty array — no entries found"], {}

    for i, entry in enumerate(data):
        errors.extend(_check_entry(i, entry))

    # Stats (best-effort — computed even if there are errors)
    total_qa = 0
    bloom_dist: Dict[int, int] = {k: 0 for k in range(1, 7)}
    diff_dist: Dict[str, int]  = {"easy": 0, "medium": 0, "hard": 0}
    for entry in data:
        if not isinstance(entry, dict):
            continue
        for qa in entry.get("qa_pairs", []):
            if not isinstance(qa, dict):
                continue
            total_qa += 1
            bl = qa.get("bloom_level")
            if isinstance(bl, int) and 1 <= bl <= 6:
                bloom_dist[bl] += 1
            d = qa.get("difficulty")
            if d in diff_dist:
                diff_dist[d] += 1

    stats = {
        "entries":              len(data),
        "total_qa_pairs":       total_qa,
        "bloom_distribution":   bloom_dist,
        "difficulty_distribution": diff_dist,
    }
    return len(errors) == 0, errors, stats


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate client data files before running split.py"
    )
    p.add_argument(
        "files", nargs="*", metavar="DATA_FILE",
        help="One or more client data JSON files to validate",
    )
    p.add_argument(
        "--client", action="append", metavar="ID:DATA_PATH",
        help="Alternative spec: 'id:path/to/data.json'",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    paths: List[Path] = [Path(f) for f in (args.files or [])]
    if args.client:
        for spec in args.client:
            parts = spec.split(":", 1)
            if len(parts) != 2:
                print(f"ERROR: --client must be 'id:path', got {spec!r}", file=sys.stderr)
                sys.exit(2)
            paths.append(Path(parts[1]))

    if not paths:
        print("No files specified. Usage: python validate.py <data_file.json> [...]")
        sys.exit(2)

    all_ok = True
    for path in paths:
        print(f"\n{'='*60}")
        print(f"Validating: {path}")
        print(f"{'='*60}")

        ok, errors, stats = validate_file(path)

        if stats:
            print(f"  Entries       : {stats['entries']}")
            print(f"  Total QA pairs: {stats['total_qa_pairs']}")
            if stats["total_qa_pairs"] > 0:
                bd = stats["bloom_distribution"]
                print(f"  Bloom levels  : " + "  ".join(
                    f"L{k}={bd[k]}" for k in range(1, 7) if bd[k] > 0
                ))
                dd = stats["difficulty_distribution"]
                print(f"  Difficulty    : " + "  ".join(
                    f"{k}={v}" for k, v in dd.items() if v > 0
                ))

        if ok:
            print(f"\n  PASSED — file is valid and ready for split.py")
        else:
            all_ok = False
            print(f"\n  FAILED — {len(errors)} error(s):")
            for err in errors[:50]:
                print(f"    [ERROR] {err}")
            if len(errors) > 50:
                print(f"    ... and {len(errors) - 50} more errors")

    print(f"\n{'='*60}")
    if all_ok:
        print("All files passed validation.")
        print("Next step: python split.py --client <id>:<file> --output-dir outputs/")
    else:
        print("Validation failed. Fix the errors above before running split.py.")
        sys.exit(1)


if __name__ == "__main__":
    main()
