"""
Step 5 — Enhance all three datasets with per-QA question_topic and bloom_level.

Calls GPT-4o-mini once per entry (context + all its QA pairs) and annotates
each pair with:
  - question_topic : 2-4 word noun phrase for the ML concept tested
  - bloom_level    : integer 1-6 (Bloom's Taxonomy)

For MIT/Stanford the chunk-level bloom_level string in input_meta is used as a
fallback if the GPT call fails for a given pair. For Papers v2 there is no
fallback, so bloom_level defaults to 2 (Understand) on failure.

The script is resumable: entries whose qa_pairs already all have question_topic
are skipped. Progress is saved every SAVE_EVERY entries.

Usage (Colab terminal):
    python experiments/05_enhance_all_datasets.py \\
        --openai-api-key sk-... \\
        --data-dir     /content/drive/MyDrive/unifiedfl_fed_experiment/data
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))

SAVE_EVERY = 10

BLOOM_STR_TO_INT: dict[str, int] = {
    "remember": 1, "recall": 1,
    "understand": 2, "comprehend": 2,
    "apply": 3, "application": 3,
    "analyse": 4, "analyze": 4, "analysis": 4,
    "evaluate": 5, "evaluation": 5,
    "create": 6, "synthesis": 6, "design": 6,
}

SOURCES = [
    ("MIT",      "ML_QA_LectureNotes_MIT.json",          "ML_QA_LectureNotes_MIT_enhanced.json",          True),
    ("Stanford", "ML_QA_LectureNotes_StanfordCS229.json", "ML_QA_LectureNotes_StanfordCS229_enhanced.json", True),
    ("Papers",   "ML_QA_Papers_v2.json",                  "ML_QA_Papers_v2_enhanced.json",                  False),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--openai-api-key", required=True)
    p.add_argument("--data-dir",       required=True,
                   help="Directory that contains the source JSON files AND where "
                        "enhanced files will be written.")
    p.add_argument("--model",          default="gpt-4o-mini")
    p.add_argument("--max-retries",    type=int, default=3)
    return p.parse_args()


def _chunk_bloom_fallback(entry: dict, has_meta_bloom: bool) -> int:
    """Return chunk-level bloom_level as int, or 2 if unavailable."""
    if not has_meta_bloom:
        return 2
    raw = entry.get("input_meta", {}).get("bloom_level", "understand")
    return BLOOM_STR_TO_INT.get(str(raw).lower().strip(), 2)


def _already_enhanced(entry: dict) -> bool:
    """True if every qa_pair already has question_topic populated."""
    pairs = entry.get("qa_pairs", [])
    return pairs and all(p.get("question_topic", "").strip() for p in pairs)


def _annotate_entry(
    entry: dict,
    fallback_bloom: int,
    client,
    model: str,
    max_retries: int,
) -> bool:
    """
    Call GPT to annotate all qa_pairs in an entry with question_topic + bloom_level.
    Returns True on success, False if all retries fail.
    """
    ctx = entry.get("clean_context", "")
    pairs = entry.get("qa_pairs", [])
    if not pairs:
        return True

    # Build numbered QA list for the prompt
    qa_lines = "\n".join(
        f"{i+1}. Q: {p['question']}\n   A: {p['answer']}"
        for i, p in enumerate(pairs)
    )

    prompt = (
        f"Given the following ML educational context and the QA pairs extracted from it, "
        f"annotate each QA pair with:\n"
        f"  1. question_topic — a 2-4 word noun phrase naming the specific ML concept tested "
        f"(e.g. \"learning rate scheduling\", \"kernel trick\").\n"
        f"  2. bloom_level — integer 1-6 (1=Remember, 2=Understand, 3=Apply, "
        f"4=Analyse, 5=Evaluate, 6=Create).\n\n"
        f"Context:\n\"\"\"{ctx[:1200]}\"\"\"\n\n"
        f"QA pairs:\n{qa_lines}\n\n"
        f"Return ONLY valid JSON — no markdown, no commentary:\n"
        f"{{\"annotations\": ["
        f"{{\"question_topic\": \"...\", \"bloom_level\": N}}, ..."
        f"]}}"
    )

    delay = 4.0
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=512,
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            raw = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', raw)
            annotations = json.loads(raw).get("annotations", [])

            if len(annotations) != len(pairs):
                raise ValueError(
                    f"Expected {len(pairs)} annotations, got {len(annotations)}"
                )

            for pair, ann in zip(pairs, annotations):
                pair["question_topic"] = str(ann.get("question_topic", "")).strip() or "ML concept"
                bl = int(ann.get("bloom_level", fallback_bloom))
                pair["bloom_level"] = max(1, min(6, bl))

            return True

        except Exception as e:
            print(f"    [attempt {attempt}/{max_retries}] GPT annotation failed: {e}")
            if attempt < max_retries:
                time.sleep(delay)
                delay *= 2

    # Fallback: leave question_topic as empty string substitute, use chunk bloom
    for pair in pairs:
        if not pair.get("question_topic", "").strip():
            pair["question_topic"] = "ML concept"
        if "bloom_level" not in pair:
            pair["bloom_level"] = fallback_bloom
    return False


def enhance_dataset(
    input_path: Path,
    output_path: Path,
    has_meta_bloom: bool,
    client,
    model: str,
    max_retries: int,
    label: str,
) -> None:
    print(f"\n{'─'*60}")
    print(f"  {label}: {input_path.name}")
    print(f"{'─'*60}")

    # Load (resume from output if it exists)
    if output_path.exists():
        entries = json.loads(output_path.read_text(encoding="utf-8"))
        print(f"  Resuming from existing output ({len(entries)} entries).")
    else:
        entries = json.loads(input_path.read_text(encoding="utf-8"))
        print(f"  Loaded {len(entries)} entries from source.")

    n_skip = sum(1 for e in entries if _already_enhanced(e))
    print(f"  Already enhanced: {n_skip}/{len(entries)}")

    n_done, n_fail = 0, 0
    for i, entry in enumerate(entries):
        if _already_enhanced(entry):
            continue

        fallback_bloom = _chunk_bloom_fallback(entry, has_meta_bloom)
        ok = _annotate_entry(entry, fallback_bloom, client, model, max_retries)
        if ok:
            n_done += 1
        else:
            n_fail += 1

        if (i + 1) % SAVE_EVERY == 0 or i == len(entries) - 1:
            output_path.write_text(
                json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            pct = 100 * (n_skip + n_done + n_fail) / max(len(entries), 1)
            print(f"  [{pct:5.1f}%] entry {i+1}/{len(entries)} "
                  f"done={n_done} fail={n_fail}", flush=True)

    print(f"\n  {label} done. Annotated={n_done}, failed={n_fail}.")
    print(f"  Saved → {output_path}")


def main() -> None:
    args = parse_args()

    from openai import OpenAI
    client = OpenAI(api_key=args.openai_api_key)

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    for label, src_name, dst_name, has_meta_bloom in SOURCES:
        # Source: either the repo's data dir or the provided data_dir
        src = data_dir / src_name
        if not src.exists():
            src = REPO_DIR / "unifiedfl" / "data" / src_name
        if not src.exists():
            print(f"\nWARNING: source not found for {label}: {src} — skipping.")
            continue

        dst = data_dir / dst_name
        enhance_dataset(
            input_path=src,
            output_path=dst,
            has_meta_bloom=has_meta_bloom,
            client=client,
            model=args.model,
            max_retries=args.max_retries,
            label=label,
        )

    print("\nAll datasets enhanced.")


if __name__ == "__main__":
    main()
