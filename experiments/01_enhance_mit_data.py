"""
Step 1 — Add per-QA `question_topic` and integer `bloom_level` to the MIT
lecture-notes dataset. The raw file only carries entry-level Bloom labels and
no topic field, but topic / Bloom conditioning need per-pair labels.

Uses GPT-4o-mini. The script is resumable — it skips QA pairs that already
have both fields. Cost on the MIT dataset is ~$0.20-0.50.

Usage (Colab terminal):
    python experiments/01_enhance_mit_data.py \\
        --input  unifiedfl/data/ML_QA_LectureNotes_MIT.json \\
        --output /content/drive/MyDrive/unifiedfl_mit_experiment/data/ML_QA_LectureNotes_MIT_enhanced.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Repo paths
REPO_DIR = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, str(Path(__file__).parent))                  # for _common
sys.path.insert(0, str(Path(REPO_DIR) / "unifiedfl"))           # for data.preprocessing

from _common import load_openai_key, default_key_search_paths
from data.preprocessing import load_json


ENHANCE_PROMPT = (
    "Given a CONTEXT and a QUESTION-ANSWER pair, return JSON with two fields:\n"
    "- question_topic: a short noun phrase (3-6 words) naming the specific "
    "topic this QA pair targets within the context.\n"
    "- bloom_level: integer 1-6 according to Bloom's taxonomy (1=Remember, "
    "2=Understand, 3=Apply, 4=Analyze, 5=Evaluate, 6=Create).\n\n"
    "CONTEXT: {context}\n\n"
    "QUESTION: {question}\n"
    "ANSWER: {answer}\n\n"
    "Return ONLY a JSON object: {{\"question_topic\": \"...\", \"bloom_level\": <int>}}"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True, help="Raw JSON file (e.g. ML_QA_LectureNotes_MIT.json)")
    p.add_argument("--output", required=True, help="Enhanced JSON output path")
    p.add_argument("--openai-api-key", default=None,
                   help="Override key. By default loads from env / repo / drive.")
    p.add_argument("--drive-dir", default=None,
                   help="Optional Drive root that contains an openai_api_key file")
    p.add_argument("--save-every", type=int, default=10,
                   help="Save progress every N entries")
    p.add_argument("--model", default="gpt-4o-mini")
    return p.parse_args()


def enhance_one(client, model, context, question, answer):
    prompt = ENHANCE_PROMPT.format(
        context=context[:2000], question=question, answer=answer,
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0, max_tokens=128,
        response_format={"type": "json_object"},
    )
    parsed = json.loads(resp.choices[0].message.content.strip())
    level = int(parsed.get("bloom_level", 2))
    level = max(1, min(6, level))
    topic = str(parsed.get("question_topic", "")).strip() or "the main topic"
    return topic, level


def main() -> None:
    args = parse_args()

    # Load API key
    if args.openai_api_key and args.openai_api_key.startswith("sk-"):
        api_key, source = args.openai_api_key, "<--openai-api-key arg>"
    else:
        api_key, source = load_openai_key(*default_key_search_paths(REPO_DIR, args.drive_dir))
    if not api_key:
        print("ERROR: no OpenAI API key (set OPENAI_API_KEY, place a file named "
              "'openai_api_key' at the repo root, or pass --openai-api-key)", file=sys.stderr)
        sys.exit(2)
    print(f"OpenAI key loaded from: {source}", flush=True)

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    raw = load_json(Path(args.input))
    print(f"Loaded {len(raw)} entries / "
          f"{sum(len(e.get('qa_pairs', [])) for e in raw)} QA pairs from {args.input}",
          flush=True)

    # Resume support
    out_path = Path(args.output)
    if out_path.exists():
        enhanced = load_json(out_path)
        print(f"Resuming from existing enhanced file ({len(enhanced)} entries)", flush=True)
    else:
        enhanced = []

    done_index = {
        (e.get("input_index") or e.get("entry_id")): i
        for i, e in enumerate(enhanced)
    }

    n_qas_total = sum(len(e.get("qa_pairs", [])) for e in raw)
    qas_done = 0
    qas_called = 0

    for entry_idx, entry in enumerate(raw):
        key = entry.get("input_index") or entry.get("entry_id") or entry_idx
        existing = enhanced[done_index[key]] if key in done_index else None
        if existing and all(
            isinstance(qa.get("bloom_level"), int) and qa.get("question_topic")
            for qa in existing.get("qa_pairs", [])
        ):
            qas_done += len(existing.get("qa_pairs", []))
            continue

        new_entry = json.loads(json.dumps(entry))  # deep copy
        ctx = new_entry.get("clean_context", "")
        for qa in new_entry.get("qa_pairs", []):
            if isinstance(qa.get("bloom_level"), int) and qa.get("question_topic"):
                qas_done += 1
                continue
            try:
                topic, level = enhance_one(client, args.model, ctx,
                                           qa["question"], qa["answer"])
                qa["question_topic"] = topic
                qa["bloom_level"] = level
                qas_called += 1
            except Exception as e:
                print(f"  [warn] entry {entry_idx} qa failed: {e}", flush=True)
                qa.setdefault("question_topic", "the main topic")
                qa.setdefault("bloom_level", 2)
            qas_done += 1

        if existing:
            enhanced[done_index[key]] = new_entry
        else:
            done_index[key] = len(enhanced)
            enhanced.append(new_entry)

        if (entry_idx + 1) % args.save_every == 0:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(enhanced, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"  enhanced {entry_idx + 1}/{len(raw)} entries — "
                  f"{qas_done}/{n_qas_total} QAs ({qas_called} new API calls so far)",
                  flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(enhanced, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDone. Enhanced file saved to {out_path}")
    print(f"  total entries:           {len(enhanced)}")
    print(f"  total QA pairs:          {sum(len(e.get('qa_pairs', [])) for e in enhanced)}")
    print(f"  new API calls this run:  {qas_called}")


if __name__ == "__main__":
    main()
