"""
QA Generation API - call this once per context chunk after your own preprocessing.

YOUR RESPONSIBILITY (before calling this):
  1. Convert your raw source (PDF, PPTX, Word, images, etc.) to plain text.
  2. Split the text into context chunks:
       - 150-400 words per chunk.
       - Plain prose only - no bullet points, LaTeX, figure captions, headers.
  3. Call generate_qa_for_context() for each chunk.
  4. Collect results into a JSON array and save as described in OUTPUT FORMAT below.

OUTPUT FORMAT  (one JSON file per institution, a flat array of entries):
[
  {
    "entry_id":           "client<N>_<zero-padded index>",  // e.g. "client2_0042"
    "source_description": "<lecture name / paper title / slide deck>",
    "clean_context":      "<your 150-400 word plain-text chunk>",
    "context_topics":     ["<topic1>", "<topic2>", ...],    // filled by this function
    "qa_pairs": [
      {
        "question":              "<question text>",
        "answer":                "<answer text>",
        "question_topic":        "<specific ML concept this question tests>",
        "bloom_level":           <int 1-6>,
        "bloom_justification":   "<one sentence>",
        "difficulty":            "<easy|medium|hard>",
        "answerable_from_context": true
      }
    ]
  }
]

USAGE EXAMPLE:
  import json
  from generate_qa import generate_qa_for_context

  API_KEY    = "sk-..."
  CLIENT_ID  = 2                          # your assigned institution ID
  SOURCE     = "Intro to ML Lecture 2024" # human-readable label
  my_chunks  = [...]                      # your preprocessed plain-text chunks

  entries = []
  for i, chunk in enumerate(my_chunks):
      result = generate_qa_for_context(
          context = chunk,
          api_key = API_KEY,
          n_pairs = 5,
          model   = "gpt-4o",   # or "gpt-4o-mini" for cheaper drafts
      )
      if result is None:
          print(f"[SKIP] chunk {i} - not suitable or failed (check logs)")
          continue
      entries.append({
          "entry_id":           f"client{CLIENT_ID}_{i:04d}",
          "source_description": SOURCE,
          "clean_context":      chunk,
          "context_topics":     result["context_topics"],
          "qa_pairs":           result["qa_pairs"],
      })

  with open(f"client{CLIENT_ID}_data.json", "w", encoding="utf-8") as f:
      json.dump(entries, f, indent=2, ensure_ascii=False)
  print(f"Saved {len(entries)} entries.")
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("generate_qa")

# ─────────────────────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM = """\
You are an expert educational content designer specialising in machine learning \
and AI. Your task is to generate pedagogically rigorous question-answer pairs \
from ML educational text.

Core principles:
- Every question must be answerable SOLELY from the provided text.
- Questions must span a range of Bloom's Taxonomy cognitive levels.
- Answers must be complete, self-contained explanations - not bare phrases.

IMPORTANT - when to skip:
If the text contains no substantive educational content - for example it is a \
table of contents, an index, a list of references, an acknowledgements section, \
a figure caption list, author affiliations, or any other non-explanatory \
boilerplate - you must NOT generate QA pairs. Instead respond with:
{{"skip": true, "reason": "<one sentence explaining why>"}}
"""

_USER = """\
Analyse the following machine learning text and produce:
  1. context_topics - 2 to 6 concise ML/AI concept labels (2-4 words each).
  2. qa_pairs - exactly {n_pairs} question-answer pairs.

--- TOPIC GUIDELINES ---
context_topics:
  Focus on technical ML concepts, ordered from most to least central.
  GOOD: "Gradient Descent", "Learning Rate Sensitivity", "Mini-batch SGD"
  BAD:  "Introduction", "Overview", "Definition"

question_topic:
  One label (2-4 words) for the specific ML concept the question tests.
  Match or sub-topic of a context_topic. Be specific: "Vanishing Gradient" not "Backpropagation".

--- BLOOM'S TAXONOMY ---
  1 Remember  - define, list, name, state
  2 Understand - describe, summarise, explain
  3 Apply      - solve, demonstrate, compute
  4 Analyse    - compare, contrast, distinguish
  5 Evaluate   - critique, assess, justify
  6 Create     - design, propose, formulate
Cover a VARIETY of levels - do not cluster at 1 or 2.

--- HARD RULES ---
  - Answerable from the text alone (set answerable_from_context: true; omit pair if not).
  - No questions about "the author" or "the paper" or "the date" etc.
  - Answers must synthesise - do not copy a single sentence verbatim.
  - If the text has no substantive educational content, respond with {{"skip": true, "reason": "..."}} and nothing else.

TEXT:
\"\"\"{context}\"\"\"

Respond ONLY with a valid JSON object - no markdown, no commentary.
Either the full QA response:
{{
  "context_topics": ["<topic>", ...],
  "qa_pairs": [
    {{
      "question": "...",
      "answer": "...",
      "question_topic": "...",
      "bloom_level": <1-6>,
      "bloom_justification": "...",
      "difficulty": "<easy|medium|hard>",
      "answerable_from_context": true
    }}
  ]
}}
Or, if the text is not suitable for QA generation:
{{"skip": true, "reason": "<one sentence>"}}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_qa_for_context(
    context: str,
    api_key: str,
    n_pairs: int = 5,
    model: str = "gpt-4o",
    max_retries: int = 3,
) -> Optional[Dict[str, Any]]:
    """
    Generate QA pairs for a single context chunk.

    Args:
        context:     Plain-text passage, 150-400 words, one coherent ML topic.
        api_key:     Your OpenAI API key.
        n_pairs:     Number of QA pairs to generate (default 5).
        model:       OpenAI model - "gpt-4o" (suggested).
        max_retries: Number of retry attempts on parse or API failure.

    Returns:
        {
            "context_topics": ["topic1", ...],
            "qa_pairs":       [{question, answer, question_topic,
                                bloom_level, bloom_justification,
                                difficulty, answerable_from_context}, ...]
        }
        None if the model determines the context has no substantive educational
        content (table of contents, references, boilerplate, etc.), or if all
        retries fail due to API/parse errors.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai>=1.0")

    client = OpenAI(api_key=api_key)
    prompt = _USER.format(n_pairs=n_pairs, context=context.strip())
    delay  = 5.0

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.7,
                max_tokens=1800,
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError(f"Expected JSON object, got {type(parsed)}")

            if parsed.get("skip"):
                logger.info(f"Skipped by model: {parsed.get('reason', 'no reason given')}")
                return None

            topics = [t.strip() for t in parsed.get("context_topics", []) if str(t).strip()]
            pairs  = [
                qa for qa in parsed.get("qa_pairs", [])
                if qa.get("answerable_from_context", True)
                and qa.get("question", "").strip()
            ]
            return {"context_topics": topics, "qa_pairs": pairs}

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Attempt {attempt}: parse error - {e}")
        except Exception as e:
            logger.warning(f"Attempt {attempt}: API error - {e}")

        if attempt < max_retries:
            time.sleep(delay)
            delay *= 2

    logger.error("All retries failed for this context chunk.")
    return None
