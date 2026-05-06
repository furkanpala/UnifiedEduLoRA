from __future__ import annotations

from typing import Dict, List

import torch
from torch.utils.data import Dataset


PROMPT_TEMPLATE = (
    "Generate a question and answer pair from the following "
    "machine learning text:\n\n{context}"
)
TARGET_TEMPLATE = "Question: {question}\nAnswer: {answer}"

# ── Conditioning support ──────────────────────────────────────────────────────
# When training, the prompt can optionally embed extra conditioning fields drawn
# from the sample so the model learns to produce QAs that vary by topic or
# Bloom level. At inference time the same fields are supplied to control the
# style of the generated output.
PROMPT_TEMPLATES: Dict[str, str] = {
    "baseline": (
        "Generate a question and answer pair from the following "
        "machine learning text:\n\n{context}"
    ),
    "topic": (
        "Generate a question and answer pair about \"{question_topic}\" "
        "from the following machine learning text:\n\n{context}"
    ),
    "bloom": (
        "Generate a Bloom Level-{bloom_level} ({bloom_verb}) question and "
        "answer pair from the following machine learning text:\n\n{context}"
    ),
}

BLOOM_VERBS: Dict[int, str] = {
    1: "Remember", 2: "Understand", 3: "Apply",
    4: "Analyze",  5: "Evaluate",   6: "Create",
}


def render_prompt(sample: Dict[str, object], conditioning: str) -> str:
    """Render the prompt for a sample under a given conditioning mode."""
    if conditioning not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown conditioning: {conditioning!r}")
    template = PROMPT_TEMPLATES[conditioning]
    if conditioning == "baseline":
        return template.format(context=sample["context"])
    if conditioning == "topic":
        return template.format(
            context=sample["context"],
            question_topic=sample.get("question_topic") or "the main topic",
        )
    # bloom: clamp to [1, 6]. Use `is None` rather than truthiness so that an
    # explicit bloom_level=0 (which is out-of-range) gets clamped to 1 instead
    # of being silently replaced by the default of 2.
    raw = sample.get("bloom_level")
    level = 2 if raw is None else int(raw)
    level = max(1, min(6, level))
    return template.format(
        context=sample["context"],
        bloom_level=level,
        bloom_verb=BLOOM_VERBS[level],
    )


class QADataset(Dataset):
    """
    Dataset for conditional QA generation.

    Each sample maps a context string to a (question, answer) target.
    Tokenisation is performed lazily in __getitem__.

    `conditioning` selects the prompt template:
        - "baseline": context only (default, original behavior)
        - "topic":    prompt also names the question_topic field
        - "bloom":    prompt also names the bloom_level field
    """

    def __init__(
        self,
        samples: List[Dict[str, str]],
        tokenizer: object,
        max_input_len: int,
        max_target_len: int,
        conditioning: str = "baseline",
    ) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        self.conditioning = conditioning

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        input_text = render_prompt(sample, self.conditioning)
        target_text = TARGET_TEMPLATE.format(
            question=sample["question"], answer=sample["answer"]
        )

        input_enc = self.tokenizer(
            input_text,
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        target_enc = self.tokenizer(
            target_text,
            max_length=self.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = target_enc["input_ids"].squeeze(0)
        # Replace pad token id with -100 so it is ignored in the loss
        labels = labels.masked_fill(labels == self.tokenizer.pad_token_id, -100)

        return {
            "input_ids": input_enc["input_ids"].squeeze(0),
            "attention_mask": input_enc["attention_mask"].squeeze(0),
            "labels": labels,
        }
