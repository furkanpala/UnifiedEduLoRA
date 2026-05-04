"""
Student-facing inference script.

Given a saved BART+LoRA checkpoint and a context passage, generates one or more
QA pairs using the same prompt templates used during training.

Usage examples
--------------
# Baseline — no conditioning, just give me a question:
python infer.py --checkpoint path/to/best --context "Gradient descent is ..."

# Topic — ask for a question about a specific concept:
python infer.py --checkpoint path/to/best --conditioning topic \
    --topic "learning rate" --context "Gradient descent is ..."

# Bloom — ask for a question at a specific cognitive level:
python infer.py --checkpoint path/to/best --conditioning bloom \
    --bloom-level 4 --context "Gradient descent is ..."

# Read the context from a text file and generate 3 QA pairs:
python infer.py --checkpoint path/to/best --conditioning topic \
    --topic "backpropagation" --context-file lecture.txt --n 3

The checkpoint directory must contain:
  adapter_config.json  (written by PEFT when saving the LoRA adapter)
  adapter_model.safetensors (or adapter_model.bin)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BLOOM_VERBS = {
    1: "Remember", 2: "Understand", 3: "Apply",
    4: "Analyse",  5: "Evaluate",   6: "Create",
}

PROMPT_TEMPLATES = {
    "baseline": (
        "Generate a question and answer pair from the following "
        "machine learning text:\n\n{context}"
    ),
    "topic": (
        "Generate a question and answer pair about \"{topic}\" "
        "from the following machine learning text:\n\n{context}"
    ),
    "bloom": (
        "Generate a Bloom Level-{level} ({verb}) question and "
        "answer pair from the following machine learning text:\n\n{context}"
    ),
}


def build_prompt(args, context: str) -> str:
    if args.conditioning == "baseline":
        return PROMPT_TEMPLATES["baseline"].format(context=context)
    if args.conditioning == "topic":
        if not args.topic:
            sys.exit("--topic is required when --conditioning topic")
        return PROMPT_TEMPLATES["topic"].format(topic=args.topic, context=context)
    if args.conditioning == "bloom":
        level = max(1, min(6, args.bloom_level))
        return PROMPT_TEMPLATES["bloom"].format(
            level=level, verb=BLOOM_VERBS[level], context=context
        )
    sys.exit(f"Unknown conditioning: {args.conditioning!r}")


def load_model(checkpoint: str):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from peft import PeftModel

    base_model_id = "facebook/bart-base"
    print(f"Loading base {base_model_id} …")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    base = AutoModelForSeq2SeqLM.from_pretrained(base_model_id)

    print(f"Loading LoRA adapter from {checkpoint} …")
    model = PeftModel.from_pretrained(base, checkpoint)
    model.eval()
    return tokenizer, model


def generate(tokenizer, model, prompt: str, device, n: int) -> list[str]:
    import torch

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            num_beams=max(n * 2, 4),
            num_return_sequences=n,
            early_stopping=True,
        )

    return [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]


def parse_args():
    p = argparse.ArgumentParser(description="Generate QA pairs from a trained BART+LoRA model.")
    p.add_argument("--checkpoint",   required=True,
                   help="Path to the saved adapter checkpoint (the 'best/' folder).")
    p.add_argument("--conditioning", default="baseline",
                   choices=["baseline", "topic", "bloom"],
                   help="Conditioning mode (default: baseline).")
    p.add_argument("--topic",        default=None,
                   help="Topic label for --conditioning topic.")
    p.add_argument("--bloom-level",  type=int, default=2,
                   choices=range(1, 7), metavar="{1-6}",
                   help="Bloom level for --conditioning bloom (default: 2=Understand).")
    p.add_argument("--context",      default=None,
                   help="Context text (pass directly on the command line).")
    p.add_argument("--context-file", default=None,
                   help="Path to a plain-text file containing the context.")
    p.add_argument("--n",            type=int, default=1,
                   help="Number of QA pairs to generate (default: 1).")
    p.add_argument("--device",       default=None,
                   help="Device override, e.g. 'cuda' or 'cpu'. Auto-detected if omitted.")
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve context
    if args.context_file:
        context = Path(args.context_file).read_text(encoding="utf-8").strip()
    elif args.context:
        context = args.context.strip()
    else:
        sys.exit("Provide context via --context or --context-file.")

    # Device
    import torch
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer, model = load_model(args.checkpoint)
    model = model.to(device)

    prompt = build_prompt(args, context)

    print(f"\n{'─'*60}")
    print(f"Conditioning : {args.conditioning}")
    if args.conditioning == "topic":
        print(f"Topic        : {args.topic}")
    if args.conditioning == "bloom":
        level = args.bloom_level
        print(f"Bloom level  : {level} ({BLOOM_VERBS[level]})")
    print(f"Generating {args.n} QA pair(s) …")
    print(f"{'─'*60}\n")

    results = generate(tokenizer, model, prompt, device, args.n)

    for i, text in enumerate(results, 1):
        if args.n > 1:
            print(f"── QA pair {i} ──────────────────────────")
        print(text)
        print()


if __name__ == "__main__":
    main()
