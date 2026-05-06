"""
Re-evaluate an existing Phase-2 checkpoint with diverse beam search (Strategy C).

The original train_client.py used deterministic beam search, so the model produced
the SAME generation for every val sample sharing a context (5 references → 1 output).
This script:

  1. Loads BART/T5/LED + the LoRA adapter from <fold_dir>/best/lora_model/
  2. Reads the existing generated_qas_val.json to recover (context, reference) pairs
  3. Groups by unique context
  4. Generates K diverse outputs per context with diverse beam search
  5. Hungarian-matches the K generations to the K references (max ROUGE-L)
  6. Computes ROUGE-L / BLEU-4 / BERTScore on matched pairs
  7. Prints old vs new metrics side-by-side

No retraining required. Reads only the LoRA adapter weights already on disk.

Usage (Colab):
    python unifiedfl/eval_diverse_decoding.py \
        --fold-dir /content/drive/MyDrive/unifiedfl/outputs/client_0/fold1 \
        --base-model facebook/bart-base \
        --num-return-sequences 5 \
        --num-beam-groups 5 \
        --diversity-penalty 1.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))

from data.dataset import PROMPT_TEMPLATE
from evaluation.metrics import compute_all_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--fold-dir",     required=True,
                   help="Path to a foldN directory containing best/lora_model/ "
                        "and generated_qas_val.json")
    p.add_argument("--base-model",   required=True,
                   help="HF base model id, e.g. facebook/bart-base")
    p.add_argument("--num-return-sequences", type=int, default=5,
                   help="Number of diverse outputs per context (= refs per context)")
    p.add_argument("--decoding", choices=["dbs", "sampling"], default="dbs",
                   help="dbs = diverse beam search (recommended); "
                        "sampling = nucleus + top-k (no remote code required)")
    p.add_argument("--num-beam-groups",      type=int, default=5)
    p.add_argument("--num-beams",            type=int, default=10,
                   help="Total beams; must be a multiple of num-beam-groups")
    p.add_argument("--diversity-penalty",    type=float, default=1.0)
    p.add_argument("--top-p",                type=float, default=0.95,
                   help="Nucleus sampling p (only used when --decoding sampling)")
    p.add_argument("--top-k",                type=int,   default=50,
                   help="Top-k sampling cutoff (only used when --decoding sampling)")
    p.add_argument("--temperature",          type=float, default=1.0)
    p.add_argument("--max-input-len",        type=int,   default=512)
    p.add_argument("--max-target-len",       type=int,   default=128)
    p.add_argument("--device",               default="cuda")
    p.add_argument("--output-suffix",        default="diverse",
                   help="Suffix for output files (metrics_val_<suffix>.json etc.)")
    return p.parse_args()


def _parse_qa(text: str) -> Tuple[str, str]:
    """Split 'Question: …\\nAnswer: …' into (q, a). Mirrors evaluation/metrics._parse_qa."""
    import re
    q = re.search(r"(?i)question:\s*(.*?)(?=\nanswer:|\Z)", text, re.DOTALL)
    a = re.search(r"(?i)answer:\s*(.*?)$", text, re.DOTALL)
    return (q.group(1).strip() if q else "",
            a.group(1).strip() if a else text.strip())


def _group_by_context(records: List[dict]) -> Dict[str, List[dict]]:
    """Group records sharing the same context. Preserves insertion order."""
    groups: Dict[str, List[dict]] = {}
    for r in records:
        groups.setdefault(r["context"], []).append(r)
    return groups


@torch.no_grad()
def _generate_diverse(
    model, tokenizer, context: str, args, device: torch.device,
) -> List[str]:
    prompt = PROMPT_TEMPLATE.format(context=context)
    enc = tokenizer(prompt, max_length=args.max_input_len,
                    truncation=True, padding=False, return_tensors="pt").to(device)
    common = dict(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        num_return_sequences=args.num_return_sequences,
        max_new_tokens=args.max_target_len,
        no_repeat_ngram_size=3,
    )
    if args.decoding == "dbs":
        # Diverse beam search. Recent transformers (>=4.50) moved DBS to a
        # custom-code repo, so pass the migration kwargs unconditionally.
        out = model.generate(
            **common,
            num_beams=args.num_beams,
            num_beam_groups=args.num_beam_groups,
            diversity_penalty=args.diversity_penalty,
            early_stopping=True,
            custom_generate="transformers-community/group-beam-search",
            trust_remote_code=True,
        )
    else:
        # Nucleus + top-k sampling — fully local, no remote code.
        out = model.generate(
            **common,
            do_sample=True,
            top_p=args.top_p,
            top_k=args.top_k,
            temperature=args.temperature,
        )
    return [tokenizer.decode(ids, skip_special_tokens=True) for ids in out]


def _hungarian_match(
    generations: List[str], references: List[str],
) -> List[Tuple[str, str]]:
    """
    Pair each generation with one reference such that total ROUGE-L is maximized.
    Returns list of (generation, reference) pairs in matched order.
    """
    from rouge_score import rouge_scorer
    from scipy.optimize import linear_sum_assignment

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    n = max(len(generations), len(references))
    cost = np.full((n, n), 1.0)  # 1 - rouge → minimize = maximize rouge
    for i, g in enumerate(generations):
        for j, r in enumerate(references):
            cost[i, j] = 1.0 - scorer.score(r, g)["rougeL"].fmeasure
    row_idx, col_idx = linear_sum_assignment(cost)
    pairs = []
    for i, j in zip(row_idx, col_idx):
        if i < len(generations) and j < len(references):
            pairs.append((generations[i], references[j]))
    return pairs


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    fold_dir = Path(args.fold_dir)

    # ── Load model + LoRA ─────────────────────────────────────────────────────
    print(f"Loading base {args.base_model} …")
    base = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)
    tok  = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    lora_dir = fold_dir / "best" / "lora_model"
    print(f"Loading LoRA adapter from {lora_dir} …")
    model = PeftModel.from_pretrained(base, str(lora_dir))
    model.to(device).eval()

    # ── Load existing val records ─────────────────────────────────────────────
    qa_path = fold_dir / "generated_qas_val.json"
    print(f"Loading val records from {qa_path} …")
    records = json.loads(qa_path.read_text(encoding="utf-8"))
    print(f"  total samples = {len(records)}")
    groups = _group_by_context(records)
    print(f"  unique contexts = {len(groups)}")
    print(f"  refs per context = {len(records) // max(len(groups), 1)}")

    # ── Generate K diverse outputs per context ───────────────────────────────
    if args.decoding == "dbs":
        print(f"\nGenerating {args.num_return_sequences} diverse outputs per context "
              f"[diverse beam search: beams={args.num_beams}, groups={args.num_beam_groups}, "
              f"diversity_penalty={args.diversity_penalty}] …")
    else:
        print(f"\nGenerating {args.num_return_sequences} diverse outputs per context "
              f"[sampling: top_p={args.top_p}, top_k={args.top_k}, "
              f"temperature={args.temperature}] …")

    matched_preds: List[str] = []
    matched_refs:  List[str] = []
    distinct_counts: List[int] = []
    detailed: List[dict] = []

    for ctx_idx, (ctx, recs) in enumerate(groups.items()):
        refs = [r["reference"] for r in recs]
        gens = _generate_diverse(model, tok, ctx, args, device)
        n_distinct = len(set(gens))
        distinct_counts.append(n_distinct)
        pairs = _hungarian_match(gens, refs)
        for g, r in pairs:
            matched_preds.append(g)
            matched_refs.append(r)
        detailed.append({
            "context_idx": ctx_idx,
            "context_preview": ctx[:120].replace("\n", " "),
            "n_distinct_generations": n_distinct,
            "generations": gens,
            "references": refs,
            "matched_pairs": [{"gen": g, "ref": r} for g, r in pairs],
        })
        print(f"  ctx {ctx_idx + 1}/{len(groups)}: "
              f"{n_distinct}/{args.num_return_sequences} distinct generations")

    # ── Score matched pairs ───────────────────────────────────────────────────
    print("\nComputing ROUGE-L, BLEU-4, BERTScore on matched pairs …")
    new_metrics = compute_all_metrics(matched_preds, matched_refs, device)

    # ── Compare with original metrics_val.json ────────────────────────────────
    old_path = fold_dir / "metrics_val.json"
    old_metrics = {}
    if old_path.exists():
        old_full = json.loads(old_path.read_text())
        for k in ("rouge_l", "bleu_4", "bertscore_f1"):
            old_metrics[k] = old_full.get(k, float("nan"))

    print(f"\n{'─' * 60}")
    print(f"  Diverse decoding results — fold = {fold_dir.name}")
    print(f"{'─' * 60}")
    print(f"  unique contexts        : {len(groups)}")
    print(f"  refs per context       : {args.num_return_sequences}")
    print(f"  mean distinct gens/ctx : {np.mean(distinct_counts):.2f} "
          f"/ {args.num_return_sequences}")
    print(f"  matched pairs scored   : {len(matched_preds)}")
    print()
    print(f"  {'metric':<14} {'original':>12} {'diverse + match':>18} {'Δ':>10}")
    for k in ("rouge_l", "bleu_4", "bertscore_f1"):
        old = old_metrics.get(k, float("nan"))
        new = new_metrics[k]
        delta = new - old if old == old else float("nan")  # NaN check
        print(f"  {k:<14} {old:>12.4f} {new:>18.4f} {delta:>+10.4f}")

    # ── Persist outputs ───────────────────────────────────────────────────────
    out_metrics = fold_dir / f"metrics_val_{args.output_suffix}.json"
    cfg = {
        "decoding": "diverse_beam_search" if args.decoding == "dbs" else "sampling",
        "num_return_sequences": args.num_return_sequences,
    }
    if args.decoding == "dbs":
        cfg.update({
            "num_beams": args.num_beams,
            "num_beam_groups": args.num_beam_groups,
            "diversity_penalty": args.diversity_penalty,
        })
    else:
        cfg.update({
            "top_p": args.top_p,
            "top_k": args.top_k,
            "temperature": args.temperature,
        })
    out_metrics.write_text(
        json.dumps({
            **cfg,
            "n_unique_contexts": len(groups),
            "mean_distinct_generations": float(np.mean(distinct_counts)),
            "matching": "hungarian_max_rougeL",
            **new_metrics,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n  metrics saved → {out_metrics}")

    out_qa = fold_dir / f"generated_qas_val_{args.output_suffix}.json"
    out_qa.write_text(json.dumps(detailed, indent=2, ensure_ascii=False),
                      encoding="utf-8")
    print(f"  detailed generations saved → {out_qa}")


if __name__ == "__main__":
    main()
