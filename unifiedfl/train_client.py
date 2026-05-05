"""
Train a single client with LoRA only (no GNN, no FiLM) — baseline individual training.
Supports 3-fold CV, early stopping, and comprehensive val-set evaluation.
Designed for Colab + Drive.

Usage (Colab):
    # Mount Drive first:
    #   from google.colab import drive; drive.mount('/content/drive')

    python train_client.py \
        --client-id 0 \
        --fold 1 \
        --model google/flan-t5-small \
        --family t5 \
        --targets q v \
        --splits-dir /content/drive/MyDrive/unifiedfl/outputs/splits \
        --output-dir /content/drive/MyDrive/unifiedfl/outputs \
        --num-epochs 60

    # Resume from checkpoint:
    python train_client.py ... --fold 1 --resume-from-epoch 30

    # With comprehensive OpenAI evaluation:
    python train_client.py ... --fold 1 --openai-api-key sk-...
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import gc

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent))

from data.dataset import QADataset, render_prompt
from evaluation.metrics import compute_all_metrics, compute_comprehensive_metrics
from models.client_model import ClientModel
from utils.logging_utils import setup_logging


def _ensure_nltk_punkt() -> None:
    """Make sure NLTK sentence tokenizer data is available."""
    import nltk
    for pkg in ("punkt_tab", "punkt"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
            return
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
                return
            except Exception:
                continue


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a single client with LoRA")

    # Client / model
    p.add_argument("--client-id", type=int, required=True)
    p.add_argument("--fold",      type=int, required=True, choices=[1, 2, 3],
                   help="Which CV fold to train on (1, 2, or 3)")
    p.add_argument("--model",     required=True, help="HuggingFace model ID")
    p.add_argument("--family",    required=True, choices=["t5", "bart", "led"])
    p.add_argument("--targets",   nargs="+", required=True,
                   help="LoRA target module names, e.g. --targets q v")

    # Data
    p.add_argument("--splits-dir", default="outputs/splits",
                   help="Directory containing split JSON files from split.py")

    # Training
    p.add_argument("--num-epochs",     type=int,   default=100)
    p.add_argument("--batch-size",     type=int,   default=4)
    p.add_argument("--lr",             type=float, default=3e-4)
    p.add_argument("--warmup-ratio",   type=float, default=0.1)
    p.add_argument("--grad-clip",      type=float, default=1.0)
    p.add_argument("--max-input-len",  type=int,   default=512)
    p.add_argument("--max-target-len", type=int,   default=128)

    # Early stopping
    p.add_argument("--patience",  type=int,   default=10,
                   help="Stop if the early-stop metric does not improve for this many epochs.")
    p.add_argument("--min-delta", type=float, default=1e-4,
                   help="Minimum improvement in the early-stop metric to reset the patience counter.")
    p.add_argument("--early-stop-metric", choices=["rouge_l", "val_loss"],
                   default="rouge_l",
                   help="Metric tracked for early stopping and best-checkpoint selection. "
                        "rouge_l (default): greedy-decoded ROUGE-L on a fixed val subsample - "
                        "directly reflects autoregressive generation quality. "
                        "val_loss: teacher-forced cross-entropy on the full val set - faster "
                        "but can decouple from real generation quality (exposure bias).")
    p.add_argument("--fast-eval-samples", type=int, default=50,
                   help="Number of val samples used for the fast ROUGE-L early-stop metric. "
                        "Drawn once with seed=args.seed and reused every epoch.")

    # LoRA
    p.add_argument("--lora-r",       type=int,   default=16)
    p.add_argument("--lora-alpha",   type=int,   default=32)
    p.add_argument("--lora-dropout", type=float, default=0.1)

    # Conditioning
    p.add_argument("--conditioning", choices=["baseline", "topic", "bloom"],
                   default="baseline",
                   help="Prompt-conditioning mode for both training and evaluation. "
                        "baseline = context only (uses diverse beam search at eval). "
                        "topic    = prompt names sample.question_topic. "
                        "bloom    = prompt names sample.bloom_level.")
    p.add_argument("--eval-num-return-sequences", type=int, default=5,
                   help="Only used when --conditioning baseline: K diverse outputs "
                        "per unique context, then Hungarian-matched to references.")
    p.add_argument("--eval-num-beams",            type=int, default=10)
    p.add_argument("--eval-num-beam-groups",      type=int, default=5)
    p.add_argument("--eval-diversity-penalty",    type=float, default=1.0)

    # Checkpointing / resuming
    p.add_argument("--checkpoint-every", type=int, default=5,
                   help="Save a checkpoint every N epochs (0 = only at end)")
    p.add_argument("--resume-from-epoch", type=int, default=0,
                   help="Resume from this epoch's checkpoint (0 = start fresh)")

    # Output / preview
    p.add_argument("--output-dir",    default="outputs/",
                   help="Root output dir (can be a Drive path on Colab)")
    p.add_argument("--preview-every", type=int, default=5,
                   help="Generate a sample QA from the val set every N epochs (0 = off)")
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--device", default="cuda")

    # Post-training evaluation
    p.add_argument("--openai-api-key", default=None,
                   help="OpenAI API key for Answer Relevancy and Bloom's LLM judge")
    p.add_argument("--no-heavy", action="store_true",
                   help="Skip heavy reference-free metrics (UnifiedQA + DeBERTa) in final eval")
    p.add_argument("--blooms-model", default="cip29/bert-blooms-taxonomy-classifier",
                   help="HuggingFace model ID for the local Bloom's classifier "
                        "(set to '' to skip)")
    p.add_argument("--global-test-file", default=None,
                   help="Path to a JSON file of flat QA samples. If provided, the best "
                        "checkpoint is also evaluated on this set and results are saved "
                        "to metrics_global_test.json alongside metrics_val.json.")

    return p.parse_args()


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_split(splits_dir: Path, client_id: int, fold: int, name: str) -> list:
    """Load a fold-specific split file (train or val)."""
    path = splits_dir / f"client_{client_id}_fold{fold}_{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found: {path}\n"
            "Run split.py first to generate splits."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _set_lora_weights(client_model: ClientModel, lora_dir: Path) -> None:
    """Overwrite the existing 'default' LoRA adapter with weights from disk.

    PEFT's `load_adapter` ADDS a new adapter — it cannot overwrite one that
    already exists. We instead load the saved state dict directly into the
    already-attached default adapter.
    """
    from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
    state = load_peft_weights(str(lora_dir), device=str(client_model.device))
    set_peft_model_state_dict(client_model.model, state)


def _save_checkpoint(
    client_model: ClientModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    train_loss_history: list,
    val_loss_history: list,
    best_val_loss: float,
    patience_count: int,
    ckpt_dir: Path,
    es_metric_history: list = None,
    best_es_metric: float = None,
) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    client_model.model.save_pretrained(str(ckpt_dir / "lora_model"))
    torch.save(
        {
            "epoch":              epoch,
            "optimizer":          optimizer.state_dict(),
            "scheduler":          scheduler.state_dict(),
            "train_loss_history": train_loss_history,
            "val_loss_history":   val_loss_history,
            "es_metric_history":  es_metric_history if es_metric_history is not None else [],
            "best_es_metric":     best_es_metric if best_es_metric is not None else best_val_loss,
            # Kept for backward compatibility with old checkpoints
            "best_val_loss":      best_val_loss,
            "patience_count":     patience_count,
            "rng_python":         random.getstate(),
            "rng_numpy":          np.random.get_state(),
            "rng_torch":          torch.get_rng_state(),
            "rng_torch_cuda":     (torch.cuda.get_rng_state_all()
                                   if torch.cuda.is_available() else None),
        },
        ckpt_dir / "training_state.pt",
    )
    print(f"  [checkpoint] saved → {ckpt_dir}")


def _load_checkpoint(
    client_model: ClientModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    ckpt_dir: Path,
) -> dict:
    lora_dir = ckpt_dir / "lora_model"
    if not lora_dir.exists():
        raise FileNotFoundError(f"Checkpoint not found: {lora_dir}")
    _set_lora_weights(client_model, lora_dir)
    state = torch.load(ckpt_dir / "training_state.pt", map_location="cpu")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if "rng_python" in state:
        random.setstate(state["rng_python"])
        np.random.set_state(state["rng_numpy"])
        torch.set_rng_state(state["rng_torch"])
        if state.get("rng_torch_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["rng_torch_cuda"])
    print(f"  [checkpoint] resumed from epoch {state['epoch']} ← {ckpt_dir}")
    return state


def _save_best(client_model: ClientModel, best_dir: Path) -> None:
    best_dir.mkdir(parents=True, exist_ok=True)
    client_model.model.save_pretrained(str(best_dir / "lora_model"))


def _load_best(client_model: ClientModel, best_dir: Path) -> None:
    lora_dir = best_dir / "lora_model"
    if lora_dir.exists():
        _set_lora_weights(client_model, lora_dir)
        print(f"  [best model] loaded ← {lora_dir}")
    else:
        print("  [best model] not found — using final weights for evaluation")


@torch.no_grad()
def _preview(
    client_model: ClientModel,
    sample: dict,
    args: argparse.Namespace,
    device: torch.device,
    use_amp: bool,
    epoch: int,
) -> None:
    client_model.model.eval()
    prompt = (
        "Generate a question and answer pair from the following machine learning text:"
        f"\n\n{sample['context']}"
    )
    enc = client_model.tokenizer(
        prompt, max_length=args.max_input_len,
        truncation=True, padding=False, return_tensors="pt",
    )
    with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
        out_ids = client_model.generate(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
            num_beams=4, max_new_tokens=args.max_target_len,
            no_repeat_ngram_size=3, early_stopping=True,
        )
    generated = client_model.tokenizer.decode(out_ids[0], skip_special_tokens=True)
    ctx = sample["context"][:300].replace("\n", " ")
    if len(sample["context"]) > 300:
        ctx += "…"
    ref_q = sample.get("question", "")[:120]
    ref_a = sample.get("answer", "")[:120]
    print(f"\n  ┌─ Epoch {epoch} val preview {'─' * 32}")
    print(f"  │ CONTEXT  : {ctx}")
    print(f"  │ REFERENCE: Question: {ref_q}")
    print(f"  │            Answer:   {ref_a}")
    print(f"  │ GENERATED: {generated}")
    print(f"  └{'─' * 50}")
    client_model.model.train()


@torch.no_grad()
def _fast_rouge_l(
    client_model: ClientModel,
    samples: list,
    args: argparse.Namespace,
    device: torch.device,
    use_amp: bool,
) -> float:
    """
    Greedy-decoded ROUGE-L on a fixed val subsample. Used as the early-stop
    metric. Per-sample greedy gives a fast, deterministic signal that tracks
    autoregressive generation quality (unlike teacher-forced val loss).
    """
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    tokenizer = client_model.tokenizer

    client_model.model.eval()
    total = 0.0
    for sample in samples:
        prompt = render_prompt(sample, args.conditioning)
        enc = tokenizer(
            prompt, max_length=args.max_input_len,
            truncation=True, padding=False, return_tensors="pt",
        ).to(device)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            # Greedy decoding (num_beams=1) — early_stopping is omitted on
            # purpose because it only applies to beam search and emits a
            # noisy UserWarning when set with num_beams=1.
            out_ids = client_model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                num_beams=1,
                max_new_tokens=args.max_target_len,
                no_repeat_ngram_size=3,
            )
        gen = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        ref = f"Question: {sample['question']}\nAnswer: {sample['answer']}"
        total += scorer.score(ref, gen)["rougeL"].fmeasure
    client_model.model.train()
    return total / max(len(samples), 1)


@torch.no_grad()
def _evaluate(
    client_model: ClientModel,
    samples: list,
    args: argparse.Namespace,
    device: torch.device,
    use_amp: bool,
) -> tuple[list, list, list]:
    """
    Generate predictions for all val samples, returned in val-sample order so
    each prediction lines up with its reference.

    Two evaluation strategies depending on --conditioning:
      - baseline: every val sample sharing a context yields the SAME input, so
                  deterministic generation collapses to one output for that
                  context. We therefore (a) deduplicate to unique contexts,
                  (b) run diverse beam search to produce K distinct outputs per
                  context, (c) Hungarian-match the K generations to the K refs
                  via ROUGE-L, then re-expand back into val-sample order.
      - topic / bloom: every val sample has its own (context + topic-or-bloom)
                  combination, so the input is unique per sample → standard
                  deterministic beam search, one output per sample.
    """
    client_model.model.eval()
    tokenizer = client_model.tokenizer

    contexts = [s["context"] for s in samples]
    references = [
        f"Question: {s['question']}\nAnswer: {s['answer']}" for s in samples
    ]

    if args.conditioning == "baseline":
        preds = _evaluate_baseline_diverse(
            client_model, samples, references, args, device, use_amp,
        )
    else:
        preds = _evaluate_per_sample(
            client_model, samples, args, device, use_amp,
        )

    client_model.model.train()
    return preds, references, contexts


@torch.no_grad()
def _evaluate_per_sample(
    client_model: ClientModel,
    samples: list,
    args: argparse.Namespace,
    device: torch.device,
    use_amp: bool,
) -> list:
    """One forward pass per sample — input is unique because of conditioning."""
    tokenizer = client_model.tokenizer
    loader = DataLoader(
        QADataset(samples, tokenizer, args.max_input_len, args.max_target_len,
                  conditioning=args.conditioning),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )
    preds: list = []
    for batch in loader:
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            out = client_model.generate(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                num_beams=4, max_new_tokens=args.max_target_len,
                no_repeat_ngram_size=3, early_stopping=True,
            )
        for ids in out:
            preds.append(tokenizer.decode(ids, skip_special_tokens=True))
    return preds


@torch.no_grad()
def _evaluate_baseline_diverse(
    client_model: ClientModel,
    samples: list,
    references: list,
    args: argparse.Namespace,
    device: torch.device,
    use_amp: bool,
) -> list:
    """
    Group val samples by context, run diverse beam search per unique context,
    Hungarian-match generations to that context's references, then re-emit
    predictions in original val-sample order.
    """
    from collections import defaultdict
    import numpy as np
    from rouge_score import rouge_scorer
    from scipy.optimize import linear_sum_assignment

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    tokenizer = client_model.tokenizer

    groups: dict = defaultdict(list)  # context → list of original indices
    for i, s in enumerate(samples):
        groups[s["context"]].append(i)

    K = args.eval_num_return_sequences
    preds: list = [None] * len(samples)

    for ctx, idxs in groups.items():
        sample = samples[idxs[0]]  # any of them — they share context
        prompt = render_prompt(sample, "baseline")
        enc = tokenizer(prompt, max_length=args.max_input_len,
                        truncation=True, padding=False, return_tensors="pt").to(device)

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            out = client_model.model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                num_beams=args.eval_num_beams,
                num_beam_groups=args.eval_num_beam_groups,
                diversity_penalty=args.eval_diversity_penalty,
                num_return_sequences=K,
                max_new_tokens=args.max_target_len,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )
        gens = [tokenizer.decode(ids, skip_special_tokens=True) for ids in out]

        # Hungarian-match generations to this group's references
        local_refs = [references[i] for i in idxs]
        m, n = len(gens), len(local_refs)
        size = max(m, n)
        cost = np.full((size, size), 1.0)
        for i, g in enumerate(gens):
            for j, r in enumerate(local_refs):
                cost[i, j] = 1.0 - scorer.score(r, g)["rougeL"].fmeasure
        row_idx, col_idx = linear_sum_assignment(cost)
        # row=gen, col=ref. Place gens in original sample order.
        for i, j in zip(row_idx, col_idx):
            if i < m and j < n:
                orig_idx = idxs[j]
                preds[orig_idx] = gens[i]
        # Fill any unmatched slot with the first generation
        for j, orig_idx in enumerate(idxs):
            if preds[orig_idx] is None:
                preds[orig_idx] = gens[0] if gens else ""
    return preds


def _run_full_eval(
    client_model: ClientModel,
    samples: list,
    args: argparse.Namespace,
    device: torch.device,
    use_amp: bool,
    out_dir: Path,
    split_name: str,
) -> dict:
    """
    Generate predictions for `samples`, run the comprehensive metric suite, and
    write metrics_<split_name>.json + generated_qas_<split_name>.json into
    `out_dir`. Handles GPU memory:
      - moves the LM to GPU for generation
      - moves it to CPU before loading the heavy metric models
    Returns the full metrics dict.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Make sure the LM is on GPU for autoregressive generation
    if device.type == "cuda":
        client_model.model.to(device)
        gc.collect()
        torch.cuda.empty_cache()

    print(f"  Generating predictions on {len(samples)} {split_name} samples …")
    preds, refs, contexts = _evaluate(client_model, samples, args, device, use_amp)

    # Free the LM from GPU before loading the heavy metric models
    if device.type == "cuda":
        client_model.model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

    quick = compute_all_metrics(preds, refs, device)
    print(
        f"  ROUGE-L={quick['rouge_l']:.3f}  "
        f"BLEU-4={quick['bleu_4']:.3f}  "
        f"BERTScore={quick['bertscore_f1']:.3f}"
    )

    print(f"  Running comprehensive metrics …")
    metrics = compute_comprehensive_metrics(
        generated=preds,
        references=refs,
        contexts=contexts,
        device=device,
        openai_api_key=args.openai_api_key,
        run_heavy=not args.no_heavy,
        blooms_model=args.blooms_model or None,
    )

    metrics_path = out_dir / f"metrics_{split_name}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"  Metrics saved → {metrics_path}")

    qa_records = [
        {
            "context":        s["context"],
            "reference":      f"Question: {s['question']}\nAnswer: {s['answer']}",
            "generated":      p,
            "question_topic": s.get("question_topic", ""),
            "bloom_level":    s.get("bloom_level"),
            "difficulty":     s.get("difficulty", ""),
        }
        for s, p in zip(samples, preds)
    ]
    qa_path = out_dir / f"generated_qas_{split_name}.json"
    qa_path.write_text(json.dumps(qa_records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Generated QAs saved → {qa_path}")
    return metrics


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    set_seeds(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    pin_mem = device.type == "cuda"

    output_dir = Path(args.output_dir)
    # Conditioning is embedded in the path so multiple --conditioning runs with
    # the same --output-dir don't overwrite each other. Final path:
    #   {output_dir}/{conditioning}/client_{id}/fold{N}/
    client_dir = output_dir / args.conditioning / f"client_{args.client_id}" / f"fold{args.fold}"
    ckpt_base  = client_dir / "checkpoints"
    best_dir   = client_dir / "best"
    client_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(str(output_dir))

    # Load fold-specific splits (val only — no test set used here)
    splits_dir    = Path(args.splits_dir)
    train_samples = _load_split(splits_dir, args.client_id, args.fold, "train")
    val_samples   = _load_split(splits_dir, args.client_id, args.fold, "val")

    print(f"\nClient {args.client_id}  Fold {args.fold} — {args.model}")
    print(f"  train={len(train_samples)}  val={len(val_samples)} QA pairs")
    print(f"  Output → {client_dir}")

    # Build model
    client_model = ClientModel(
        model_name=args.model,
        model_family=args.family,
        lora_target_modules=args.targets,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        device=device,
    )

    # DataLoaders
    train_loader = DataLoader(
        QADataset(train_samples, client_model.tokenizer, args.max_input_len,
                  args.max_target_len, conditioning=args.conditioning),
        batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=pin_mem,
    )
    val_loader = DataLoader(
        QADataset(val_samples, client_model.tokenizer, args.max_input_len,
                  args.max_target_len, conditioning=args.conditioning),
        batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=pin_mem,
    )

    # Optimizer & cosine-warmup scheduler
    total_steps  = len(train_loader) * args.num_epochs
    warmup_steps = max(1, int(args.warmup_ratio * total_steps))
    optimizer    = torch.optim.AdamW(client_model.get_lora_params(), lr=args.lr, weight_decay=0.01)
    scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Resume from checkpoint if requested
    loss_history:        list = []   # avg train CE per epoch
    val_loss_history:    list = []   # val CE per epoch (always recorded)
    es_metric_history:   list = []   # early-stop metric per epoch
    best_es_metric:      float = float("-inf") if args.early_stop_metric == "rouge_l" else float("inf")
    higher_is_better:    bool  = (args.early_stop_metric == "rouge_l")
    patience_count:      int   = 0
    start_epoch:         int   = 0

    if args.resume_from_epoch > 0:
        ckpt_dir = ckpt_base / f"epoch_{args.resume_from_epoch}"
        state = _load_checkpoint(client_model, optimizer, scheduler, ckpt_dir)
        start_epoch       = state["epoch"]
        loss_history      = state.get("train_loss_history",
                                       state.get("loss_history", []))
        val_loss_history  = state.get("val_loss_history", [])
        es_metric_history = state.get("es_metric_history", [])
        best_es_metric    = state.get("best_es_metric",
                                       state.get("best_val_loss", best_es_metric))
        patience_count    = state.get("patience_count", 0)
        # Scheduler state is already restored inside _load_checkpoint;
        # do NOT step it manually — that would double-advance the LR curve.

    # Fix one val sample for consistent per-epoch preview
    preview_sample = random.choice(val_samples)

    # Fix the val subsample used by the fast ROUGE-L early-stop metric.
    # Drawn once with the run's seed so the metric is comparable across epochs
    # AND across runs that resume from a checkpoint.
    es_subsample = []
    if args.early_stop_metric == "rouge_l":
        n_es = min(args.fast_eval_samples, len(val_samples))
        es_rng = random.Random(args.seed)
        es_subsample = es_rng.sample(val_samples, n_es)
        print(f"  Early stopping on greedy ROUGE-L over {n_es} fixed val samples.")

    # ── Training loop ─────────────────────────────────────────────────────────
    client_model.model.train()
    stopped_early = False

    for epoch in range(start_epoch, args.num_epochs):
        # Train
        train_losses: list[float] = []
        for batch in train_loader:
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
                loss = client_model.forward(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["labels"].to(device),
                ).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(client_model.get_lora_params(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())

        avg_loss = sum(train_losses) / max(len(train_losses), 1)
        loss_history.append(avg_loss)

        # Validation loss
        client_model.model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for batch in val_loader:
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
                    val_losses.append(
                        client_model.forward(
                            batch["input_ids"].to(device),
                            batch["attention_mask"].to(device),
                            batch["labels"].to(device),
                        ).loss.item()
                    )
        val_loss = sum(val_losses) / max(len(val_losses), 1)
        val_loss_history.append(val_loss)
        client_model.model.train()

        # Compute the early-stop metric
        if args.early_stop_metric == "rouge_l":
            es_metric = _fast_rouge_l(
                client_model, es_subsample, args, device, use_amp,
            )
        else:
            es_metric = val_loss
        es_metric_history.append(es_metric)

        # Early stopping check (direction depends on metric type)
        if higher_is_better:
            improved = es_metric > best_es_metric + args.min_delta
        else:
            improved = es_metric < best_es_metric - args.min_delta
        if improved:
            best_es_metric = es_metric
            patience_count = 0
            _save_best(client_model, best_dir)
            status = "best"
        else:
            patience_count += 1
            status = f"patience {patience_count}/{args.patience}"

        es_label = "rouge_l" if args.early_stop_metric == "rouge_l" else "val_loss"
        print(
            f"  Epoch {epoch + 1:>3}/{args.num_epochs}"
            f" — train={avg_loss:.4f}  val={val_loss:.4f}  "
            f"{es_label}={es_metric:.4f}  [{status}]"
        )

        # Val sample preview
        if args.preview_every > 0 and (epoch + 1) % args.preview_every == 0:
            _preview(client_model, preview_sample, args, device, use_amp, epoch + 1)

        # Periodic checkpoint
        if args.checkpoint_every > 0 and (epoch + 1) % args.checkpoint_every == 0:
            _save_checkpoint(
                client_model, optimizer, scheduler,
                epoch=epoch + 1,
                train_loss_history=loss_history,
                val_loss_history=val_loss_history,
                best_val_loss=best_es_metric,
                patience_count=patience_count,
                ckpt_dir=ckpt_base / f"epoch_{epoch + 1}",
                es_metric_history=es_metric_history,
                best_es_metric=best_es_metric,
            )

        # Early stopping exit
        if patience_count >= args.patience:
            print(f"\n  Early stopping at epoch {epoch + 1} (patience={args.patience})")
            stopped_early = True
            break

    # Save histories
    with open(client_dir / "loss_history.json", "w") as f:
        json.dump({
            "train":           loss_history,
            "val":             val_loss_history,
            "es_metric":       es_metric_history,
            "es_metric_name":  args.early_stop_metric,
        }, f, indent=2)

    # Save final model weights (state at the last completed epoch)
    final_dir = client_dir / "final"
    client_model.model.save_pretrained(str(final_dir / "lora_model"))
    print(f"\nFinal model saved → {final_dir}")

    # Free DataLoaders before evaluation
    del train_loader, val_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Make sure NLTK sentence tokenizer is available for downstream metrics
    if not args.no_heavy:
        _ensure_nltk_punkt()

    # Optionally load the global test set once
    global_test_samples = None
    if args.global_test_file:
        global_test_path = Path(args.global_test_file)
        if global_test_path.exists():
            global_test_samples = json.loads(global_test_path.read_text(encoding="utf-8"))
        else:
            print(f"\nWARNING: --global-test-file not found: {global_test_path} — skipping.")

    results_dir = client_dir / "results"

    # ── Eval FINAL checkpoint (model already has final-epoch weights) ────────
    print(f"\n{'=' * 60}\n  Evaluating FINAL checkpoint\n{'=' * 60}")
    _run_full_eval(
        client_model, val_samples, args, device, use_amp,
        out_dir=results_dir / "final", split_name="val",
    )
    if global_test_samples is not None:
        _run_full_eval(
            client_model, global_test_samples, args, device, use_amp,
            out_dir=results_dir / "final", split_name="global_test",
        )

    # ── Eval BEST checkpoint ─────────────────────────────────────────────────
    print(f"\n{'=' * 60}\n  Loading BEST checkpoint\n{'=' * 60}")
    # Move the LM back to GPU before swapping LoRA weights and generating.
    if device.type == "cuda":
        client_model.model.to(device)
        gc.collect()
        torch.cuda.empty_cache()
    _load_best(client_model, best_dir)

    print(f"\n{'=' * 60}\n  Evaluating BEST checkpoint\n{'=' * 60}")
    _run_full_eval(
        client_model, val_samples, args, device, use_amp,
        out_dir=results_dir / "best", split_name="val",
    )
    if global_test_samples is not None:
        _run_full_eval(
            client_model, global_test_samples, args, device, use_amp,
            out_dir=results_dir / "best", split_name="global_test",
        )

    print(f"\nAll outputs saved to {client_dir}")
    print(f"  Per-checkpoint metrics → {results_dir}/{{best,final}}/metrics_*.json")


if __name__ == "__main__":
    main()
