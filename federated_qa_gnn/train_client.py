"""
Train a single client with LoRA only (no GNN, no FiLM) — baseline individual training.
Supports checkpointing and resuming, designed for Colab + Drive.

Usage (Colab):
    # Mount Drive first:
    #   from google.colab import drive; drive.mount('/content/drive')

    python train_client.py \
        --client-id 0 \
        --model google/flan-t5-small \
        --family t5 \
        --targets q v \
        --d-model 512 \
        --splits-dir /content/drive/MyDrive/federated_qa_gnn/outputs/splits \
        --output-dir /content/drive/MyDrive/federated_qa_gnn/outputs \
        --num-epochs 60

    # Resume from checkpoint:
    python train_client.py ... --resume-from-epoch 30
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent))

from data.dataset import QADataset
from evaluation.metrics import compute_all_metrics
from models.client_model import ClientModel
from utils.logging_utils import setup_logging


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a single client with LoRA")

    # Client / model
    p.add_argument("--client-id", type=int, required=True)
    p.add_argument("--model",     required=True, help="HuggingFace model ID")
    p.add_argument("--family",    required=True, choices=["t5", "bart", "led"])
    p.add_argument("--targets",   nargs="+", required=True,
                   help="LoRA target module names, e.g. --targets q v")
    p.add_argument("--d-model",   type=int, required=True,
                   help="Model hidden dim (512 for T5-small, 768 for BART/LED)")

    # Data
    p.add_argument("--splits-dir", default="outputs/splits",
                   help="Directory containing split JSON files from split.py")

    # Training
    p.add_argument("--num-epochs",     type=int,   default=60)
    p.add_argument("--batch-size",     type=int,   default=4)
    p.add_argument("--lr",             type=float, default=3e-4)
    p.add_argument("--warmup-ratio",   type=float, default=0.1)
    p.add_argument("--grad-clip",      type=float, default=1.0)
    p.add_argument("--max-input-len",  type=int,   default=512)
    p.add_argument("--max-target-len", type=int,   default=128)

    # LoRA
    p.add_argument("--lora-r",       type=int,   default=16)
    p.add_argument("--lora-alpha",   type=int,   default=32)
    p.add_argument("--lora-dropout", type=float, default=0.1)

    # Checkpointing / resuming
    p.add_argument("--checkpoint-every", type=int, default=10,
                   help="Save checkpoint every N epochs (0 = only at end)")
    p.add_argument("--resume-from-epoch", type=int, default=0,
                   help="Resume from this epoch's checkpoint (0 = start fresh)")

    # Output / preview
    p.add_argument("--output-dir",    default="outputs/",
                   help="Root output dir (can be a Drive path on Colab)")
    p.add_argument("--preview-every", type=int, default=5,
                   help="Print a sample generation every N epochs (0 = off)")
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--device", default="cuda")

    return p.parse_args()


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_split(splits_dir: Path, client_id: int, name: str) -> list:
    path = splits_dir / f"client_{client_id}_{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found: {path}\n"
            "Run split.py first to generate splits."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_checkpoint(
    client_model: ClientModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    loss_history: list,
    ckpt_dir: Path,
) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    client_model.model.save_pretrained(str(ckpt_dir / "lora_model"))
    torch.save(
        {
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "loss_history": loss_history,
        },
        ckpt_dir / "training_state.pt",
    )
    print(f"  [checkpoint] saved → {ckpt_dir}")


def _load_checkpoint(
    client_model: ClientModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    ckpt_dir: Path,
) -> tuple[int, list]:
    from peft import PeftModel
    lora_dir = ckpt_dir / "lora_model"
    if not lora_dir.exists():
        raise FileNotFoundError(f"Checkpoint not found: {lora_dir}")
    client_model.model.load_adapter(str(lora_dir), adapter_name="default")
    state = torch.load(ckpt_dir / "training_state.pt", map_location="cpu")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    print(f"  [checkpoint] resumed from epoch {state['epoch']} ← {ckpt_dir}")
    return state["epoch"], state["loss_history"]


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
    print(f"\n  ┌─ Epoch {epoch} preview {'─' * 35}")
    print(f"  │ CONTEXT  : {ctx}")
    print(f"  │ GENERATED: {generated}")
    print(f"  └{'─' * 50}")
    client_model.model.train()


@torch.no_grad()
def _evaluate(
    client_model: ClientModel,
    samples: list,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list, list]:
    client_model.model.eval()
    loader = DataLoader(
        QADataset(samples, client_model.tokenizer, args.max_input_len, args.max_target_len),
        batch_size=args.batch_size, shuffle=False, num_workers=2,
    )
    preds, refs = [], []
    for batch in loader:
        out = client_model.generate(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            num_beams=4, max_new_tokens=args.max_target_len,
            no_repeat_ngram_size=3, early_stopping=True,
        )
        for ids in out:
            preds.append(client_model.tokenizer.decode(ids, skip_special_tokens=True))
        for lab in batch["labels"]:
            lab = lab.masked_fill(lab == -100, client_model.tokenizer.pad_token_id)
            refs.append(client_model.tokenizer.decode(lab, skip_special_tokens=True))
    return preds, refs


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    set_seeds(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    pin_mem = device.type == "cuda"

    output_dir = Path(args.output_dir)
    client_dir = output_dir / f"client_{args.client_id}"
    ckpt_base  = client_dir / "checkpoints"
    client_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(str(output_dir))

    # Load splits
    splits_dir = Path(args.splits_dir)
    train_samples = _load_split(splits_dir, args.client_id, "train")
    val_samples   = _load_split(splits_dir, args.client_id, "val")
    test_samples  = _load_split(splits_dir, args.client_id, "test")
    global_test_path = splits_dir / "global_test.json"
    global_test = json.loads(global_test_path.read_text()) if global_test_path.exists() else []

    print(f"\nClient {args.client_id} — {args.model}")
    print(f"  train={len(train_samples)}  val={len(val_samples)}  test={len(test_samples)} QA pairs")
    if global_test:
        print(f"  global_test={len(global_test)} QA pairs")

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
        QADataset(train_samples, client_model.tokenizer, args.max_input_len, args.max_target_len),
        batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=pin_mem,
    )
    val_loader = DataLoader(
        QADataset(val_samples, client_model.tokenizer, args.max_input_len, args.max_target_len),
        batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=pin_mem,
    )

    # Optimizer & scheduler (built over full remaining steps for correct LR curve)
    remaining_epochs = args.num_epochs - args.resume_from_epoch
    total_steps   = len(train_loader) * args.num_epochs
    warmup_steps  = max(1, int(args.warmup_ratio * total_steps))
    optimizer     = torch.optim.AdamW(client_model.get_lora_params(), lr=args.lr, weight_decay=0.01)
    scheduler     = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Resume from checkpoint if requested
    loss_history: list = []
    start_epoch = 0
    if args.resume_from_epoch > 0:
        ckpt_dir = ckpt_base / f"epoch_{args.resume_from_epoch}"
        start_epoch, loss_history = _load_checkpoint(
            client_model, optimizer, scheduler, ckpt_dir
        )
        # Fast-forward scheduler to the correct step
        done_steps = start_epoch * len(train_loader)
        for _ in range(done_steps):
            scheduler.step()

    # Training loop
    client_model.model.train()
    for epoch in range(start_epoch, args.num_epochs):
        losses = []
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
            losses.append(loss.item())

        avg_loss = sum(losses) / max(len(losses), 1)
        loss_history.append(avg_loss)

        # Validation loss
        client_model.model.eval()
        val_losses = []
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
        client_model.model.train()

        print(f"  Epoch {epoch + 1:>3}/{args.num_epochs} — train={avg_loss:.4f}  val={val_loss:.4f}")

        if args.preview_every > 0 and (epoch + 1) % args.preview_every == 0:
            _preview(client_model, train_samples[0], args, device, use_amp, epoch + 1)

        # Checkpoint
        if args.checkpoint_every > 0 and (epoch + 1) % args.checkpoint_every == 0:
            _save_checkpoint(
                client_model, optimizer, scheduler,
                epoch + 1, loss_history,
                ckpt_base / f"epoch_{epoch + 1}",
            )

    # Save loss history
    with open(client_dir / "loss_history.json", "w") as f:
        json.dump(loss_history, f, indent=2)

    # Final model save
    final_dir = client_dir / "final"
    client_model.model.save_pretrained(str(final_dir / "lora_model"))
    print(f"\nFinal model saved → {final_dir}")

    # Evaluation
    print("\nEvaluating on local test set …")
    preds, refs = _evaluate(client_model, test_samples, args, device)
    local_metrics = compute_all_metrics(preds, refs, device)
    print(
        f"  Local  — ROUGE-L={local_metrics['rouge_l']:.3f}  "
        f"BLEU-4={local_metrics['bleu_4']:.3f}  "
        f"BERTScore={local_metrics['bertscore_f1']:.3f}"
    )

    if global_test:
        print("Evaluating on global test set …")
        preds, refs = _evaluate(client_model, global_test, args, device)
        global_metrics = compute_all_metrics(preds, refs, device)
        print(
            f"  Global — ROUGE-L={global_metrics['rouge_l']:.3f}  "
            f"BLEU-4={global_metrics['bleu_4']:.3f}  "
            f"BERTScore={global_metrics['bertscore_f1']:.3f}"
        )
        with open(client_dir / "metrics.json", "w") as f:
            json.dump({"local": local_metrics, "global": global_metrics}, f, indent=2)
    else:
        with open(client_dir / "metrics.json", "w") as f:
            json.dump({"local": local_metrics}, f, indent=2)


if __name__ == "__main__":
    main()
