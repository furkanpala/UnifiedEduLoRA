"""
Post-hoc evaluation for a saved federated training run.

Loads a fed snapshot directory (best or final) without re-training,
re-registers the FiLM hooks via the saved GNN, and evaluates each
client on any subset of: per-client val / per-client local test /
global test.

Background. train_federated.py only writes per-client metrics on the
local test set (and global test if a global_test.json exists). The
indiv-vs-fed comparison (experiments/14_compare_indiv_vs_fed.py) wants
all three split families. This script back-fills whatever's missing
without touching the training pipeline.

Output layout (mirrors what train_federated.py already writes):
    {fed_output_dir}/results/{best,final}/client_{cid}_metrics_{val,test,global_test}.json

Usage (Colab):
    python experiments/13_eval_federated_post_hoc.py \\
        --fed-output-dir /content/drive/MyDrive/unifiedfl_fp_federated_experiment/unbalanced \\
        --splits-dir     /content/drive/MyDrive/unifiedfl_fp_individual_experiment/unbalanced/splits \\
        --fold           1 \\
        --conditioning   topic \\
        --eval-on        val test global_test \\
        --no-heavy

Note: --no-heavy is the user's stated default for fed evaluations.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "unifiedfl"))
sys.path.insert(0, str(REPO / "experiments"))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

from config.config import (
    Config, ClientConfig, LoRAConfig, GNNConfig, FiLMConfig,
)
from evaluation.evaluator import Evaluator
from evaluation.metrics import compute_comprehensive_metrics
from train_federated import (
    _build_clients,
    _ensure_nltk_punkt,
    _load_split,
    _load_state_from_dir,
    _parse_client_spec,
)
from utils.reproducibility import set_seeds

from _common import default_key_search_paths, load_openai_key


# Matches the spec list hard-coded in 08_run_federated_training.py
DEFAULT_CLIENT_SPECS = [
    "0:facebook/bart-base:bart:q_proj,v_proj:768",
    "1:google/flan-t5-base:t5:q,v:768",
    "2:allenai/led-base-16384:led:q_proj,v_proj:768",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--fed-output-dir", required=True,
                   help="Federated run dir (contains fed_final/, "
                        "fed_final_round/, fed_checkpoints/).")
    p.add_argument("--splits-dir", required=True)
    p.add_argument("--fold", type=int, required=True, choices=[1, 2, 3],
                   help="Which fold's splits to load. Must match the fold "
                        "the fed run was trained on.")
    p.add_argument("--conditioning", required=True,
                   choices=["baseline", "topic", "bloom"])

    p.add_argument("--eval-on", nargs="+",
                   choices=["val", "test", "global_test"],
                   default=["val", "test", "global_test"],
                   help="Which split(s) to evaluate. Default: all three.")
    p.add_argument("--ckpts", nargs="+",
                   choices=["best", "final"], default=["best", "final"],
                   help="Which snapshot(s) to evaluate. Default: both.")

    p.add_argument("--global-test-file", default=None,
                   help="Path to a global_test.json. Default: <splits-dir>/global_test.json. "
                        "Only used when 'global_test' is in --eval-on.")
    p.add_argument("--client", action="append", default=None,
                   metavar="ID:MODEL:FAMILY:TARGETS:D_MODEL",
                   help="Override client specs (default: 0:bart, 1:flan-t5-base, "
                        "2:led-base-16384). Repeat for each client.")

    # Hyperparams that must match the training run (the saved checkpoint
    # only carries weights, so r/alpha/dropout etc. need to be re-supplied).
    p.add_argument("--lora-r",       type=int,   default=16)
    p.add_argument("--lora-alpha",   type=int,   default=32)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--gnn-hidden",   type=int,   default=64)
    p.add_argument("--gnn-heads",    type=int,   default=4)
    p.add_argument("--gnn-layers",   type=int,   default=3)
    p.add_argument("--gnn-dropout",  type=float, default=0.1)
    p.add_argument("--film-hidden",  type=int,   default=128)
    p.add_argument("--film-alpha",   type=float, default=0.0)
    p.add_argument("--max-input-len",  type=int, default=512)
    p.add_argument("--max-target-len", type=int, default=128)
    p.add_argument("--batch-size",     type=int, default=4)

    p.add_argument("--no-heavy", action="store_true",
                   help="Skip UnifiedQA + DeBERTa NLI (RTC/Faithfulness/QAFE/RQUGE).")
    p.add_argument("--blooms-model",
                   default="cip29/bert-blooms-taxonomy-classifier",
                   help="HF model ID for the local Bloom's classifier "
                        "(set to '' to skip).")
    p.add_argument("--openai-api-key", default=None)
    p.add_argument("--drive-dir",      default=None)

    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--force",  action="store_true",
                   help="Re-run a (ckpt, split) cell even if its metrics file "
                        "is already on disk. Default: skip existing.")
    return p.parse_args()


def _resolve_openai_key(args: argparse.Namespace) -> str | None:
    """Mirror 11/03: if comprehensive (no-heavy off), find a key."""
    if args.no_heavy:
        return None
    if args.openai_api_key and args.openai_api_key.startswith("sk-"):
        os.environ["OPENAI_API_KEY"] = args.openai_api_key
        return args.openai_api_key
    api_key, source = load_openai_key(
        *default_key_search_paths(str(REPO), args.drive_dir)
    )
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        print(f"OpenAI key from {source}")
    else:
        print("WARNING: no OpenAI key — Answer Relevancy / LLM judges will be skipped.")
    return api_key


def _eval_one_client_on_splits(
    client, evaluator, results_dir: Path, splits_to_run: list,
    args, device, openai_api_key: str | None,
) -> None:
    """Evaluate `client` on each (name, samples) tuple in splits_to_run."""
    cid = client.client_id
    evaluator._activate_hooks(client)
    try:
        for split_name, samples in splits_to_run:
            out_path = results_dir / f"client_{cid}_metrics_{split_name}.json"
            if out_path.exists() and not args.force:
                print(f"    [skip] {split_name} — {out_path.name} already exists")
                continue
            if not samples:
                print(f"    [skip] {split_name} — no samples")
                continue

            print(f"    [{split_name}] generating on {len(samples)} samples …")
            preds, refs, contexts = evaluator.collect_predictions(client, samples)

            # Free LM before loading heavy metric models, mirrors train_client.
            if device.type == "cuda":
                client.client_model.model.to("cpu")
                gc.collect()
                torch.cuda.empty_cache()

            metrics = compute_comprehensive_metrics(
                generated=preds, references=refs, contexts=contexts,
                device=device,
                openai_api_key=openai_api_key,
                run_heavy=not args.no_heavy,
                blooms_model=args.blooms_model or None,
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(metrics, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"    [{split_name}] saved → {out_path}")

            if device.type == "cuda":
                client.client_model.model.to(device)
    finally:
        evaluator._deactivate_hooks(client)


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)

    api_key = _resolve_openai_key(args)
    args.openai_api_key = api_key

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    fed_dir   = Path(args.fed_output_dir)
    splits_dir = Path(args.splits_dir)

    if not fed_dir.exists():
        raise FileNotFoundError(f"--fed-output-dir not found: {fed_dir}")

    # Optional global test
    global_test = []
    if "global_test" in args.eval_on:
        gt_path = Path(args.global_test_file or splits_dir / "global_test.json")
        if not gt_path.exists():
            raise FileNotFoundError(
                f"global_test.json not found at {gt_path}. "
                f"Build it first: python experiments/12_make_global_test.py "
                f"--splits-dir {splits_dir}"
            )
        global_test = json.loads(gt_path.read_text(encoding="utf-8"))
        print(f"Global test: {len(global_test)} samples ← {gt_path}")

    # Parse client specs (default = same as 08)
    spec_strs = args.client if args.client else DEFAULT_CLIENT_SPECS
    client_cfgs = [_parse_client_spec(s) for s in spec_strs]
    client_cfgs.sort(key=lambda c: c.client_id)

    # Load fold-specific train/val + per-client test
    client_splits: dict = {}
    for c in client_cfgs:
        cid = c.client_id
        client_splits[cid] = {
            "train": _load_split(splits_dir, cid, "train", fold=args.fold),
            "val":   _load_split(splits_dir, cid, "val",   fold=args.fold),
            "test":  _load_split(splits_dir, cid, "test"),
        }

    # Build a Config matching the training-time hyperparams. The saved
    # checkpoint only stores weights; we have to recreate the model
    # architecture (LoRA r/alpha, GNN dims, FiLM dims) the same way it was
    # built during training. If any of these differ from training, weights
    # won't load cleanly.
    cfg = Config(
        seed=args.seed, num_rounds=33, local_epochs=3,
        batch_size=args.batch_size,
        max_input_len=args.max_input_len, max_target_len=args.max_target_len,
        lr_lora=3e-4, lr_gnn=1e-3, lr_film=1e-3,
        warmup_ratio=0.1, grad_clip=1.0, eval_every_n=5,
        device=args.device, output_dir=str(fed_dir),
        lora=LoRAConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                        lora_dropout=args.lora_dropout),
        gnn=GNNConfig(hidden=args.gnn_hidden, heads=args.gnn_heads,
                      layers=args.gnn_layers, dropout=args.gnn_dropout),
        film=FiLMConfig(hidden=args.film_hidden, alpha_init=args.film_alpha),
        clients=client_cfgs,
        conditioning=args.conditioning,
    )

    print("\nBuilding clients (same architecture used during training) …")
    clients = _build_clients(cfg, client_splits, device)
    evaluator = Evaluator(clients, cfg, device)

    if not args.no_heavy:
        _ensure_nltk_punkt()

    for ckpt_label in args.ckpts:
        snap_dir = fed_dir / ("fed_final" if ckpt_label == "best" else "fed_final_round")
        if not snap_dir.exists():
            print(f"\n[skip] {ckpt_label}: {snap_dir} missing")
            continue

        print(f"\n{'=' * 70}")
        print(f"  Loading {ckpt_label.upper()} from {snap_dir}")
        print(f"{'=' * 70}")
        _load_state_from_dir(clients, snap_dir, device)

        results_dir = fed_dir / "results" / ckpt_label

        for client in clients:
            cid = client.client_id
            print(f"\n  Client {cid} ({client.client_model.model_name})")
            splits_for_client = []
            if "val" in args.eval_on:
                splits_for_client.append(("val", client.val_samples))
            if "test" in args.eval_on:
                splits_for_client.append(("test", client.test_samples))
            if "global_test" in args.eval_on:
                splits_for_client.append(("global_test", global_test))
            _eval_one_client_on_splits(
                client, evaluator, results_dir, splits_for_client,
                args, device, api_key,
            )

    print(f"\nAll done. Results under {fed_dir / 'results'}/{{{','.join(args.ckpts)}}}/")


if __name__ == "__main__":
    main()
