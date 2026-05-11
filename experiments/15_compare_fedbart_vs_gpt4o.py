"""
Compare federated-BART (UnifiedEduLoRA, conditioning=topic, client_0) against
GPT-4o on QA generation from the global mixed test set.

Both models see the SAME input — render_prompt(sample, "topic") — and produce
one "Question: ...\\nAnswer: ..." pair per (context, topic) row in the global
mixed test set. Reference-based lightweight metrics are computed against the
gold QA. GPT-4o receives the verbatim BART prompt followed by a single
format-directive sentence so its output is parseable into the same shape.

Pipeline:
    1. Load global_test.json; optionally stratified-sample N unique contexts.
    2. For each --folds k: build the 3-client architecture, load
       fed_final/ snapshot, activate FiLM hooks on client_0, run
       collect_predictions over all sampled rows. Save predictions.
    3. GPT-4o pass — same input rows, same prompt. Save predictions.
    4. Compute ROUGE-L / BLEU-4 / BERTScore-F1 / Bloom-cls EVS per
       (predictions, gold-refs) pair, per-client and overall. Average
       fed-BART numbers across folds (mean ± std).
    5. Emit per-run metrics JSON + a human-readable summary table.

Output layout (all under --out-dir):
    sample_ids.json                        frozen list of (context, topic) row idxs
    generations_fedbart_fold{1,2,3}.json   {"sample_idx": ..., "pred": "Question: ...\\nAnswer: ..."}
    generations_gpt4o.json                 same shape
    metrics_per_run.json                   raw numbers per (model, fold, client-slice)
    comparison_summary.txt                 human-readable table

Usage (pilot, BART fold1 only, no GPT-4o):
    python experiments/15_compare_fedbart_vs_gpt4o.py \\
        --limit-contexts 10 --folds 1 --skip-gpt4o

Usage (full run, all 3 folds + GPT-4o on all contexts):
    python experiments/15_compare_fedbart_vs_gpt4o.py
"""

from __future__ import annotations

import argparse
import collections
import gc
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "unifiedfl"))
sys.path.insert(0, str(REPO / "experiments"))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

from config.config import (
    Config, ClientConfig, LoRAConfig, GNNConfig, FiLMConfig,
)
from data.dataset import PROMPT_TEMPLATES, render_prompt
from evaluation.evaluator import Evaluator
from evaluation.metrics import (
    compute_blooms_classifier,
    compute_bertscore,
    compute_bleu4,
    compute_rouge_l,
)
from train_federated import (
    _build_clients,
    _load_split,
    _load_state_from_dir,
    _parse_client_spec,
)
from utils.reproducibility import set_seeds

from _common import default_key_search_paths, load_openai_key


DEFAULT_CLIENT_SPECS = [
    "0:facebook/bart-base:bart:q_proj,v_proj:768",
    "1:google/flan-t5-base:t5:q,v:768",
    "2:allenai/led-base-16384:led:q_proj,v_proj:768",
]

# Single trailing directive appended to the BART training prompt so GPT-4o
# returns parseable Question/Answer pairs. BART learned this format from
# TARGET_TEMPLATE during fine-tuning; GPT-4o needs to be told once.
GPT4O_FORMAT_DIRECTIVE = (
    "\n\nRespond with exactly:\nQuestion: <one question>\nAnswer: <one answer>"
)


# ─────────────────────────────────────────────────────────────────────────────
# Arg parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--global-test-file",
                   default=str(REPO / "data/splits/global_test.json"))
    p.add_argument("--fed-base-dir",
                   default=str(REPO / "experiment_outputs/unifiedfl_fp_federated_experiment"),
                   help="Each fold lives at {base}_fold{k}/fed_final/")
    p.add_argument("--splits-dir", default=str(REPO / "data/splits"))
    p.add_argument("--out-dir",
                   default=str(REPO / "experiment_outputs/slm_vs_llm"))
    p.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--conditioning",
                   choices=["baseline", "topic", "bloom"], default="topic")
    p.add_argument("--limit-contexts", type=int, default=0,
                   help="If >0, stratified-sample N unique contexts. 0 = use all 179.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")

    p.add_argument("--skip-bart",  action="store_true")
    p.add_argument("--skip-gpt4o", action="store_true")
    p.add_argument("--force-regen", action="store_true",
                   help="Re-run BART/GPT-4o even if generation files already exist.")

    # GPT-4o
    p.add_argument("--gpt4o-model", default="gpt-4o")
    p.add_argument("--gpt4o-temperature", type=float, default=0.0)
    p.add_argument("--gpt4o-max-tokens", type=int, default=160,
                   help="max_completion_tokens for GPT-4o — Q+A fits in ~80-120 tokens.")
    p.add_argument("--openai-api-key-file", default=str(REPO / "openai_api_key"))

    # Bloom classifier
    p.add_argument("--blooms-model",
                   default="cip29/bert-blooms-taxonomy-classifier")

    # Federated-architecture hyperparams (must match training)
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
    p.add_argument("--client", action="append", default=None,
                   metavar="ID:MODEL:FAMILY:TARGETS:D_MODEL")
    p.add_argument("--client-id-bart", type=int, default=0,
                   help="Which federated client to use as the SLM under "
                        "test (0=BART, 1=Flan-T5, 2=LED). Name kept "
                        "'bart' for backward compat; the other 2 clients "
                        "are still built so FiLM hooks match training topology.")
    p.add_argument("--model-label", default=None,
                   help="Pretty label for this client in summary tables. "
                        "Default: 'fed-BART' / 'fed-T5' / 'fed-LED' for "
                        "client_id 0 / 1 / 2.")
    p.add_argument("--generation-stem", default=None,
                   help="Filename stem for per-fold generation files. "
                        "Default: 'fedbart' / 'fedt5' / 'fedled' for "
                        "client_id 0 / 1 / 2.")
    p.add_argument("--gpt4o-cache-path", default=None,
                   help="Optional explicit path to a cached GPT-4o "
                        "generations JSON (same 916 rows, same prompt). "
                        "If set and n_rows matches, GPT-4o stage is "
                        "skipped and predictions are loaded from this "
                        "file. Default behavior unchanged.")
    return p.parse_args()


# Defaults derived from client-id when --model-label / --generation-stem unset.
_LABEL_DEFAULTS = {0: ("fed-BART", "fedbart"),
                   1: ("fed-T5",   "fedt5"),
                   2: ("fed-LED",  "fedled")}


# ─────────────────────────────────────────────────────────────────────────────
# Sampling
# ─────────────────────────────────────────────────────────────────────────────

def select_samples(global_test: List[Dict], limit_contexts: int, seed: int
                   ) -> Tuple[List[Dict], List[int]]:
    """Either return the full global test set, or stratified-sample N
    unique contexts (with all rows belonging to each chosen context).

    Strata: client_0 (MIT), client_1 (Stanford), client_2 (ML papers). When
    N exceeds a stratum's unique-context count, that stratum is consumed
    entirely and the remainder is drawn from client_2 (the largest).

    Returns (samples, original_row_indices).
    """
    if limit_contexts <= 0:
        return global_test, list(range(len(global_test)))

    rng = random.Random(seed)
    by_client: Dict[int, List[str]] = collections.defaultdict(list)
    for s in global_test:
        ctx = s["context"]
        if ctx not in by_client[s["_source_client"]]:
            by_client[s["_source_client"]].append(ctx)

    avail = {c: len(v) for c, v in by_client.items()}
    # Aim: even split across the 3 clients; overflow flows to client_2.
    per_client = limit_contexts // 3
    chosen: List[str] = []
    for cid in (0, 1, 2):
        ctxs = by_client[cid][:]
        rng.shuffle(ctxs)
        n = min(per_client, avail[cid])
        chosen.extend(ctxs[:n])
    deficit = limit_contexts - len(chosen)
    if deficit > 0:
        leftover_c2 = [c for c in by_client[2] if c not in chosen]
        rng.shuffle(leftover_c2)
        chosen.extend(leftover_c2[:deficit])

    chosen_set = set(chosen)
    rows: List[Dict] = []
    idxs: List[int] = []
    for i, s in enumerate(global_test):
        if s["context"] in chosen_set:
            rows.append(s); idxs.append(i)
    return rows, idxs


# ─────────────────────────────────────────────────────────────────────────────
# Fed-BART generation per fold
# ─────────────────────────────────────────────────────────────────────────────

def run_fedbart_fold(
    fold: int, samples: List[Dict], args, device: torch.device,
) -> List[str]:
    """Build clients for fold, load fed_final/ snapshot, activate FiLM hooks
    on the BART client, run collect_predictions on `samples` and return
    decoded prediction strings (one per sample, in input order)."""
    fed_dir = Path(f"{args.fed_base_dir}_fold{fold}")
    snap_dir = fed_dir / "fed_final"
    if not snap_dir.exists():
        raise FileNotFoundError(f"missing fed_final snapshot at {snap_dir}")

    spec_strs = args.client if args.client else DEFAULT_CLIENT_SPECS
    client_cfgs = [_parse_client_spec(s) for s in spec_strs]
    client_cfgs.sort(key=lambda c: c.client_id)

    # _build_clients needs val/test splits to construct each client (used by
    # internal helpers). We don't evaluate on them — we only ever call
    # collect_predictions on our own sample list — so the inner content
    # doesn't matter for correctness, only that the splits exist.
    client_splits: Dict[int, Dict[str, list]] = {}
    splits_dir = Path(args.splits_dir)
    for c in client_cfgs:
        cid = c.client_id
        client_splits[cid] = {
            "train": _load_split(splits_dir, cid, "train", fold=fold),
            "val":   _load_split(splits_dir, cid, "val",   fold=fold),
            "test":  _load_split(splits_dir, cid, "test"),
        }

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

    print(f"  [fold {fold}] building clients …")
    clients = _build_clients(cfg, client_splits, device)
    evaluator = Evaluator(clients, cfg, device)

    print(f"  [fold {fold}] loading {snap_dir.name} …")
    _load_state_from_dir(clients, snap_dir, device)

    bart_client = next(c for c in clients if c.client_id == args.client_id_bart)

    print(f"  [fold {fold}] generating on {len(samples)} samples …")
    evaluator._activate_hooks(bart_client)
    try:
        t0 = time.time()
        preds, _refs, _ctxs = evaluator.collect_predictions(bart_client, samples)
        dt = time.time() - t0
        print(f"  [fold {fold}] done in {dt:.1f}s "
              f"({len(samples)/max(dt,1e-9):.1f} samples/s)")
    finally:
        evaluator._deactivate_hooks(bart_client)

    # Free GPU memory before next fold
    del clients, evaluator
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return preds


# ─────────────────────────────────────────────────────────────────────────────
# GPT-4o generation
# ─────────────────────────────────────────────────────────────────────────────

def run_gpt4o(samples: List[Dict], api_key: str, args) -> List[str]:
    """Generate one Q/A pair per sample using the identical BART prompt
    plus a one-line format directive. Returns predictions in input order
    formatted as "Question: ...\\nAnswer: ..." (TARGET_TEMPLATE)."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    preds: List[str] = []
    bad = 0
    for i, s in enumerate(samples):
        prompt = render_prompt(s, args.conditioning) + GPT4O_FORMAT_DIRECTIVE
        try:
            resp = client.chat.completions.create(
                model=args.gpt4o_model,
                temperature=args.gpt4o_temperature,
                max_tokens=args.gpt4o_max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            text = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"    [gpt-4o] sample {i}: API error {e!r} — leaving blank")
            text = ""
            bad += 1

        preds.append(_normalize_qa(text))

        if (i + 1) % 25 == 0 or (i + 1) == len(samples):
            print(f"    [gpt-4o] {i+1}/{len(samples)} done "
                  f"({bad} errors so far)")

    return preds


def _normalize_qa(text: str) -> str:
    """Coerce GPT-4o output to "Question: ...\\nAnswer: ..." shape.

    The format directive in the prompt usually elicits exactly that shape,
    but a small minority of responses come back with leading bullets, code
    fences, or "Q:"/"A:" abbreviations. Best-effort normalization here so
    downstream parsers (_parse_qa, ROUGE/BLEU on the joined string) see a
    consistent format."""
    if not text:
        return ""
    t = text.strip().strip("`").strip()
    # Common alt forms
    for pre, after in (
        ("Q:", "Question:"), ("Q.", "Question:"), ("A:", "Answer:"), ("A.", "Answer:"),
        ("**Question:**", "Question:"), ("**Answer:**", "Answer:"),
    ):
        t = t.replace(pre, after)
    return t


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def gold_reference(sample: Dict) -> str:
    return f"Question: {sample['question']}\nAnswer: {sample['answer']}"


def compute_lightweight(
    predictions: List[str], references: List[str],
    device: torch.device, blooms_model: str,
) -> Dict[str, float]:
    """ROUGE-L, BLEU-4, BERTScore-F1, Bloom-cls EVS. Returns floats only."""
    # Filter out empty predictions paired-wise so they don't drag scores to 0
    # silently; we still report n_failed.
    keep = [(p, r) for p, r in zip(predictions, references) if p]
    n_failed = len(predictions) - len(keep)
    if not keep:
        return {"rouge_l": 0.0, "bleu_4": 0.0, "bertscore_f1": 0.0,
                "blooms_cls_evs_mean": 0.0, "n": 0, "n_failed": n_failed}
    pp, rr = zip(*keep)
    pp, rr = list(pp), list(rr)
    out: Dict[str, float] = {
        "rouge_l":      compute_rouge_l(pp, rr),
        "bleu_4":       compute_bleu4(pp, rr),
        "bertscore_f1": compute_bertscore(pp, rr, device),
    }
    if blooms_model:
        bloom = compute_blooms_classifier(pp, device, model_name=blooms_model)
        out["blooms_cls_evs_mean"] = bloom["evs_mean"]
    else:
        out["blooms_cls_evs_mean"] = float("nan")
    out["n"] = len(pp)
    out["n_failed"] = n_failed
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_summary(
    samples: List[Dict],
    fedbart_preds_by_fold: Dict[int, List[str]],
    gpt4o_preds: List[str] | None,
    device: torch.device, blooms_model: str,
) -> Dict[str, dict]:
    """Compute metrics for each model, per source-client slice and overall.
    Fed-BART numbers are first computed per fold then averaged."""
    references = [gold_reference(s) for s in samples]
    source_clients = [s["_source_client"] for s in samples]

    def slice_idxs(filter_cid: int | None):
        if filter_cid is None:
            return list(range(len(samples)))
        return [i for i, c in enumerate(source_clients) if c == filter_cid]

    out: Dict[str, dict] = {}

    # ── Fed-BART per fold ─────────────────────────────────────────────────
    for fold, preds in fedbart_preds_by_fold.items():
        per_fold: Dict[str, dict] = {}
        for cid in (0, 1, 2, None):
            idxs = slice_idxs(cid)
            if not idxs:
                continue
            p = [preds[i] for i in idxs]
            r = [references[i] for i in idxs]
            key = "overall" if cid is None else f"client_{cid}"
            per_fold[key] = compute_lightweight(p, r, device, blooms_model)
        out[f"fedbart_fold{fold}"] = per_fold

    # ── Fed-BART averaged across folds ────────────────────────────────────
    if fedbart_preds_by_fold:
        avg: Dict[str, dict] = {}
        metric_keys = ("rouge_l", "bleu_4", "bertscore_f1", "blooms_cls_evs_mean")
        for slice_key in ("client_0", "client_1", "client_2", "overall"):
            present = [out[f"fedbart_fold{f}"][slice_key]
                       for f in fedbart_preds_by_fold
                       if slice_key in out[f"fedbart_fold{f}"]]
            if not present:
                continue
            avg[slice_key] = {}
            for mk in metric_keys:
                vals = [d[mk] for d in present if isinstance(d.get(mk), (int, float))]
                avg[slice_key][f"{mk}_mean"] = (sum(vals) / len(vals)) if vals else float("nan")
                if len(vals) > 1:
                    mu = avg[slice_key][f"{mk}_mean"]
                    var = sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)
                    avg[slice_key][f"{mk}_std"] = var ** 0.5
                else:
                    avg[slice_key][f"{mk}_std"] = 0.0
            avg[slice_key]["n"] = present[0]["n"]
        out["fedbart_avg"] = avg

    # ── GPT-4o ────────────────────────────────────────────────────────────
    if gpt4o_preds is not None:
        gpt_block: Dict[str, dict] = {}
        for cid in (0, 1, 2, None):
            idxs = slice_idxs(cid)
            if not idxs:
                continue
            p = [gpt4o_preds[i] for i in idxs]
            r = [references[i] for i in idxs]
            key = "overall" if cid is None else f"client_{cid}"
            gpt_block[key] = compute_lightweight(p, r, device, blooms_model)
        out["gpt4o"] = gpt_block

    return out


def format_summary_text(summary: Dict[str, dict], model_label: str = "fed-BART") -> str:
    """Pretty per-slice comparison: <model_label> (mean±std across folds) vs GPT-4o."""
    lines: List[str] = []
    metric_order = ("rouge_l", "bleu_4", "bertscore_f1", "blooms_cls_evs_mean")
    metric_labels = ("ROUGE-L", "BLEU-4", "BERTScore-F1", "Bloom-EVS")

    has_bart = "fedbart_avg" in summary
    has_gpt  = "gpt4o" in summary

    def fmt_bart(slice_key: str, mk: str) -> str:
        d = summary["fedbart_avg"].get(slice_key)
        if not d:
            return "  --  "
        mu, sd = d.get(f"{mk}_mean"), d.get(f"{mk}_std", 0.0)
        if mu is None or mu != mu:
            return "  --  "
        return f"{mu:.4f}±{sd:.4f}"

    def fmt_gpt(slice_key: str, mk: str) -> str:
        d = summary["gpt4o"].get(slice_key)
        if not d:
            return "  --  "
        v = d.get(mk)
        if v is None or v != v:
            return "  --  "
        return f"{v:.4f}      "

    lines.append("=" * 96)
    lines.append(f"  Federated {model_label} (mean±std over folds) vs GPT-4o   |  global mixed test")
    lines.append("=" * 96)
    header = f"  {'slice':10s}  {'model':22s}  " + "  ".join(f"{m:>14s}" for m in metric_labels) + "    n"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for slice_key, slice_label in (
        ("client_0", "client_0 (MIT)"),
        ("client_1", "client_1 (Stanford)"),
        ("client_2", "client_2 (ML papers)"),
        ("overall",  "OVERALL"),
    ):
        n = None
        if has_bart and slice_key in summary["fedbart_avg"]:
            n = summary["fedbart_avg"][slice_key].get("n")
        if n is None and has_gpt and slice_key in summary["gpt4o"]:
            n = summary["gpt4o"][slice_key].get("n")

        if has_bart:
            cells = "  ".join(f"{fmt_bart(slice_key, mk):>14s}" for mk in metric_order)
            lines.append(f"  {slice_label:10s}  {model_label:22s}  {cells}    {n}")
        if has_gpt:
            cells = "  ".join(f"{fmt_gpt(slice_key, mk):>14s}" for mk in metric_order)
            lines.append(f"  {slice_label:10s}  {'GPT-4o':22s}  {cells}    {n}")
        if has_bart and has_gpt and slice_key in summary["fedbart_avg"] and slice_key in summary["gpt4o"]:
            # Delta line (GPT-4o − fed-SLM mean)
            delta_cells = []
            for mk in metric_order:
                mu = summary["fedbart_avg"][slice_key].get(f"{mk}_mean")
                v  = summary["gpt4o"][slice_key].get(mk)
                if mu is None or v is None or mu != mu or v != v:
                    delta_cells.append(f"{'--':>14s}")
                else:
                    sign = "+" if v >= mu else ""
                    delta_cells.append(f"{sign}{v - mu:.4f}".rjust(14))
            lines.append(f"  {'':10s}  {'  delta (GPT-fed)':22s}  {'  '.join(delta_cells)}")
        lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    set_seeds(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Derive label + filename stem from --client-id-bart unless explicitly set.
    default_label, default_stem = _LABEL_DEFAULTS.get(
        args.client_id_bart, (f"fed-client{args.client_id_bart}",
                              f"fedclient{args.client_id_bart}"))
    if args.model_label is None:
        args.model_label = default_label
    if args.generation_stem is None:
        args.generation_stem = default_stem

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"SLM under test: client_{args.client_id_bart} → label={args.model_label!r}  "
          f"file-stem={args.generation_stem!r}")

    # ── 1. Load + sample ───────────────────────────────────────────────────
    global_test = json.loads(Path(args.global_test_file).read_text(encoding="utf-8"))
    print(f"Global mixed test: {len(global_test)} rows from {args.global_test_file}")

    samples, idxs = select_samples(global_test, args.limit_contexts, args.seed)
    n_ctx = len({s['context'] for s in samples})
    by_client = collections.Counter(s["_source_client"] for s in samples)
    print(f"Sampled {len(samples)} rows / {n_ctx} unique contexts "
          f"(client breakdown: {dict(by_client)})")

    (out_dir / "sample_ids.json").write_text(
        json.dumps({"limit_contexts": args.limit_contexts,
                    "seed": args.seed,
                    "n_rows": len(samples),
                    "n_unique_contexts": n_ctx,
                    "row_indices_in_global_test": idxs}, indent=2),
        encoding="utf-8",
    )

    # ── 2. Fed-SLM per fold ───────────────────────────────────────────────
    fedbart_preds_by_fold: Dict[int, List[str]] = {}
    if not args.skip_bart:
        for fold in args.folds:
            out_path = out_dir / f"generations_{args.generation_stem}_fold{fold}.json"
            if out_path.exists() and not args.force_regen:
                cached = json.loads(out_path.read_text(encoding="utf-8"))
                if cached.get("n_rows") == len(samples):
                    print(f"  [fold {fold}] cached → reusing {out_path.name}")
                    fedbart_preds_by_fold[fold] = cached["predictions"]
                    continue

            preds = run_fedbart_fold(fold, samples, args, device)
            fedbart_preds_by_fold[fold] = preds
            out_path.write_text(
                json.dumps({"fold": fold, "n_rows": len(samples),
                            "model_label": args.model_label,
                            "client_id": args.client_id_bart,
                            "predictions": preds}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"  [fold {fold}] saved → {out_path}")

    # ── 3. GPT-4o ──────────────────────────────────────────────────────────
    gpt4o_preds: List[str] | None = None
    if not args.skip_gpt4o:
        # Check optional external cache first (lets multiple architecture
        # runs share a single 916-row GPT-4o pass).
        external_cache = args.gpt4o_cache_path and Path(args.gpt4o_cache_path)
        if external_cache and external_cache.exists() and not args.force_regen:
            cached = json.loads(external_cache.read_text(encoding="utf-8"))
            if cached.get("n_rows") == len(samples):
                print(f"  [gpt-4o] external cache → {external_cache}")
                gpt4o_preds = cached["predictions"]

        out_path = out_dir / "generations_gpt4o.json"
        if gpt4o_preds is None and out_path.exists() and not args.force_regen:
            cached = json.loads(out_path.read_text(encoding="utf-8"))
            if cached.get("n_rows") == len(samples):
                print(f"  [gpt-4o] cached → reusing {out_path.name}")
                gpt4o_preds = cached["predictions"]

        if gpt4o_preds is None:
            api_key, source = load_openai_key(
                args.openai_api_key_file,
                *default_key_search_paths(str(REPO)),
            )
            if not api_key:
                raise RuntimeError(
                    "No OpenAI key found. Put it in "
                    f"{args.openai_api_key_file} or export OPENAI_API_KEY=…"
                )
            print(f"  [gpt-4o] using key from {source}")
            print(f"  [gpt-4o] model={args.gpt4o_model} "
                  f"temp={args.gpt4o_temperature} max_tokens={args.gpt4o_max_tokens}")

            gpt4o_preds = run_gpt4o(samples, api_key, args)
            out_path.write_text(
                json.dumps({"model": args.gpt4o_model,
                            "temperature": args.gpt4o_temperature,
                            "n_rows": len(samples),
                            "predictions": gpt4o_preds}, indent=2,
                           ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"  [gpt-4o] saved → {out_path}")

    # ── 4 & 5. Metrics + summary ──────────────────────────────────────────
    print("\nComputing metrics …")
    summary = aggregate_summary(samples, fedbart_preds_by_fold, gpt4o_preds,
                                device, args.blooms_model)
    summary["_meta"] = {
        "model_label": args.model_label,
        "client_id":   args.client_id_bart,
        "n_rows":      len(samples),
        "folds":       list(fedbart_preds_by_fold.keys()),
    }
    (out_dir / "metrics_per_run.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    txt = format_summary_text(summary, model_label=args.model_label)
    print("\n" + txt)
    (out_dir / "comparison_summary.txt").write_text(txt, encoding="utf-8")
    print(f"\nAll outputs in {out_dir}/")


if __name__ == "__main__":
    main()
