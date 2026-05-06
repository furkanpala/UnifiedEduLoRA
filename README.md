# EquitableEdu

A graph-based federated learning framework for equitable QA generation across heterogeneous language models and institutions.

---

## Phase 1 - Data Preparation 

Each collaborating institution follows these steps independently before the first training meeting. The goal is to produce a single validated JSON file and a set of reproducible splits.

---

### Step 1 — Convert Your Source Material to Plain-Text Chunks

Convert your raw educational content into plain-text **context chunks**:

- **150–400 words per chunk** — one coherent topic per chunk
- **Plain prose only** — no bullet points, LaTeX math, figure captions, headers, or footers
- Each chunk should stand alone: a reader with no surrounding context should be able to answer questions from it

Two converter scripts are provided in `unifiedfl/data/` as starting points. **They will likely need small adjustments for your specific files** — every PDF and slide deck is formatted differently. Read the comments at the top of each script before running.

#### PDF → chunks (`data/pdf_to_chunks.py`)

```bash
pip install pymupdf

# Inspect first, then run (paths are from the repo root):
python unifiedfl/data/pdf_to_chunks.py lecture.pdf

# Skip cover page, references, and appendix (0-based page indices):
python unifiedfl/data/pdf_to_chunks.py lecture.pdf --skip-pages 0 1 42 43 44

# Output: lecture_chunks.json  — a JSON array of plain-text strings
```

Key things to adjust inside the script:
- `SKIP_PAGES` — pages to exclude (cover, table of contents, references, appendix)
- `SKIP_IF_FEWER` — raise this if very sparse pages are slipping through
- `TARGET_WORDS` — default is 250; increase for dense academic text
- `_strip_boilerplate()` — removes the first/last line of each page (running headers/footers); disable if your PDF does not have these

#### PowerPoint → chunks (`data/pptx_to_chunks.py`)

```bash
pip install python-pptx

# First, see which slide layouts exist in your file:
python unifiedfl/data/pptx_to_chunks.py lecture.pptx --list-layouts

# Then run (title slides and section headers are skipped by default):
python unifiedfl/data/pptx_to_chunks.py lecture.pptx

# If your slides have speaker notes with the real explanation, include them:
python unifiedfl/data/pptx_to_chunks.py lecture.pptx --include-notes

# Output: lecture_chunks.json  — a JSON array of plain-text strings
```

Key things to adjust inside the script:
- `SKIP_LAYOUTS` — layout names to exclude (use `--list-layouts` to find the right names for your deck)
- `SKIP_IF_FEWER` — slides with fewer words than this are dropped; lower it if too many slides are being skipped
- `INCLUDE_NOTES` — set to `True` if speaker notes contain the substantive explanation rather than the slide body
- `INCLUDE_TITLES` — set to `False` if slide titles are just labels that add no educational content

#### Using the output

Both scripts produce a JSON array of chunk strings. Pass these directly into `generate_qa.py`:

```python
import json
my_chunks = json.load(open("lecture_chunks.json"))
# then proceed with the generate_qa_for_context() loop in Step 2
```

---

### Step 2 — Generate QA Pairs (`generate_qa.py`)

Use `generate_qa.py` to call the OpenAI API and produce annotated question-answer pairs for each chunk. This fills in `context_topics`, `qa_pairs`, Bloom's taxonomy levels, and difficulty labels automatically.

```python
import json
from generate_qa import generate_qa_for_context

API_KEY    = "sk-..."                       # your OpenAI API key
CLIENT_ID  = 0                              # your assigned institution ID (0, 1, 2, ...)
SOURCE     = "Intro to ML — Lecture 3"     # human-readable label for this source

my_chunks = ["...", "...", ...]             # your plain-text chunks from Step 1

entries = []
for i, chunk in enumerate(my_chunks):
    result = generate_qa_for_context(
        context = chunk,
        api_key = API_KEY,
        n_pairs = 5,          # QA pairs per chunk (5 recommended)
        model   = "gpt-4o",   # or "gpt-4o-mini" for cheaper drafts
    )
    if result is None:
        print(f"[WARNING] chunk {i} failed — skipping")
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
```

**Output:** a single file `client<N>_data.json` — a JSON array where each element is one context chunk with its generated QA pairs.

#### Expected format

```json
[
  {
    "entry_id":           "client0_0000",
    "source_description": "Intro to ML — Lecture 3",
    "clean_context":      "Gradient descent is an optimization algorithm ...",
    "context_topics":     ["Gradient Descent", "Learning Rate", "Convergence"],
    "qa_pairs": [
      {
        "question":              "What is the role of the learning rate in gradient descent?",
        "answer":                "The learning rate controls the step size taken at each iteration ...",
        "question_topic":        "Learning Rate Sensitivity",
        "bloom_level":           2,
        "bloom_justification":   "Requires understanding how a parameter affects algorithm behavior.",
        "difficulty":            "easy",
        "answerable_from_context": true
      }
    ]
  }
]
```

---

### Step 3 — Validate Your File (`validate.py`)

Before splitting, run the validator to confirm your file is correctly formatted:

```bash
python unifiedfl/validate.py client0_data.json
```

**Strict mode (default)** — for files produced by `generate_qa.py`. Checks:
- All required fields are present and non-empty
- `clean_context` is within the word-count range
- `bloom_level` is an integer between 1 and 6
- `difficulty` is one of `easy`, `medium`, `hard`
- `answerable_from_context` is `true` for every pair

**Lenient mode (`--lenient`)** — for machine-enhanced data prepared outside Phase 1 (e.g. files lacking `entry_id`, `bloom_justification`, or saved as concatenated/streaming JSON):

```bash
python unifiedfl/validate.py --lenient my_enhanced_data.json
```

Lenient mode accepts non-array JSON formats and only validates the fields the training code actually consumes (`clean_context`, `qa_pairs`, `question`, `answer`, `question_topic`, `bloom_level`).

Both modes print a summary of entry counts, total QA pairs, Bloom level distribution, and difficulty distribution.

**Fix any reported errors before proceeding to Step 4.**

---

### Step 3.5 — (optional, federated experiments) Report data statistics (`data_stats.py`)

If you're contributing to a federated experiment that uses `split.py --balance`, the coordinator needs to know the smallest entry count across all clients (that's the cap). Each participant runs `data_stats.py` on their own file and reports back the entry count — no raw data leaves their machine.

```bash
# What each participant runs
python unifiedfl/data_stats.py my_data.json

# What the coordinator runs across all collected files
python unifiedfl/data_stats.py \
    --client 0:client0_data.json \
    --client 1:client1_data.json \
    --client 2:client2_data.json \
    --save cross_client_stats.json
```

The cross-client output highlights the minimum entry count and the recommended `--balance` cap for `split.py`.

---

### Step 4 — Generate Splits (`split.py`)

Once the file passes validation, generate the train/val/test splits. **All collaborators must use `--seed 42`** — this is the anchor that makes all three experiments (individual, FedKD, UnifiedEdu) directly comparable.

```bash
python unifiedfl/split.py \
    --client 0:client0_data.json \
    --seed   42 \
    --output-dir outputs/
```

For multiple clients on the same machine:

```bash
python unifiedfl/split.py \
    --client 0:client0_data.json \
    --client 1:client1_data.json \
    --seed   42 \
    --output-dir outputs/
```

> **Note for individual participants (Phase 2 collaborators):** the basic
> command above is all you need. You only have your own data, so you can't
> meaningfully balance against anyone else.
>
> A `--balance` flag exists in `split.py` for the federated comparison
> (Phase 3): when one machine has *all* clients' data files, the coordinator
> can pass `--balance` to cap every client at the smallest client's entry
> count, isolating the GNN's effect from differences in data quantity.
> It's a coordinator-only feature and irrelevant to Phase 2.

#### Split protocol

| Step | What happens |
|------|-------------|
| 1 | 15% of entries are held out as a **fixed test set** — identical across all folds, never used for training or validation |
| 2 | The remaining 85% is divided into **3 equal folds** |
| 3 | For fold k: **train** = the other two folds (flattened into QA pairs); **val** = fold k |

All splitting is at the **entry level** (not QA-pair level) to prevent context leakage between train and val.

#### Expected outputs

```
outputs/splits/
├── client_0_test.json          ← fixed test set (same for all folds)
├── client_0_fold1_train.json
├── client_0_fold1_val.json
├── client_0_fold2_train.json
├── client_0_fold2_val.json
├── client_0_fold3_train.json
├── client_0_fold3_val.json
└── checksums.txt               ← MD5 hash of every file above
```

Each split file is a flat JSON array of QA samples:

```json
[
  {
    "context":        "Gradient descent is an optimization algorithm ...",
    "question":       "What is the role of the learning rate?",
    "answer":         "The learning rate controls the step size ...",
    "question_topic": "Learning Rate Sensitivity",
    "bloom_level":    2,
    "difficulty":     "easy"
  }
]
```

---

### Step 5 — Send Your Checksums to the Coordinator

Send `outputs/splits/checksums.txt` to the project coordinator. This file contains the MD5 hash of every split file and the exact seed/fold parameters used. The coordinator verifies that all collaborators produced identical splits from the same seed.

```
# seed=42  n_folds=3  test_ratio=0.15
543510746ea6ccc0280b595909be3e2c  client_0_test.json
e9ed4f86717a9e56d0b4866dd13800d9  client_0_fold1_train.json
...
```

---

## Phase 2 — Individual Training Baseline (Experiment 1)

Each institution trains a local LoRA adapter on their own data, under **three prompt-conditioning regimes**. No data leaves the institution. This produces the individual baseline against which the federated model is compared.

### Three conditioning regimes

| Regime | Prompt template | Decoding at evaluation |
|---|---|---|
| **baseline** | context only | diverse beam search (5 outputs per unique context, Hungarian-matched to the 5 references via ROUGE-L)  |
| **topic** | context + `question_topic` (each QA pair has a unique topic) | standard single-output beam search per sample |
| **bloom** | context + `bloom_level` and verb (1=Remember … 6=Create) | standard single-output beam search per sample |

Total runs per institution: **3 conditionings × 3 folds = 9 trainings**. The orchestrator script handles all 9 in one invocation and is resume-friendly.

### Early stopping and checkpoints

Training runs for up to **100 epochs** with early stopping based on a fast generation-based metric:

- **Greedy-decoded ROUGE-L** on a fixed 50-sample subsample of the val set (drawn once with `--seed`). This reflects the actual autoregressive generation quality — unlike teacher-forced val loss, which can keep dropping while real generation gets worse.
- Patience of **10 epochs** without improvement before stopping (configurable via `--patience`).
- Two model artifacts are kept and evaluated separately at the end:
  - **best** — lowest-error checkpoint by the early-stop metric
  - **final** — state at the last completed epoch

Pass `--early-stop-metric val_loss` if you'd rather use the older teacher-forced val loss (faster per epoch but less aligned with generation quality).

---

### Environment Setup (Google Colab)

```python
# 1. Mount your Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Clone the repo
!git clone https://github.com/basiralab/EquitableEdu /content/EquitableEdu
%cd /content/EquitableEdu

# 3. Place your OpenAI key file at the repo root (gitignored, we do not commit this).
#    The orchestrator and train scripts pick it up automatically.
!echo 'sk-...' > openai_api_key
!chmod 600 openai_api_key

# 4. Dependencies. torchao must be removed — Colab's pre-installed 0.10 breaks PEFT.
!pip uninstall -y torchao
!pip install -q transformers==4.46.0 peft accelerate sentencepiece \
    rouge_score bert_score nltk scipy openai 'numpy<2'
```

Upload your Phase 1 splits to Drive, e.g.:
```
/content/drive/MyDrive/unifiedfl/outputs/splits/
  client_<N>_test.json
  client_<N>_fold{1,2,3}_train.json
  client_<N>_fold{1,2,3}_val.json
```

The test set is **not used in Phase 2** — it stays held out for the federated comparison in Phase 3.

---

### Recommended: orchestrator script (one command, all 9 runs)

The orchestrator runs `train_client.py` 9 times (3 conditionings × 3 folds), writes outputs into per-conditioning subdirectories, and skips folds whose `metrics_val.json` already exists.

```bash
python experiments/03_run_three_conditionings.py \
    --splits-dir /content/drive/MyDrive/unifiedfl/outputs/splits \
    --output-dir /content/drive/MyDrive/unifiedfl/outputs \
    --client-id  <your_client_id_from_phase_1> \
    --model      <your_assigned_model> \
    --family     <your_assigned_model_family> \
    --targets    <your_assigned_model_targets>
```

Comprehensive evaluation (ROUGE-L, BLEU-4, BERTScore + RTC, Faithfulness, QAFactEval, RQUGE, Answer Relevancy, Bloom's BERT, Bloom's LLM judge, GPT-4o QA judge) runs by default and uses your `openai_api_key` file. Pass `--fast-eval` to skip the heavy/LLM metrics if you just want a quick sanity check.

#### Useful orchestrator flags

| Flag | Purpose |
|---|---|
| `--conditionings baseline` | Only run a subset (default: all three) |
| `--folds 1` | Only run fold 1 |
| `--force` | Re-run even if `metrics_val.json` exists |
| `--fast-eval` | Skip heavy/LLM metrics (faster, fewer numbers) |
| `--num-epochs 30` | Cut training time |



#### Model choices by architecture

| Model | `--model` | `--family` | `--targets` |
|---|---|---|---|
| Flan-T5-base | `google/flan-t5-base` | `t5` | `q v` |
| BART-base | `facebook/bart-base` | `bart` | `q_proj v_proj` |
| LED-base | `allenai/led-base-16384` | `led` | `q_proj v_proj` |
| Pegasus-X-base | `google/pegasus-x-base` | `pegasus_x` | `q_proj v_proj` |
| MarianMT (en→de) | `Helsinki-NLP/opus-mt-en-de` | `marian` | `q_proj v_proj` |
| ProphetNet-large | `microsoft/prophetnet-large-uncased` | `prophetnet` | `query_proj value_proj` |

---

### Alternative: invoke `train_client.py` directly

You can also run a single conditioning + fold at a time. The output path now embeds the conditioning automatically, so different `--conditioning` runs cannot overwrite each other:

```bash
python unifiedfl/train_client.py \
    --client-id      <your_client_id> \
    --fold           1 \
    --conditioning   baseline \
    --model          facebook/bart-base \
    --family         bart \
    --targets        q_proj v_proj \
    --splits-dir     /content/drive/MyDrive/unifiedfl/outputs/splits \
    --output-dir     /content/drive/MyDrive/unifiedfl/outputs \
    --openai-api-key sk-...
```

**Note:** unlike the orchestrator, `train_client.py` does **not** auto-load the `openai_api_key` file from the repo root. To enable comprehensive evaluation when calling it directly, pass `--openai-api-key sk-...` explicitly (or set the `OPENAI_API_KEY` env var and point the flag at it). Without an API key, only the local-model metrics run.

#### Key hyperparameters

| Argument | Default | What it controls |
|---|---|---|
| `--conditioning` | `baseline` | One of `baseline`, `topic`, `bloom` (drives both prompt template and eval strategy) |
| `--fold` | required | Which CV fold to train on (1, 2, or 3) |
| `--num-epochs` | 100 | Maximum training epochs |
| `--batch-size` | 4 | Samples per gradient step |
| `--lr` | 3e-4 | Peak learning rate (cosine decay with warmup) |
| `--warmup-ratio` | 0.1 | Fraction of total steps used for LR warm-up |
| `--grad-clip` | 1.0 | Gradient clipping norm |
| `--patience` | 10 | Early stopping — halt if the early-stop metric does not improve for this many epochs |
| `--min-delta` | 1e-4 | Minimum improvement in the early-stop metric to reset the patience counter |
| `--early-stop-metric` | `rouge_l` | `rouge_l` (greedy decoding on a 50-sample val subsample, default) or `val_loss` (teacher-forced cross-entropy on full val) |
| `--fast-eval-samples` | 50 | Number of val samples used for the fast ROUGE-L metric (fixed across epochs via `--seed`) |
| `--lora-r` | 16 | LoRA rank |
| `--lora-alpha` | 32 | LoRA scaling factor |
| `--lora-dropout` | 0.1 | LoRA dropout |
| `--checkpoint-every` | 5 | Save a checkpoint every N epochs |
| `--no-heavy` | off | Skip heavy metrics (UnifiedQA + DeBERTa) in final evaluation |
| `--openai-api-key` | none — must be passed explicitly when calling `train_client.py` directly (the orchestrator auto-loads from the `openai_api_key` file) | Enables Answer Relevancy, Bloom's LLM judge, and GPT-4o QA judge |

---

### Output structure (after all 9 runs)

```
outputs/
├── baseline/
│   └── client_<N>/
│       ├── fold1/
│       │   ├── best/lora_model/                         ← best-checkpoint LoRA weights
│       │   ├── final/lora_model/                        ← final-epoch LoRA weights
│       │   ├── checkpoints/                             ← periodic checkpoints (every 5 epochs)
│       │   ├── loss_history.json                        ← train loss, val loss, early-stop metric per epoch
│       │   └── results/
│       │       ├── best/
│       │       │   ├── metrics_val.json                 ← comprehensive metrics on val (best ckpt)
│       │       │   └── generated_qas_val.json           ← model outputs vs. references (best ckpt)
│       │       └── final/
│       │           ├── metrics_val.json                 ← same metrics, evaluated on final ckpt
│       │           └── generated_qas_val.json           ← outputs from the final ckpt
│       ├── fold2/
│       └── fold3/
├── topic/
│   └── client_<N>/fold{1,2,3}/...
└── bloom/
    └── client_<N>/fold{1,2,3}/...
```

Both checkpoints are evaluated independently so you can compare best vs. final and see how much overfitting occurred.

`metrics_val.json` contains (with comprehensive eval — i.e. without `--fast-eval`):

| Metric | Description |
|---|---|
| `rouge_l` | ROUGE-L F1 |
| `bleu_4` | BLEU-4 |
| `bertscore_f1` | BERTScore F1 (DeBERTa-v3) |
| `rtc` | Round-Trip Consistency — UnifiedQA re-answers the generated question from context |
| `faithfulness` | RAGAS Faithfulness — fraction of answer claims entailed by the context |
| `qafacteval` | QAFactEval (approx.) — UnifiedQA yes/no factual consistency |
| `rquge` | RQUGE (approx.) — answer quality score in [1, 5] |
| `answer_relevancy` | RAGAS Answer Relevancy — cosine similarity of original vs. reverse-generated questions |
| `blooms_cls_distribution` | Bloom level counts from a local fine-tuned BERT classifier |
| `blooms_cls_evs_mean` | Educational Value Score from classifier — mean normalized Bloom level in [0, 1] |
| `blooms_llm_distribution` | Bloom level counts from GPT-4o-mini LLM judge |
| `blooms_llm_evs_mean` | Educational Value Score from LLM judge in [0, 1] |
| `llm_judge_context_grounding_mean` | LLM judge — is the question answerable from the context (1–5) |
| `llm_judge_educational_value_mean` | LLM judge — pedagogical merit of the question (1–5) |
| `llm_judge_answer_correctness_mean` | LLM judge — answer is factually correct given the context (1–5) |
| `llm_judge_answer_relevance_mean` | LLM judge — answer actually addresses the question (1–5) |
| `llm_judge_overall_mean` | LLM judge — mean across the four dimensions, normalized to [0, 1] |

---

### Aggregating results across folds

After the orchestrator finishes, get a summary table of per-fold metrics + per-conditioning means:

```bash
# best checkpoint (default)
python experiments/04_aggregate_results.py \
    --output-dir /content/drive/MyDrive/unifiedfl/outputs \
    --client-id  <your_client_id> \
    --save-summary /content/drive/MyDrive/unifiedfl/outputs/summary_best.json

# final checkpoint (compare against best to see overfitting)
python experiments/04_aggregate_results.py \
    --output-dir /content/drive/MyDrive/unifiedfl/outputs \
    --client-id  <your_client_id> \
    --checkpoint final \
    --save-summary /content/drive/MyDrive/unifiedfl/outputs/summary_final.json
```

Send both `summary_*.json` files and the entire `outputs/` directory back to the project coordinator.

---

### Resuming After a Colab Disconnect

Three resume scenarios, in order of granularity:

**1. Orchestrator-level (most common)** — re-run the same `experiments/03_run_three_conditionings.py` command and it will skip any (conditioning, fold) pair whose `results/best/metrics_val.json` already exists. To force re-running a specific subset, add `--force` and (optionally) `--conditionings <name>` / `--folds N`.

**2. Mid-fold resume** — `train_client.py` directly supports `--resume-from-epoch N`. It reloads the checkpoint at `outputs/<conditioning>/client_<N>/fold<k>/checkpoints/epoch_N/` and continues from epoch N+1, preserving the early-stopping state and best checkpoint.

**3. Eval-only resume** — if training finished and the `best/lora_model/` + `final/lora_model/` adapters are on disk but the eval block crashed (e.g. comprehensive eval OOMed, Colab session died during the OpenAI-judge calls), use `experiments/11_recover_eval_only.py`. It re-runs only the post-training eval on the saved adapters and writes the missing `results/{best,final}/metrics_val.json`. Without this you'd otherwise re-train the fold from scratch because the orchestrator's skip-check looks for `metrics_val.json`, not the adapter directories.

```bash
python experiments/11_recover_eval_only.py \
    --output-dir   /content/drive/MyDrive/unifiedfl/outputs \
    --splits-dir   /content/drive/MyDrive/unifiedfl/outputs/splits \
    --client-id    <id> --fold <k> --conditioning <baseline|topic|bloom> \
    --model <model> --family <family> --targets <targets...>
# Auto-skips a side that already has metrics_val.json; pass --skip-best
# or --skip-final to force-skip one. Add --no-heavy to disable
# UnifiedQA + DeBERTa metrics for a faster recovery.
```

---

## Repository Structure

```
unifiedfl/
├── generate_qa.py            ← Step 2: generate QA pairs from plain-text chunks
├── validate.py               ← Step 3: validate your client data file (--lenient mode for non-Phase-1 data)
├── data_stats.py             ← Step 3.5: report per-client / cross-client data statistics
├── split.py                  ← Step 4: create train/val/test splits (--balance for federated experiments)
├── train_client.py           ← Experiment 1: individual LoRA training (baseline)
├── train_federated.py        ← Experiment 3: UnifiedEdu federated training
├── eval_diverse_decoding.py  ← re-evaluate saved checkpoints with diverse beam search
├── visualize_graph.py        ← architecture-graph visualizer (debugging the GNN input)
├── config/                   ← model and training hyperparameters
├── data/                     ← dataset classes and preprocessing utilities
│   ├── pdf_to_chunks.py      ← Step 1 helper: PDF → plain-text chunks
│   └── pptx_to_chunks.py     ← Step 1 helper: PowerPoint → plain-text chunks
├── models/                   ← GNN, FiLM adapter, client model wrappers
├── federation/               ← federated server and client logic
├── training/                 ← training loop and checkpointing
├── evaluation/               ← metrics and evaluator
└── utils/                    ← logging utilities

experiments/                  ← experiment orchestrators (sit on top of train_client.py / train_federated.py)
├── 01_enhance_mit_data.py                  ← GPT-4o-mini enhances raw MIT data with question_topic + bloom_level
├── 02_split_3fold.py                       ← per-client 3-fold CV split (single-dataset MIT experiment)
├── 03_run_three_conditionings.py           ← Phase 2 orchestrator: 3 conditionings × 3 folds = 9 trainings
├── 04_aggregate_results.py                 ← per-client summary table (best or final checkpoint)
├── 05_enhance_all_datasets.py              ← enhances MIT, Stanford, Papers in one pass
├── 06_split_fed_experiment.py              ← single-split (80/10/10) for the 3-client federated experiment
├── 07_run_individual_baselines_fed_exp.py  ← individual baselines for the federated comparison
├── 08_run_federated_training.py            ← train_federated.py wrapper with the federated-experiment defaults
├── 09_compare_results.py                   ← individual-vs-federated comparison table
├── 10_eval_checkpoint.py                   ← evaluate any federated checkpoint and save metrics for comparison
├── 11_recover_eval_only.py                 ← re-run only the post-train eval on saved best+final adapters (Phase 2 recovery)
└── quick_summary.py                        ← per-fold + best-vs-final analysis helper for the 3-conditioning x 3-fold tree

tests/                        ← pytest suite (run with `pytest tests/` — 64 tests, CPU-only, no model downloads)
├── test_metrics_helpers.py            ← parse_qa, token_f1, cosine, sanitize_json_escapes
├── test_dataset.py                    ← render_prompt edge cases + label -100 invariant
├── test_gnn.py                        ← ArchitectureGNN forward / grad / shape invariants
├── test_federated_server.py           ← weighted FedAvg correctness
├── test_split_3fold.py                ← CV fold disjointness + reproducibility
└── test_orchestrator_defaults.py      ← regression guard: orchestrator defaults must match train_client + no .load_adapter() calls

infer.py                      ← student-facing inference: load a saved checkpoint, generate QA from a context
```
