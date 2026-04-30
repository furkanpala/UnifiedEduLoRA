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

# Inspect first, then run:
python data/pdf_to_chunks.py lecture.pdf

# Skip cover page, references, and appendix (0-based page indices):
python data/pdf_to_chunks.py lecture.pdf --skip-pages 0 1 42 43 44

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
python data/pptx_to_chunks.py lecture.pptx --list-layouts

# Then run (title slides and section headers are skipped by default):
python data/pptx_to_chunks.py lecture.pptx

# If your slides have speaker notes with the real explanation, include them:
python data/pptx_to_chunks.py lecture.pptx --include-notes

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
    "clean_context":      "Gradient descent is an optimisation algorithm ...",
    "context_topics":     ["Gradient Descent", "Learning Rate", "Convergence"],
    "qa_pairs": [
      {
        "question":              "What is the role of the learning rate in gradient descent?",
        "answer":                "The learning rate controls the step size taken at each iteration ...",
        "question_topic":        "Learning Rate Sensitivity",
        "bloom_level":           2,
        "bloom_justification":   "Requires understanding how a parameter affects algorithm behaviour.",
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
python validate.py client0_data.json
```

The validator checks:
- All required fields are present and non-empty
- `clean_context` is within the word-count range
- `bloom_level` is an integer between 1 and 6
- `difficulty` is one of `easy`, `medium`, `hard`
- `answerable_from_context` is `true` for every pair

It also prints a summary of entry counts, total QA pairs, Bloom level distribution, and difficulty distribution.

**Fix any reported errors before proceeding to Step 4.**

---

### Step 4 — Generate Splits (`split.py`)

Once the file passes validation, generate the train/val/test splits. **All collaborators must use `--seed 42`** — this is the anchor that makes all three experiments (individual, FedKD, UnifiedEdu) directly comparable.

```bash
python split.py \
    --client 0:client0_data.json \
    --seed   42 \
    --output-dir outputs/
```

For multiple clients on the same machine:

```bash
python split.py \
    --client 0:client0_data.json \
    --client 1:client1_data.json \
    --seed   42 \
    --output-dir outputs/
```

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
    "context":        "Gradient descent is an optimisation algorithm ...",
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

---

## Phase 2 — Individual Training Baseline (Experiment 1)

Each institution trains a local LoRA adapter on their own data. No data leaves the institution. This produces the individual baseline against which the federated model is compared.

---

### Environment Setup (Google Colab)

```python
# 1. Mount your Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Clone the repo (or upload it to Drive and add the path)
!git clone https://github.com/basiralab/EquitableEdu /content/EquitableEdu
%cd /content/EquitableEdu

# 3. Install dependencies
!pip install -q transformers peft torch torch-geometric \
    rouge_score bert_score nltk accelerate sentencepiece \
    "numpy<2" openai
```

Upload your `client<N>_fold*.json` split files (produced in Phase 1) to Drive, e.g. to:
```
/content/drive/MyDrive/unifiedfl/outputs/splits/
```

---

### Training Command

Run one fold at a time. Repeat for folds 1, 2, and 3.

```bash
python unifiedfl/train_client.py \
    --client-id   <your_client_id_you_used_in_phase1> \
    --fold        1 \
    --model       <your_assigned_model> \
    --family      <your_assigned_model_family> \
    --targets     <your_assigned_model_targets> \
    --splits-dir  /content/drive/MyDrive/unifiedfl/outputs/splits \
    --output-dir  /content/drive/MyDrive/unifiedfl/outputs \
    --num-epochs  60 \
    --batch-size  4 \
    --lr          3e-4 \
    --patience    10 \
    --openai-api-key  sk-... (your api key provided by email)
```

#### Key hyperparameters

| Argument | Default | What it controls |
|---|---|---|
| `--fold` | required | Which CV fold to train on (1, 2, or 3) |
| `--num-epochs` | 60 | Maximum training epochs |
| `--batch-size` | 4 | Samples per gradient step |
| `--lr` | 3e-4 | Peak learning rate (cosine decay with warmup) |
| `--warmup-ratio` | 0.1 | Fraction of total steps used for LR warm-up |
| `--grad-clip` | 1.0 | Gradient clipping norm |
| `--patience` | 10 | Early stopping — halt if val loss does not improve for this many epochs |
| `--min-delta` | 1e-4 | Minimum improvement in val loss to reset the patience counter |
| `--lora-r` | 16 | LoRA rank |
| `--lora-alpha` | 32 | LoRA scaling factor |
| `--lora-dropout` | 0.1 | LoRA dropout |
| `--preview-every` | 5 | Print a generated QA from the val set every N epochs (0 = off) |
| `--checkpoint-every` | 10 | Save a checkpoint every N epochs |
| `--no-heavy` | off | Skip heavy metrics (UnifiedQA + DeBERTa) in final evaluation |
| `--openai-api-key` | none | Enable Answer Relevancy and Bloom's LLM judge metrics |
| `--blooms-model` | `cip29/bert-blooms-taxonomy-classifier` | HuggingFace model ID for the local Bloom's classifier (set to `''` to skip) |

#### Model choices by architecture

| Model | `--model` | `--family` | `--targets` |
|---|---|---|---|
| Flan-T5-small | `google/flan-t5-small` | `t5` | `q v` |
| Flan-T5-base | `google/flan-t5-base` | `t5` | `q v` |
| BART-base | `facebook/bart-base` | `bart` | `q_proj v_proj` |
| LED-base | `allenai/led-base-16384` | `led` | `q_proj v_proj` |

---

### Running All Three Folds

Run the command above three times, changing `--fold 1`, `--fold 2`, `--fold 3`. Each fold writes its outputs to a separate directory.

```
outputs/
└── client_0/
    ├── fold1/
    │   ├── best/lora_model/          ← best LoRA weights (by val loss)
    │   ├── final/lora_model/         ← weights at last epoch
    │   ├── checkpoints/              ← periodic checkpoints for resuming
    │   ├── loss_history.json         ← train and val loss per epoch
    │   ├── metrics_val.json          ← all evaluation metrics on the val set
    │   └── generated_qas_val.json    ← model-generated QA pairs vs. references
    ├── fold2/
    └── fold3/
```

`metrics_val.json` contains:

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
| `blooms_cls_evs_mean` | Educational Value Score from classifier — mean normalised Bloom level in [0, 1] |
| `blooms_llm_distribution` | Bloom level counts from GPT-4o-mini LLM judge (requires `--openai-api-key`) |
| `blooms_llm_evs_mean` | Educational Value Score from LLM judge in [0, 1] |
| `llm_judge_context_grounding_mean` | LLM judge — is the question answerable from the context (1–5) |
| `llm_judge_educational_value_mean` | LLM judge — pedagogical merit of the question (1–5) |
| `llm_judge_answer_correctness_mean` | LLM judge — answer is factually correct given the context (1–5) |
| `llm_judge_answer_relevance_mean` | LLM judge — answer actually addresses the question (1–5) |
| `llm_judge_overall_mean` | LLM judge — mean across the four dimensions, normalised to [0, 1] |

---

### Resuming After a Colab Disconnect

If training is interrupted, resume from the last saved checkpoint:

```bash
python unifiedfl/train_client.py \
    --client-id 0 \
    --fold 1 \
    ... \
    --resume-from-epoch 30
```

The checkpoint at `outputs/client_0/fold1/checkpoints/epoch_30/` will be loaded and training will continue from epoch 31.

---

## Repository Structure

```
unifiedfl/
├── generate_qa.py       ← Step 2: generate QA pairs from plain-text chunks
├── validate.py          ← Step 3: validate your client data file
├── split.py             ← Step 4: create train/val/test splits
├── train_client.py      ← Experiment 1: individual LoRA training (baseline)
├── train_federated.py   ← Experiment 3: UnifiedEdu federated training
├── config/              ← model and training hyperparameters
├── data/                ← dataset classes and preprocessing utilities
├── models/              ← GNN, FiLM adapter, client model wrappers
├── federation/          ← federated server and client logic
├── training/            ← training loop and checkpointing
├── evaluation/          ← metrics and evaluator
└── utils/               ← logging utilities
```
