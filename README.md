# EquitableEdu

A graph-based federated learning framework for equitable QA generation across heterogeneous language models and institutions.

---

## Data Preparation — Step-by-Step

Each collaborating institution follows these steps independently before the first training meeting. The goal is to produce a single validated JSON file and a set of reproducible splits.

---

### Step 1 — Convert Your Source Material to Plain-Text Chunks

Convert your raw educational content (PDFs, lecture slides, papers, etc.) into plain-text **context chunks**:

- **150–400 words per chunk** — one coherent topic per chunk
- **Plain prose only** — remove bullet points, LaTeX math, figure captions, headers, and footers
- Each chunk should stand alone: a reader with no context should be able to answer questions from it

You are responsible for this step using whatever tools suit your source format (e.g. PyMuPDF for PDFs, python-pptx for slides).

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
