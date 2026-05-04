# MIT three-conditioning experiment

Train `facebook/bart-base` on `unifiedfl/data/ML_QA_LectureNotes_MIT.json` under
three prompt-conditioning regimes, with 3-fold cross-validation:

1. `baseline` — context-only prompt; evaluation uses diverse beam search
   (5 outputs / unique context, Hungarian-matched to references).
2. `topic` — prompt names `question_topic`. Each of the 5 references per
   context has its own topic, so each input is unique → standard single-output
   beam search at eval.
3. `bloom` — prompt names `bloom_level` (1–6) and the corresponding cognitive
   verb (Remember / Understand / … / Create). Same single-output eval as topic.

Total: 3 conditionings × 3 folds = 9 trainings.

## How to run on Colab via VSCode

Open a terminal on the Colab VM (VSCode → Terminal → New Terminal) and run the
four scripts in order. Each script's logs stream live to the terminal — no
Jupyter buffering.

### 0. Setup once

```bash
# Mount Drive and clone the repo (or pull latest)
git clone https://github.com/basiralab/EquitableEdu.git /content/EquitableEdu
cd /content/EquitableEdu

# Place your OpenAI key file at the repo root (gitignored — never committed)
echo 'sk-...' > openai_api_key
chmod 600 openai_api_key

# Dependencies. torchao must be removed — Colab's pre-installed 0.10 breaks PEFT.
pip uninstall -y torchao
pip install -q transformers==4.46.0 peft accelerate sentencepiece \
    rouge_score bert_score nltk scipy openai 'numpy<2'
```

### 1. Enhance the data (one-time, ~$0.30)

```bash
python experiments/01_enhance_mit_data.py \
    --input  unifiedfl/data/ML_QA_LectureNotes_MIT.json \
    --output /content/drive/MyDrive/unifiedfl_mit_experiment/data/ML_QA_LectureNotes_MIT_enhanced.json
```

The script is resumable — re-running it skips QA pairs that already have
`question_topic` and integer `bloom_level`.

### 2. Build 3-fold splits

```bash
python experiments/02_split_3fold.py \
    --input      /content/drive/MyDrive/unifiedfl_mit_experiment/data/ML_QA_LectureNotes_MIT_enhanced.json \
    --output-dir /content/drive/MyDrive/unifiedfl_mit_experiment/splits \
    --client-id  0 --seed 42 --test-ratio 0.15 --n-folds 3
```

### 3. Train all nine combinations

```bash
python experiments/03_run_three_conditionings.py \
    --splits-dir /content/drive/MyDrive/unifiedfl_mit_experiment/splits \
    --output-dir /content/drive/MyDrive/unifiedfl_mit_experiment/outputs \
    --client-id  0
```

This invokes `unifiedfl/train_client.py` once per (conditioning, fold) pair.
Each run takes ~10–15 min on T4, ~5–8 min on A100, plus ~3–5 min for the heavy
metrics suite. Total wall time ~2–3 hours on T4. Re-running the script picks up
where it left off (skips folds whose `metrics_val.json` already exists).

Useful flags:

| Flag | Purpose |
|---|---|
| `--conditionings baseline` | Only run a subset (default: all three) |
| `--folds 1` | Only run fold 1 |
| `--force` | Re-run even if `metrics_val.json` exists |
| `--no-heavy` | Skip UnifiedQA / DeBERTa metrics — faster eval, fewer numbers |
| `--num-epochs 30` | Cut training time |

### 4. Aggregate

```bash
python experiments/04_aggregate_results.py \
    --output-dir /content/drive/MyDrive/unifiedfl_mit_experiment/outputs \
    --client-id  0 \
    --save-summary /content/drive/MyDrive/unifiedfl_mit_experiment/summary.json
```

Prints per-fold metrics plus per-conditioning means in one table.

## Output layout

```
/content/drive/MyDrive/unifiedfl_mit_experiment/
├── data/ML_QA_LectureNotes_MIT_enhanced.json
├── splits/
│   ├── client_0_test.json
│   ├── client_0_fold{1,2,3}_train.json
│   └── client_0_fold{1,2,3}_val.json
└── outputs/
    ├── baseline/client_0/fold{1,2,3}/{best/, final/, metrics_val.json, ...}
    ├── topic/client_0/fold{1,2,3}/...
    ├── bloom/client_0/fold{1,2,3}/...
    └── summary.json
```

## Where the conditioning lives in code

- [unifiedfl/data/dataset.py](../unifiedfl/data/dataset.py) — `PROMPT_TEMPLATES`,
  `BLOOM_VERBS`, `render_prompt()`, `QADataset(conditioning=...)`
- [unifiedfl/train_client.py](../unifiedfl/train_client.py) —
  `--conditioning {baseline,topic,bloom}` plus the diverse-beam-search eval
  branch (`_evaluate_baseline_diverse`)
