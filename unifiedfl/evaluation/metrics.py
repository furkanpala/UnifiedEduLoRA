"""
Evaluation metrics for EquitableEdu.

Organised by weight / dependency:

  Lightweight — reference-based (always run):
    ROUGE-L, BLEU-4, BERTScore

  Reference-free — need context (run locally, download models on first call):
    Round-Trip Consistency (RTC)      UnifiedQA-v2-T5-small + token-F1
    RAGAS Faithfulness                DeBERTa-v3-small NLI per sentence
    QAFactEval (approx.)              UnifiedQA yes/no per answer claim
    RQUGE (approx.)                   UnifiedQA + BERTScore → [1, 5]

  Local classifier (no API key needed):
    Bloom's BERT classifier           Fine-tuned BERT/RoBERTa text-classification model

  Require OpenAI API key (skip gracefully if not provided):
    RAGAS Answer Relevancy            GPT-4o reverse-Q + cosine similarity
    Bloom's Taxonomy LLM judge        GPT-4o structured JSON output
    LLM QA-quality judge              GPT-4o scores 4 dimensions of QA quality

  Phase-5 only — computed across all three experiments:
    Pairwise Equity Score (PES)
    Knowledge Transfer Index (KTI)
    Anchor Question Equity
    Statistical validation (Wilcoxon, Cohen's d, 95% bootstrap CI)

All heavy models are lazy-loaded and cached in _CACHE so they are
downloaded once per process, not once per call.
"""

from __future__ import annotations

import json
import logging
import re
import string
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger("federated_qa")

_CACHE: Dict[str, Any] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_qa(text: str) -> Tuple[str, str]:
    """Split 'Question: …\\nAnswer: …' into (question, answer)."""
    q = re.search(r"(?i)question:\s*(.*?)(?=\nanswer:|\Z)", text, re.DOTALL)
    a = re.search(r"(?i)answer:\s*(.*?)$", text, re.DOTALL)
    question = q.group(1).strip() if q else ""
    answer   = a.group(1).strip() if a else text.strip()
    return question, answer


def _token_f1(pred: str, ref: str) -> float:
    """Token-level F1 between two strings (lowercased, punctuation stripped)."""
    def _tok(s: str) -> List[str]:
        s = s.lower().translate(str.maketrans("", "", string.punctuation))
        return s.split()
    pt, rt = _tok(pred), _tok(ref)
    if not pt or not rt:
        return 0.0
    common = Counter(pt) & Counter(rt)
    n = sum(common.values())
    if n == 0:
        return 0.0
    p = n / len(pt)
    r = n / len(rt)
    return 2 * p * r / (p + r)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Lightweight reference-based
# ─────────────────────────────────────────────────────────────────────────────

def compute_rouge_l(predictions: List[str], references: List[str]) -> float:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    total = sum(scorer.score(ref, pred)["rougeL"].fmeasure
                for pred, ref in zip(predictions, references))
    return total / max(len(predictions), 1)


def compute_bleu4(predictions: List[str], references: List[str]) -> float:
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    sf = SmoothingFunction().method1
    total = sum(sentence_bleu([ref.split()], pred.split(), smoothing_function=sf)
                for pred, ref in zip(predictions, references))
    return total / max(len(predictions), 1)


def compute_bertscore(
    predictions: List[str],
    references: List[str],
    device: torch.device,
) -> float:
    from bert_score import score as bscore
    _, _, F1 = bscore(
        predictions, references,
        model_type="distilbert-base-uncased",
        device=str(device), verbose=False,
    )
    return F1.mean().item()


def compute_all_metrics(
    predictions: List[str],
    references: List[str],
    device: torch.device,
) -> Dict[str, float]:
    """Lightweight combo: ROUGE-L + BLEU-4 + BERTScore."""
    if not predictions:
        return {"rouge_l": 0.0, "bleu_4": 0.0, "bertscore_f1": 0.0}
    return {
        "rouge_l":      compute_rouge_l(predictions, references),
        "bleu_4":       compute_bleu4(predictions, references),
        "bertscore_f1": compute_bertscore(predictions, references, device),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Round-Trip Consistency (RTC)
# ─────────────────────────────────────────────────────────────────────────────

def _get_unifiedqa(device: torch.device):
    if "unifiedqa" not in _CACHE:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        mid = "allenai/unifiedqa-v2-t5-small-1251000"
        logger.info(f"Loading UnifiedQA from {mid} (one-time download) …")
        # use_fast=False uses the SentencePiece slow tokenizer directly,
        # avoiding the protobuf dependency required to convert it to a fast one.
        tok = AutoTokenizer.from_pretrained(mid, use_fast=False)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(mid).to(device).eval()
        _CACHE["unifiedqa"] = (tok, mdl, device)
    return _CACHE["unifiedqa"]


@torch.no_grad()
def _unifiedqa_answer(question: str, context: str, device: torch.device) -> str:
    tok, mdl, _ = _get_unifiedqa(device)
    # UnifiedQA-v2 input format: "<question> \\n <context>"
    inp = f"{question} \\n {context}"
    enc = tok(inp, return_tensors="pt", max_length=512, truncation=True).to(device)
    out = mdl.generate(**enc, max_new_tokens=64, num_beams=4)
    return tok.decode(out[0], skip_special_tokens=True)


def compute_rtc(
    generated: List[str],
    references: List[str],
    contexts: List[str],
    device: torch.device,
) -> float:
    """
    Round-Trip Consistency.
    RTC(q, a, c) = token_F1(UnifiedQA(q, c), a_ref)
    """
    scores = []
    for gen, ref, ctx in zip(generated, references, contexts):
        gen_q, _   = _parse_qa(gen)
        _,    ref_a = _parse_qa(ref)
        if not gen_q or not ctx:
            scores.append(0.0)
            continue
        qa_ans = _unifiedqa_answer(gen_q, ctx, device)
        scores.append(_token_f1(qa_ans, ref_a))
    return float(np.mean(scores)) if scores else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — RAGAS Faithfulness (NLI-based)
# ─────────────────────────────────────────────────────────────────────────────

def _get_nli_pipeline(device: torch.device):
    if "nli" not in _CACHE:
        from transformers import pipeline as hf_pipeline
        logger.info("Loading DeBERTa-v3-small NLI model (one-time download) …")
        # torch.device("cuda") has .index == None; pass 0 in that case so the
        # HF pipeline puts the model on GPU instead of silently falling back to CPU.
        dev_id = (device.index if device.index is not None else 0) if device.type == "cuda" else -1
        pipe = hf_pipeline(
            "text-classification",
            model="cross-encoder/nli-deberta-v3-small",
            device=dev_id,
        )
        _CACHE["nli"] = pipe
    return _CACHE["nli"]


def compute_faithfulness(
    generated: List[str],
    contexts: List[str],
    device: torch.device,
    openai_api_key: Optional[str] = None,
) -> float:
    """
    RAGAS Faithfulness.
    Decompose answer into sentences (or atomic claims via LLM if api_key given),
    then check DeBERTa-v3-small NLI(context, claim) == ENTAILMENT for each.
    Faithfulness(a, c) = (1/n) * sum(entailed_i)
    """
    from nltk.tokenize import sent_tokenize

    pipe = _get_nli_pipeline(device)

    # Optionally decompose into atomic claims using LLM
    def _decompose(answer: str) -> List[str]:
        if openai_api_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=openai_api_key)
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content":
                        f"Break the following text into a list of short, self-contained "
                        f"atomic factual claims (one sentence each). "
                        f"Return JSON: {{\"claims\": [\"claim1\", \"claim2\", ...]}}\n\n"
                        f"Text: \"{answer}\""}],
                    temperature=0.0, max_tokens=512,
                )
                raw = resp.choices[0].message.content.strip()
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
                return json.loads(raw).get("claims", sent_tokenize(answer))
            except Exception as e:
                logger.warning(f"LLM claim decomposition failed, falling back to sent_tokenize: {e}")
        return sent_tokenize(answer)

    doc_scores = []
    for gen, ctx in zip(generated, contexts):
        _, answer = _parse_qa(gen)
        if not answer or not ctx:
            doc_scores.append(0.0)
            continue
        claims = _decompose(answer)
        if not claims:
            doc_scores.append(0.0)
            continue
        # Batch NLI: premise=ctx, hypothesis=claim
        pairs = [{"text": ctx[:1024], "text_pair": c} for c in claims]
        results = pipe(pairs)
        entailed = [1.0 if "entail" in r["label"].lower() else 0.0 for r in results]
        doc_scores.append(float(np.mean(entailed)))

    return float(np.mean(doc_scores)) if doc_scores else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — QAFactEval (approximation)
# ─────────────────────────────────────────────────────────────────────────────

def compute_qafacteval(
    generated: List[str],
    contexts: List[str],
    device: torch.device,
) -> float:
    """
    QAFactEval (approximation).
    For each sentence s in the answer, ask UnifiedQA:
      q_int = 'Is it true that: {s}?'
      a_hat = UnifiedQA(q_int, context)
    Score = 1.0 if 'yes' in a_hat, else 0.0.
    QAFactEval(a, c) = mean(scores)

    Note: Full QAFactEval uses LERC scoring with question generation from spans.
    This approximation uses yes/no QA as a factual grounding proxy.
    """
    from nltk.tokenize import sent_tokenize

    doc_scores = []
    for gen, ctx in zip(generated, contexts):
        _, answer = _parse_qa(gen)
        if not answer or not ctx:
            doc_scores.append(0.0)
            continue
        sentences = sent_tokenize(answer)
        if not sentences:
            doc_scores.append(0.0)
            continue
        claim_scores = []
        for sent in sentences:
            q = f"Is it true that: {sent}?"
            a_hat = _unifiedqa_answer(q, ctx, device)
            claim_scores.append(1.0 if "yes" in a_hat.lower() else 0.0)
        doc_scores.append(float(np.mean(claim_scores)))

    return float(np.mean(doc_scores)) if doc_scores else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — RQUGE (approximation)
# ─────────────────────────────────────────────────────────────────────────────

def compute_rquge(
    generated: List[str],
    references: List[str],
    contexts: List[str],
    device: torch.device,
) -> float:
    """
    RQUGE (approximation).
    a_c = UnifiedQA(q_gen, context)
    score = BERTScore-F1(a_c, a_ref), scaled to [1, 5].

    Note: Full RQUGE uses a MOCHA-finetuned RoBERTa scorer for the
    quality assessment. This approximation substitutes BERTScore-F1.
    """
    from bert_score import score as bscore

    qa_answers, ref_answers = [], []
    for gen, ref, ctx in zip(generated, references, contexts):
        gen_q, _    = _parse_qa(gen)
        _,     ref_a = _parse_qa(ref)
        if not gen_q or not ctx:
            qa_answers.append("")
            ref_answers.append(ref_a)
            continue
        qa_answers.append(_unifiedqa_answer(gen_q, ctx, device))
        ref_answers.append(ref_a)

    if not qa_answers:
        return 1.0

    _, _, F1 = bscore(
        qa_answers, ref_answers,
        model_type="distilbert-base-uncased",
        device=str(device), verbose=False,
    )
    # Scale [0, 1] → [1, 5], clamped because BertScore can occasionally
    # produce values slightly outside [0, 1].
    f1_mean = float(F1.mean().item())
    f1_mean = max(0.0, min(1.0, f1_mean))
    return 1.0 + 4.0 * f1_mean


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — RAGAS Answer Relevancy (requires OpenAI)
# ─────────────────────────────────────────────────────────────────────────────

def compute_answer_relevancy(
    generated: List[str],
    references: List[str],
    openai_api_key: str,
    n_reverse: int = 3,
) -> float:
    """
    RAGAS Answer Relevancy.
    Generate n reverse questions from the answer using GPT-4o-mini,
    embed all questions with OpenAI text-embedding-3-small,
    return mean cosine similarity between original Q and each reverse Q.
    """
    from openai import OpenAI
    client = OpenAI(api_key=openai_api_key)

    def _embed(texts: List[str]) -> np.ndarray:
        resp = client.embeddings.create(
            model="text-embedding-3-small", input=texts
        )
        return np.array([d.embedding for d in resp.data])

    def _reverse_questions(answer: str) -> List[str]:
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content":
                    f"Given the following answer, generate {n_reverse} different "
                    f"questions that this answer would naturally respond to. "
                    f"Return JSON: {{\"questions\": [\"q1\", \"q2\", ...]}}\n\n"
                    f"Answer: \"{answer}\""}],
                temperature=0.7, max_tokens=256,
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            return json.loads(raw).get("questions", [])
        except Exception as e:
            logger.warning(f"Reverse-question generation failed: {e}")
            return []

    scores = []
    for gen, ref in zip(generated, references):
        gen_q, gen_a = _parse_qa(gen)
        if not gen_q or not gen_a:
            scores.append(0.0)
            continue
        rev_qs = _reverse_questions(gen_a)
        if not rev_qs:
            scores.append(0.0)
            continue
        all_qs = [gen_q] + rev_qs
        try:
            embs = _embed(all_qs)
            orig_emb = embs[0]
            sims = [_cosine(orig_emb, embs[i]) for i in range(1, len(embs))]
            scores.append(float(np.mean(sims)))
        except Exception as e:
            logger.warning(f"Answer-relevancy embedding failed: {e}")
            scores.append(0.0)

    return float(np.mean(scores)) if scores else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Section 7a — Bloom's BERT classifier (local, no API key required)
# ─────────────────────────────────────────────────────────────────────────────

def _get_blooms_classifier(device: torch.device, model_name: str):
    key = f"blooms_cls_{model_name}"
    if key not in _CACHE:
        from transformers import pipeline as hf_pipeline
        logger.info(f"Loading Bloom's classifier from {model_name} …")
        # torch.device("cuda") has .index == None; pass 0 in that case so the
        # HF pipeline puts the model on GPU instead of silently falling back to CPU.
        dev_id = (device.index if device.index is not None else 0) if device.type == "cuda" else -1
        _CACHE[key] = hf_pipeline(
            "text-classification",
            model=model_name,
            tokenizer=model_name,
            device=dev_id,
            truncation=True,
            max_length=512,
        )
    return _CACHE[key]


def compute_blooms_classifier(
    generated: List[str],
    device: torch.device,
    model_name: str = "cip29/bert-blooms-taxonomy-classifier",
) -> Dict[str, Any]:
    """
    Bloom's Taxonomy classification using a fine-tuned BERT/RoBERTa model.

    The model must be a HuggingFace text-classification model that maps
    educational questions to Bloom's cognitive levels (1–6).

    Label format is handled automatically:
      - "LABEL_0" … "LABEL_5"  → levels 1–6 (HuggingFace default numbering)
      - "Level 1" … "Level 6"  → levels 1–6 (human-readable)
      - Any string containing a single digit → that digit as the level

    Default model: cip29/bert-blooms-taxonomy-classifier (BertForSequenceClassification, 6 labels)
    Override with --blooms-model <hf_model_id> in train_client.py.

    Note: different fine-tuned classifiers may use different label orderings.
    Verify that LABEL_0 corresponds to the lowest cognitive level (Remember).

    Returns:
        {
          "per_sample": [{"level": int|None, "score": float, "evs": float|None}, ...],
          "distribution": {1: int, ..., 6: int},
          "evs_mean": float,   EVS = (level - 1) / 5 in [0, 1]
        }
    """
    classifier = _get_blooms_classifier(device, model_name)
    per_sample  = []
    distribution = {k: 0 for k in range(1, 7)}

    for gen in generated:
        gen_q, _ = _parse_qa(gen)
        if not gen_q:
            per_sample.append({"level": None, "score": 0.0, "evs": None})
            continue
        try:
            # Default text-classification pipeline returns [{"label": ..., "score": ...}]
            # for a single string input — a flat list of one dict, not a nested list.
            out   = classifier(gen_q)[0]
            label = out["label"]
            score = float(out["score"])

            if label.upper().startswith("LABEL_"):
                level = int(label.split("_")[-1]) + 1
            else:
                digits = re.findall(r"\d+", label)
                level  = int(digits[0]) if digits else None

            if level is None or not (1 <= level <= 6):
                per_sample.append({"level": None, "score": score, "evs": None})
                continue

            evs = (level - 1) / 5.0
            per_sample.append({"level": level, "score": score, "evs": evs})
            distribution[level] += 1
        except Exception as e:
            logger.warning(f"Bloom's classifier failed: {e}")
            per_sample.append({"level": None, "score": 0.0, "evs": None})

    valid_evs = [s["evs"] for s in per_sample if s["evs"] is not None]
    return {
        "per_sample":   per_sample,
        "distribution": distribution,
        "evs_mean":     float(np.mean(valid_evs)) if valid_evs else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Section 7b — Bloom's Taxonomy LLM judge (requires OpenAI)
# ─────────────────────────────────────────────────────────────────────────────

def compute_blooms_llm(
    generated: List[str],
    openai_api_key: str,
) -> Dict[str, Any]:
    """
    Bloom's Taxonomy LLM judge using GPT-4o-mini.

    Returns:
        {
          "per_sample": [{"level": int, "reason": str, "evs": float}, ...],
          "distribution": {1: int, ..., 6: int},
          "evs_mean": float,       EVS = (level - 1) / 5 in [0, 1]
        }
    """
    from openai import OpenAI
    client = OpenAI(api_key=openai_api_key)

    bloom_desc = (
        "1=Remember (define, list, state), "
        "2=Understand (explain, describe, summarise), "
        "3=Apply (solve, compute, demonstrate), "
        "4=Analyse (compare, contrast, distinguish), "
        "5=Evaluate (critique, assess, justify), "
        "6=Create (design, propose, formulate)"
    )

    per_sample = []
    distribution = {k: 0 for k in range(1, 7)}

    for gen in generated:
        gen_q, _ = _parse_qa(gen)
        if not gen_q:
            per_sample.append({"level": None, "reason": "no question parsed", "evs": None})
            continue
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content":
                    f"Classify the following educational question according to "
                    f"Bloom's Taxonomy ({bloom_desc}). "
                    f"Return JSON: {{\"level\": <1-6>, \"reason\": \"<one sentence>\"}}\n\n"
                    f"Question: \"{gen_q}\""}],
                temperature=0.0, max_tokens=128,
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            parsed = json.loads(raw)
            level  = int(parsed["level"])
            reason = parsed.get("reason", "")
            evs    = (level - 1) / 5.0
            per_sample.append({"level": level, "reason": reason, "evs": evs})
            if 1 <= level <= 6:
                distribution[level] += 1
        except Exception as e:
            logger.warning(f"Bloom's LLM judge failed for question: {e}")
            per_sample.append({"level": None, "reason": str(e), "evs": None})

    valid_evs = [s["evs"] for s in per_sample if s["evs"] is not None]
    return {
        "per_sample":   per_sample,
        "distribution": distribution,
        "evs_mean":     float(np.mean(valid_evs)) if valid_evs else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Section 7c — LLM QA quality judge (requires OpenAI)
# ─────────────────────────────────────────────────────────────────────────────

LLM_JUDGE_PROMPT = """You are an expert evaluator of educational question-answer pairs.

Given a CONTEXT (educational text) and a generated QUESTION-ANSWER pair, score \
the QA pair on FOUR dimensions, each on an integer 1-5 scale (1 = very poor, 5 = excellent).

CONTEXT:
\"\"\"{context}\"\"\"

GENERATED QA:
Question: {question}
Answer: {answer}

Score each dimension. Use the rubrics below — be strict.

1. context_grounding: Is the question answerable using ONLY information in the context, \
and are the facts in the answer supported by the context?
   1 = Question or answer requires information not in the context (off-topic or hallucinated).
   3 = Partially grounded — some facts come from the context, but key parts are external.
   5 = Fully grounded — both question and answer are entirely supported by the context.

2. educational_value: Does this QA pair test meaningful understanding of the material?
   1 = Trivial, vague, or pointless (e.g. \"What does the text say?\", a yes/no with no insight).
   3 = Tests basic recall or surface-level facts, but not deeper understanding.
   5 = Tests genuine comprehension, application, analysis, or evaluation of the concept.

3. answer_correctness: Given the context, is the answer factually correct and complete?
   1 = Answer is wrong, contradicts the context, or contains hallucinations.
   3 = Answer is partially correct but missing key information or contains minor errors.
   5 = Answer is fully correct and complete given the context.

4. answer_relevance: Does the answer actually address the question that was asked?
   1 = Answer ignores or evades the question.
   3 = Answer is tangentially related — touches on the topic but does not directly answer.
   5 = Answer directly and clearly addresses what was asked.

Return ONLY a single JSON object in this exact schema, with no extra text or markdown:
{{
  "context_grounding": <int 1-5>,
  "educational_value": <int 1-5>,
  "answer_correctness": <int 1-5>,
  "answer_relevance": <int 1-5>,
  "justification": "<one or two sentences explaining the lowest-scoring dimension>"
}}"""


def compute_llm_judge(
    generated: List[str],
    contexts: List[str],
    openai_api_key: str,
    model: str = "gpt-4o",
) -> Dict[str, Any]:
    """
    LLM-as-judge QA quality evaluator.

    For each (context, generated_qa) pair, the judge scores four dimensions on
    a 1-5 integer scale:
        - context_grounding   : Q is answerable from the context, A is supported by it
        - educational_value   : Q tests meaningful understanding (not trivia)
        - answer_correctness  : A is factually correct given the context
        - answer_relevance    : A actually addresses the question that was asked

    A per-sample `overall` score is the mean of the four dimensions normalised
    to [0, 1] via (mean - 1) / 4.

    Args:
        generated:      list of model output strings "Question: …\\nAnswer: …"
        contexts:       list of source contexts (same length as generated)
        openai_api_key: OpenAI API key
        model:          OpenAI chat model (default: gpt-4o; gpt-4o-mini is cheaper)

    Returns:
        {
          "per_sample": [
            {
              "context_grounding":  int|None,
              "educational_value":  int|None,
              "answer_correctness": int|None,
              "answer_relevance":   int|None,
              "overall":            float|None,   # in [0, 1]
              "justification":      str,
            }, ...
          ],
          "context_grounding_mean":  float,       # in [1, 5]
          "educational_value_mean":  float,
          "answer_correctness_mean": float,
          "answer_relevance_mean":   float,
          "overall_mean":            float,       # in [0, 1]
        }
    """
    from openai import OpenAI
    client = OpenAI(api_key=openai_api_key)

    per_sample: List[Dict[str, Any]] = []
    dims = ("context_grounding", "educational_value",
            "answer_correctness", "answer_relevance")

    for gen, ctx in zip(generated, contexts):
        gen_q, gen_a = _parse_qa(gen)
        if not gen_q or not gen_a or not ctx:
            per_sample.append({
                "context_grounding":  None,
                "educational_value":  None,
                "answer_correctness": None,
                "answer_relevance":   None,
                "overall":            None,
                "justification":      "could not parse question/answer/context",
            })
            continue

        prompt = LLM_JUDGE_PROMPT.format(
            context=ctx, question=gen_q, answer=gen_a
        )
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=512,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            parsed = json.loads(raw)

            scores = {d: int(parsed[d]) for d in dims}
            # Clamp each score to [1, 5]
            scores = {d: max(1, min(5, v)) for d, v in scores.items()}
            mean = sum(scores.values()) / len(dims)
            overall = (mean - 1.0) / 4.0  # → [0, 1]

            per_sample.append({
                **scores,
                "overall":       overall,
                "justification": str(parsed.get("justification", "")),
            })
        except Exception as e:
            logger.warning(f"LLM judge failed for one sample: {e}")
            per_sample.append({
                "context_grounding":  None,
                "educational_value":  None,
                "answer_correctness": None,
                "answer_relevance":   None,
                "overall":            None,
                "justification":      f"judge error: {e}",
            })

    # Aggregate across valid samples
    def _mean(key: str) -> float:
        vals = [s[key] for s in per_sample if s.get(key) is not None]
        return float(np.mean(vals)) if vals else 0.0

    return {
        "per_sample":              per_sample,
        "context_grounding_mean":  _mean("context_grounding"),
        "educational_value_mean":  _mean("educational_value"),
        "answer_correctness_mean": _mean("answer_correctness"),
        "answer_relevance_mean":   _mean("answer_relevance"),
        "overall_mean":            _mean("overall"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Section 8 — Statistical validation (Phase 5)
# ─────────────────────────────────────────────────────────────────────────────

def compute_statistical_validation(
    scores_indiv: List[float],
    scores_uni: List[float],
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Paired Wilcoxon signed-rank test + Cohen's d + 95% bootstrap CI.

    H0: mu(uni) == mu(indiv)
    H1: mu(uni) >  mu(indiv)

    Returns:
        wilcoxon_p, cohen_d, ci_95 [lower, upper], reject_h0
    """
    from scipy.stats import wilcoxon

    a = np.array(scores_indiv, dtype=float)
    b = np.array(scores_uni,   dtype=float)

    # Paired Wilcoxon (one-sided, greater)
    try:
        stat, p_val = wilcoxon(b, a, alternative="greater")
    except Exception:
        stat, p_val = float("nan"), float("nan")

    # Cohen's d (paired)
    diff = b - a
    cohen_d = float(diff.mean() / (diff.std(ddof=1) + 1e-10))

    # Bootstrap 95% CI on mean difference
    rng = np.random.default_rng(42)
    boot_means = np.array([
        rng.choice(diff, size=len(diff), replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    ci_lower, ci_upper = float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))

    return {
        "wilcoxon_statistic": float(stat),
        "wilcoxon_p":         float(p_val),
        "reject_h0":          bool(p_val < alpha),
        "cohen_d":            cohen_d,
        "ci_95":              [ci_lower, ci_upper],
        "mean_indiv":         float(a.mean()),
        "mean_uni":           float(b.mean()),
        "mean_delta":         float(diff.mean()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Section 9 — Equity metrics (Phase 5)
# ─────────────────────────────────────────────────────────────────────────────

def compute_pairwise_equity(
    Q_k: float,
    Q_l: float,
    C_k: float,
    C_l: float,
    R_k: float,
    R_l: float,
    epsilon: float = 1e-6,
) -> float:
    """
    Pairwise Equity Score between institutions k and l.
    E_{kl} = (Q_k + Q_l) / (|C_k - C_l| + |R_k - R_l| + epsilon)
    where C = curricular richness, R = relative model capacity.
    """
    return (Q_k + Q_l) / (abs(C_k - C_l) + abs(R_k - R_l) + epsilon)


def compute_kti(Q_l_uni: float, Q_l_indiv: float) -> float:
    """
    Knowledge Transfer Index for institution l receiving from k.
    KTI(l←k) = Q_l^{uni}(T_{k\\l}) - Q_l^{indiv}(T_{k\\l})
    where T_{k\\l} are test chunks covering topics only in k's curriculum.
    """
    return Q_l_uni - Q_l_indiv


def compute_anchor_equity(
    anchor_question: str,
    answers: Dict[str, str],
    openai_api_key: str,
) -> Dict[str, Any]:
    """
    Anchor Question Equity.
    For a fixed anchor question q*, score each institution's answer using GPT-4o.
    Goal: s_l^{uni} ≈ s_k^{indiv} > s_l^{indiv}

    Args:
        anchor_question: the fixed anchor question q*
        answers: dict mapping label → answer text
                 e.g. {"k_indiv": "...", "l_indiv": "...", "l_uni": "..."}
        openai_api_key: OpenAI key for LLM judge

    Returns:
        dict mapping label → {"score": float, "reason": str}
    """
    from openai import OpenAI
    client = OpenAI(api_key=openai_api_key)

    results = {}
    for label, answer in answers.items():
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content":
                    f"Score the following answer to the question on a scale of 1-10 "
                    f"for accuracy, completeness, and clarity. "
                    f"Return JSON: {{\"score\": <1-10>, \"reason\": \"<one sentence>\"}}\n\n"
                    f"Question: \"{anchor_question}\"\n"
                    f"Answer: \"{answer}\""}],
                temperature=0.0, max_tokens=128,
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            parsed = json.loads(raw)
            results[label] = {"score": float(parsed["score"]), "reason": parsed.get("reason", "")}
        except Exception as e:
            results[label] = {"score": None, "reason": str(e)}

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Section 10 — Comprehensive combo (individual training evaluation)
# ─────────────────────────────────────────────────────────────────────────────

def compute_comprehensive_metrics(
    generated: List[str],
    references: List[str],
    contexts: List[str],
    device: torch.device,
    openai_api_key: Optional[str] = None,
    run_heavy: bool = True,
    blooms_model: Optional[str] = "cip29/bert-blooms-taxonomy-classifier",
) -> Dict[str, Any]:
    """
    Run all available metrics and return a single dict.

    Always computed:
        rouge_l, bleu_4, bertscore_f1

    Computed when blooms_model is provided (local, no API key needed):
        blooms_cls_distribution, blooms_cls_evs_mean, blooms_cls_per_sample

    Computed when run_heavy=True (require UnifiedQA + DeBERTa download):
        rtc, faithfulness, qafacteval, rquge

    Computed when openai_api_key is provided:
        answer_relevancy
        blooms_llm_distribution, blooms_llm_evs_mean, blooms_llm_per_sample
        llm_judge_*  (context_grounding_mean, educational_value_mean,
                      answer_correctness_mean, answer_relevance_mean,
                      overall_mean, per_sample)

    Args:
        generated:      model output strings "Question: …\\nAnswer: …"
        references:     gold label strings in same format
        contexts:       source contexts (same length)
        device:         torch device
        openai_api_key: optional OpenAI API key
        run_heavy:      if False, skip metrics that require additional model downloads
        blooms_model:   HuggingFace model ID for the local Bloom's classifier;
                        set to None to skip

    Returns:
        flat dict of metric_name → value
    """
    if not generated:
        return {}

    results: Dict[str, Any] = {}

    # ── Lightweight ───────────────────────────────────────────────────────────
    logger.info("Computing ROUGE-L, BLEU-4, BERTScore …")
    results.update(compute_all_metrics(generated, references, device))

    # ── Local Bloom's classifier ──────────────────────────────────────────────
    if blooms_model:
        logger.info(f"Computing Bloom's classifier ({blooms_model}) …")
        try:
            blooms_cls = compute_blooms_classifier(generated, device, blooms_model)
            results["blooms_cls_distribution"] = blooms_cls["distribution"]
            results["blooms_cls_evs_mean"]     = blooms_cls["evs_mean"]
            results["blooms_cls_per_sample"]   = blooms_cls["per_sample"]
        except Exception as e:
            logger.warning(f"Bloom's classifier skipped: {e}")

    if run_heavy:
        # ── RTC ───────────────────────────────────────────────────────────────
        logger.info("Computing Round-Trip Consistency (RTC) …")
        results["rtc"] = compute_rtc(generated, references, contexts, device)

        # ── Faithfulness ──────────────────────────────────────────────────────
        logger.info("Computing RAGAS Faithfulness …")
        results["faithfulness"] = compute_faithfulness(
            generated, contexts, device, openai_api_key
        )

        # ── QAFactEval ────────────────────────────────────────────────────────
        logger.info("Computing QAFactEval (approx.) …")
        results["qafacteval"] = compute_qafacteval(generated, contexts, device)

        # ── RQUGE ────────────────────────────────────────────────────────────
        logger.info("Computing RQUGE (approx.) …")
        results["rquge"] = compute_rquge(generated, references, contexts, device)

    # ── OpenAI-dependent ─────────────────────────────────────────────────────
    if openai_api_key:
        logger.info("Computing RAGAS Answer Relevancy …")
        results["answer_relevancy"] = compute_answer_relevancy(
            generated, references, openai_api_key
        )

        logger.info("Computing Bloom's Taxonomy (LLM judge) …")
        blooms_llm = compute_blooms_llm(generated, openai_api_key)
        results["blooms_llm_distribution"] = blooms_llm["distribution"]
        results["blooms_llm_evs_mean"]     = blooms_llm["evs_mean"]
        results["blooms_llm_per_sample"]   = blooms_llm["per_sample"]

        logger.info("Computing LLM QA-quality judge …")
        judge = compute_llm_judge(generated, contexts, openai_api_key)
        results["llm_judge_context_grounding_mean"]  = judge["context_grounding_mean"]
        results["llm_judge_educational_value_mean"]  = judge["educational_value_mean"]
        results["llm_judge_answer_correctness_mean"] = judge["answer_correctness_mean"]
        results["llm_judge_answer_relevance_mean"]   = judge["answer_relevance_mean"]
        results["llm_judge_overall_mean"]            = judge["overall_mean"]
        results["llm_judge_per_sample"]              = judge["per_sample"]
    else:
        logger.info(
            "Skipping Answer Relevancy and Bloom's LLM judge "
            "(no openai_api_key provided)"
        )

    return results
