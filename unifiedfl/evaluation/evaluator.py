from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from data.dataset import QADataset, render_prompt, TARGET_TEMPLATE
from evaluation.metrics import compute_all_metrics
from utils.logging_utils import print_qualitative_result, print_quantitative_table

logger = logging.getLogger("federated_qa")

# ── Hardcoded qualitative contexts ────────────────────────────────────────────
# These 9 contexts never appear in any train/val/test split.

# Each entry is (context_text, topic_label) so qualitative_eval works for all
# conditioning modes without a live model to suggest topics.
QUALITATIVE_CONTEXTS: Dict[int, List[tuple]] = {
    # Client 0 — BART / MIT lecture style
    0: [
        (
            "Gradient descent is an iterative optimization algorithm used to minimize a loss function "
            "by updating parameters in the direction of the steepest descent. At each step the parameters "
            "are updated as theta = theta - alpha * gradient, where alpha is the learning rate. Choosing "
            "an appropriate learning rate is critical: too large causes divergence, too small leads to "
            "very slow convergence. Variants such as stochastic gradient descent and mini-batch gradient "
            "descent are commonly used in practice to reduce computational cost per iteration.",
            "learning rate scheduling",
        ),
        (
            "Overfitting occurs when a model learns the noise in the training data rather than the "
            "underlying pattern, resulting in poor generalization to unseen data. Regularization techniques "
            "such as L1 (Lasso) and L2 (Ridge) penalties add a term proportional to the magnitude of the "
            "weights to the loss function, discouraging the model from assigning large weights to any "
            "single feature. Dropout is another widely used technique that randomly sets activations to "
            "zero during training, acting as an ensemble of many sub-networks.",
            "dropout regularization",
        ),
        (
            "Principal Component Analysis (PCA) is an unsupervised dimensionality reduction technique "
            "that projects data onto a lower-dimensional subspace by finding the directions (principal "
            "components) of maximum variance. The first principal component captures the most variance, "
            "the second captures the most remaining variance orthogonal to the first, and so on. PCA "
            "is useful for visualization, noise reduction, and as a preprocessing step before training "
            "supervised learning models on high-dimensional data.",
            "principal components",
        ),
    ],
    # Client 1 — T5 / Stanford CS229 lecture style
    1: [
        (
            "The support vector machine (SVM) finds the hyperplane that maximizes the margin between "
            "two classes in a binary classification problem. The margin is defined as the distance between "
            "the hyperplane and the nearest data points from each class, called support vectors. SVMs can "
            "handle non-linearly separable data through the kernel trick, which implicitly maps inputs "
            "into a higher-dimensional feature space where a linear separator exists. Common kernels "
            "include polynomial, radial basis function (RBF), and sigmoid.",
            "kernel trick",
        ),
        (
            "The Expectation-Maximization (EM) algorithm is an iterative method for finding maximum "
            "likelihood estimates of parameters in statistical models with latent variables. The E-step "
            "computes the expected value of the complete-data log-likelihood with respect to the current "
            "parameter estimates. The M-step then maximizes this expectation to obtain new parameter "
            "estimates. EM is guaranteed to increase the observed data log-likelihood at every iteration, "
            "converging to a local maximum.",
            "latent variable estimation",
        ),
        (
            "Bias and variance represent the two main sources of error in supervised learning. Bias "
            "is the error introduced by approximating a complex problem with a simpler model and leads "
            "to systematic underfitting. Variance is the error from sensitivity to small fluctuations "
            "in the training set and leads to overfitting. The bias-variance trade-off states that "
            "decreasing model complexity increases bias while decreasing variance, and vice versa. "
            "Cross-validation is commonly used to find the optimal model complexity.",
            "bias-variance tradeoff",
        ),
    ],
    # Client 2 — LED / ML paper abstract style
    2: [
        (
            "We propose a novel self-supervised pretraining objective for transformer-based language "
            "models that leverages contrastive learning on document-level representations. Our method "
            "constructs positive pairs by applying stochastic sentence permutations and span masking "
            "to the same document, while treating representations from different documents as negatives. "
            "Empirical evaluation on six downstream benchmarks demonstrates that our approach achieves "
            "state-of-the-art performance, outperforming RoBERTa by 2.3 points on average while requiring "
            "40% less pretraining compute.",
            "contrastive pretraining",
        ),
        (
            "Sparse attention mechanisms have emerged as a promising approach to scaling transformers "
            "to long sequences by reducing the quadratic complexity of full self-attention. We introduce "
            "an adaptive sparse attention pattern that dynamically selects a small subset of tokens for "
            "each query based on a learned routing function. Unlike fixed patterns such as local windows "
            "or strided attention, our method allows the model to attend to globally relevant tokens "
            "regardless of their position. Experiments on long-document summarization show consistent "
            "improvements over existing sparse attention baselines.",
            "sparse attention patterns",
        ),
        (
            "Federated learning enables training machine learning models across decentralized data "
            "sources without sharing raw data, addressing privacy concerns in sensitive domains such "
            "as healthcare and finance. The canonical FedAvg algorithm averages local model updates "
            "from participating clients, but suffers from client drift when data distributions are "
            "heterogeneous. Recent work has explored momentum-based correction, adaptive aggregation, "
            "and personalization strategies to mitigate these issues. However, the case of structurally "
            "heterogeneous models — where clients train different architectures — remains underexplored.",
            "federated aggregation",
        ),
    ],
}


class Evaluator:
    """
    Handles both qualitative and quantitative evaluation for all clients.

    During evaluation, FiLM hooks must be active: graphs are refreshed,
    GNN is run, hooks are registered, inference is performed, hooks removed.
    """

    def __init__(
        self,
        clients: List[Any],  # List[FederatedClient]
        config: Any,         # Config
        device: torch.device,
    ) -> None:
        self.clients = clients
        self.config = config
        self.device = device

    # ── Qualitative evaluation ────────────────────────────────────────────────

    @torch.no_grad()
    def qualitative_eval(self, round_idx: int) -> None:
        """Generate QA pairs for the 9 hardcoded contexts and print results."""
        for client in self.clients:
            cid = client.client_id
            contexts = QUALITATIVE_CONTEXTS.get(cid, [])
            model_name = client.client_model.model_name

            # Activate FiLM hooks for generation
            self._activate_hooks(client)

            for ctx, topic in contexts:
                sample = {"context": ctx, "question_topic": topic, "bloom_level": 2}
                input_text = render_prompt(sample, self.config.conditioning)
                enc = client.client_model.tokenizer(
                    input_text,
                    max_length=self.config.max_input_len,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)

                generated_ids = client.client_model.generate(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    num_beams=4,
                    max_new_tokens=self.config.max_target_len,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                )
                generated_text = client.client_model.tokenizer.decode(
                    generated_ids[0], skip_special_tokens=True
                )
                print_qualitative_result(round_idx, cid, model_name, ctx, generated_text)

            self._deactivate_hooks(client)

    # ── Quantitative evaluation ───────────────────────────────────────────────

    @torch.no_grad()
    def quantitative_eval(
        self,
        round_idx: int,
        global_test_samples: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """
        Evaluate each client on its local test set and the global test set.

        Returns:
            dict mapping client_id → {local: metrics, global: metrics}
        """
        results_table: List[Dict[str, Any]] = []
        all_results: Dict[int, Dict[str, Dict[str, float]]] = {}

        # Build global test DataLoader (tokenized with client 0's tokenizer as reference,
        # but we evaluate each client independently with its own tokenizer)
        for client in self.clients:
            cid = client.client_id
            model_name_short = f"C{cid}/{client.client_model.model_family.upper()}"

            self._activate_hooks(client)

            local_metrics = self._evaluate_on_samples(client, client.test_samples)
            global_metrics = self._evaluate_on_samples(client, global_test_samples)

            self._deactivate_hooks(client)

            all_results[cid] = {"local": local_metrics, "global": global_metrics}

            results_table.append({
                "label": f"{model_name_short} (local)",
                **local_metrics,
            })
            results_table.append({
                "label": f"{model_name_short}(global)",
                **global_metrics,
            })

            logger.info(
                f"Round {round_idx} | Client {cid} local  ROUGE-L={local_metrics['rouge_l']:.3f} "
                f"BLEU={local_metrics['bleu_4']:.3f} BERT={local_metrics['bertscore_f1']:.3f}"
            )
            logger.info(
                f"Round {round_idx} | Client {cid} global ROUGE-L={global_metrics['rouge_l']:.3f} "
                f"BLEU={global_metrics['bleu_4']:.3f} BERT={global_metrics['bertscore_f1']:.3f}"
            )

        print_quantitative_table(round_idx, results_table)
        return all_results

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _activate_hooks(self, client: Any) -> None:
        """Refresh graph, run GNN, register FiLM hooks."""
        from models.graph_constructor import refresh_graph_features

        refresh_graph_features(
            client.graph_data,
            client.client_model.model,
            self.config.lora.lora_alpha,
            self.config.lora.r,
        )
        node_emb, graph_emb = client.gnn(client.graph_data.data)
        client.film_adapter.register_hooks(
            client.client_model.model,
            node_emb,
            client.graph_data.layer_to_node_idx,
            graph_emb,
        )

    def _deactivate_hooks(self, client: Any) -> None:
        client.film_adapter.remove_hooks()

    def collect_predictions(
        self,
        client: Any,
        samples: List[Dict[str, str]],
    ) -> tuple:
        """
        Generate predictions for `samples` (FiLM hooks must already be
        active — see _activate_hooks). Returns (preds, refs, contexts).

        contexts are the raw 'context' fields from samples, in the same order
        as preds — required by compute_comprehensive_metrics for RTC,
        Faithfulness, QAFactEval, RQUGE.
        """
        if not samples:
            return [], [], []

        tokenizer = client.client_model.tokenizer
        dataset = QADataset(
            samples,
            tokenizer,
            self.config.max_input_len,
            self.config.max_target_len,
            conditioning=self.config.conditioning,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=(self.device.type == "cuda"),
        )

        predictions: List[str] = []
        references: List[str] = []

        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            raw_labels = batch["labels"]

            generated_ids = client.client_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=4,
                max_new_tokens=self.config.max_target_len,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

            for gen_ids in generated_ids:
                predictions.append(
                    tokenizer.decode(gen_ids, skip_special_tokens=True)
                )

            for label_ids in raw_labels:
                label_ids = label_ids.masked_fill(label_ids == -100, tokenizer.pad_token_id)
                references.append(tokenizer.decode(label_ids, skip_special_tokens=True))

        contexts = [s.get("context", "") for s in samples]
        return predictions, references, contexts

    def _evaluate_on_samples(
        self,
        client: Any,
        samples: List[Dict[str, str]],
    ) -> Dict[str, float]:
        """Generate predictions for `samples` and compute the lightweight
        3-metric combo (ROUGE-L / BLEU-4 / BERTScore)."""
        if not samples:
            return {"rouge_l": 0.0, "bleu_4": 0.0, "bertscore_f1": 0.0}
        preds, refs, _ = self.collect_predictions(client, samples)
        return compute_all_metrics(preds, refs, self.device)
