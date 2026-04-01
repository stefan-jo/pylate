from __future__ import annotations

from typing import Callable, Iterable, Literal

import torch

from ..models import ColBERT
from ..scores import colbert_kd_scores
from .contrastive import extract_skiplist_mask


class Distillation(torch.nn.Module):
    """Distillation loss for ColBERT model. The loss is computed with respect to the format of SentenceTransformer library.

    Parameters
    ----------
    model
        SentenceTransformer model.
    score_metric
        Function that returns a score between two sequences of embeddings.
    size_average
        Average by the size of the mini-batch or perform sum.
    normalize_scores
        Whether to min-max normalize student scores per query before KL-divergence.
    pool_factor
        Document pooling factor used before scoring. Set to ``1`` to disable pooling.
    pool_factors
        Optional list of pooling factors. If provided, one factor is sampled uniformly per
        forward pass.
    pool_factor_seed
        Seed used for deterministic pool-factor sampling.
    pool_method
        Pooling strategy for documents when ``pool_factor > 1``.
        Supported values: ``"hierarchical"``, ``"span"``, ``"kmeans"``.
    protected_tokens
        Number of leading document tokens excluded from pooling.
    use_sklearn
        Whether to use sklearn backend for k-means pooling.

    Examples
    --------
    >>> from pylate import models, losses

    >>> model = models.ColBERT(
    ...     model_name_or_path="sentence-transformers/all-MiniLM-L6-v2", device="cpu"
    ... )

    >>> distillation = losses.Distillation(model=model)

    >>> query = model.tokenize([
    ...     "fruits are healthy.",
    ... ], is_query=True)

    >>> documents = model.tokenize([
    ...     "fruits are good for health.",
    ...     "fruits are bad for health."
    ... ], is_query=False)

    >>> sentence_features = [query, documents]

    >>> labels = torch.tensor([
    ...     [0.7, 0.3],
    ... ], dtype=torch.float32)

    >>> loss = distillation(sentence_features=sentence_features, labels=labels)

    >>> assert isinstance(loss.item(), float)
    """

    def __init__(
        self,
        model: ColBERT,
        score_metric: Callable = colbert_kd_scores,
        size_average: bool = True,
        normalize_scores: bool = True,
        pool_factor: int = 1,
        pool_factors: list[int] | None = None,
        pool_factor_seed: int | None = None,
        pool_method: Literal["hierarchical", "span", "kmeans"] = "hierarchical",
        protected_tokens: int = 1,
        use_sklearn: bool = False,
    ) -> None:
        super(Distillation, self).__init__()
        if pool_factors is not None and len(pool_factors) == 0:
            raise ValueError("pool_factors must be non-empty when provided.")
        normalized_pool_factors = (
            [int(factor) for factor in pool_factors]
            if pool_factors is not None
            else [pool_factor]
        )
        if any(factor < 1 for factor in normalized_pool_factors):
            raise ValueError("All pool factors must be >= 1.")
        if protected_tokens < 0:
            raise ValueError("protected_tokens must be >= 0.")
        if pool_method not in {"hierarchical", "span", "kmeans"}:
            raise ValueError(
                "pool_method must be one of: hierarchical, span, kmeans."
            )

        self.score_metric = score_metric
        self.model = model
        self.loss_function = torch.nn.KLDivLoss(
            reduction="batchmean" if size_average else "sum", log_target=True
        )
        self.normalize_scores = normalize_scores
        self.pool_factors = normalized_pool_factors
        # Backward-compatible attribute retained for external callers reading a single value.
        self.pool_factor = self.pool_factors[0]
        self.pool_method = pool_method
        self.protected_tokens = protected_tokens
        self.use_sklearn = use_sklearn
        self._pool_factor_rng = torch.Generator()
        if pool_factor_seed is None:
            pool_factor_seed = torch.initial_seed()
        self._pool_factor_rng.manual_seed(pool_factor_seed)
        self.last_pool_factor = self.pool_factors[0]

    def _sample_pool_factor(self) -> int:
        if len(self.pool_factors) == 1:
            return self.pool_factors[0]
        sampled_idx = torch.randint(
            low=0,
            high=len(self.pool_factors),
            size=(1,),
            generator=self._pool_factor_rng,
        ).item()
        return self.pool_factors[sampled_idx]

    def _get_model_attr(self, attr_name: str):
        if hasattr(self.model, attr_name):
            return getattr(self.model, attr_name)
        return getattr(self.model.module, attr_name)

    def _pool_documents(
        self,
        documents_embeddings: torch.Tensor,
        documents_mask: torch.Tensor,
        pool_factor: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model_for_pooling = self.model
        if not hasattr(model_for_pooling, "_pool_document_embeddings"):
            model_for_pooling = self.model.module

        flattened_documents = []
        for document_embeddings, document_mask in zip(
            documents_embeddings, documents_mask
        ):
            selected_embeddings = document_embeddings[document_mask]
            if selected_embeddings.size(0) == 0:
                selected_embeddings = document_embeddings[:1]
            flattened_documents.append(selected_embeddings)

        pooled_documents = model_for_pooling._pool_document_embeddings(
            documents_embeddings=flattened_documents,
            pool_factor=pool_factor,
            protected_tokens=self.protected_tokens,
            pool_method=self.pool_method,
            use_sklearn=self.use_sklearn,
        )
        pooled_embeddings = torch.nn.utils.rnn.pad_sequence(
            pooled_documents,
            batch_first=True,
            padding_value=0.0,
        )
        pooled_mask = torch.zeros(
            pooled_embeddings.shape[:2],
            dtype=torch.bool,
            device=pooled_embeddings.device,
        )
        for idx, pooled_document in enumerate(pooled_documents):
            pooled_mask[idx, : pooled_document.size(0)] = True

        return pooled_embeddings, pooled_mask

    def forward(
        self, sentence_features: Iterable[dict[str, torch.Tensor]], labels: torch.Tensor
    ) -> torch.Tensor:
        """Computes the distillation loss with respect to SentenceTransformer.

        Parameters
        ----------
        sentence_features
            List of tokenized sentences. The first sentence is the query and the rest are documents.
        labels
            The logits for the distillation loss.

        """
        queries_embeddings = torch.nn.functional.normalize(
            self.model(sentence_features[0])["token_embeddings"], p=2, dim=-1
        )
        # Compute the bs * n_ways embeddings
        documents_embeddings = torch.nn.functional.normalize(
            self.model(sentence_features[1])["token_embeddings"], p=2, dim=-1
        )
        skiplist = self._get_model_attr("skiplist")
        do_query_expansion = self._get_model_attr("do_query_expansion")

        masks = extract_skiplist_mask(
            sentence_features=sentence_features, skiplist=skiplist
        )
        documents_embeddings_mask = masks[1].bool()
        pool_factor = self._sample_pool_factor()
        self.last_pool_factor = pool_factor

        if pool_factor > 1:
            documents_embeddings, documents_embeddings_mask = self._pool_documents(
                documents_embeddings=documents_embeddings,
                documents_mask=documents_embeddings_mask,
                pool_factor=pool_factor,
            )

        # Reshape documents back to (bs, n_ways, num_tokens, dim)
        documents_embeddings = documents_embeddings.view(
            queries_embeddings.size(0), -1, *documents_embeddings.shape[1:]
        )

        documents_embeddings_mask = documents_embeddings_mask.view(
            queries_embeddings.size(0), -1, *documents_embeddings_mask.shape[1:]
        )
        scores = self.score_metric(
            queries_embeddings,
            documents_embeddings,
            queries_mask=masks[0] if not do_query_expansion else None,
            documents_mask=documents_embeddings_mask,
        )
        if self.normalize_scores:
            # Compute max and min along the num_scores dimension (dim=1)
            max_scores, _ = torch.max(scores, dim=1, keepdim=True)
            min_scores, _ = torch.min(scores, dim=1, keepdim=True)

            # Avoid division by zero by adding a small epsilon
            epsilon = 1e-8

            # Normalize the scores
            scores = (scores - min_scores) / (max_scores - min_scores + epsilon)
        return self.loss_function(
            torch.nn.functional.log_softmax(scores, dim=-1),
            torch.nn.functional.log_softmax(labels, dim=-1),
        )
