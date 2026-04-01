from __future__ import annotations

import pytest
import torch

from pylate.losses import Distillation


class DummyModel:
    def __init__(self) -> None:
        self.skiplist = [0]
        self.do_query_expansion = False
        self.pool_calls = 0
        self.last_pool_kwargs: dict = {}
        self.last_pooled_input_lengths: list[int] = []

    def __call__(self, sentence_feature: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"token_embeddings": sentence_feature["token_embeddings"]}

    def _pool_document_embeddings(
        self,
        documents_embeddings: list[torch.Tensor],
        pool_factor: int,
        protected_tokens: int,
        pool_method: str,
        use_sklearn: bool = False,
    ) -> list[torch.Tensor]:
        self.pool_calls += 1
        self.last_pool_kwargs = {
            "pool_factor": pool_factor,
            "protected_tokens": protected_tokens,
            "pool_method": pool_method,
            "use_sklearn": use_sklearn,
        }
        self.last_pooled_input_lengths = [
            document_embeddings.size(0) for document_embeddings in documents_embeddings
        ]
        return documents_embeddings


def make_sentence_features() -> tuple[list[dict[str, torch.Tensor]], torch.Tensor]:
    query_embeddings = torch.randn(2, 3, 8, requires_grad=True)
    document_embeddings = torch.randn(6, 4, 8, requires_grad=True)

    sentence_features = [
        {
            "token_embeddings": query_embeddings,
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
        },
        {
            "token_embeddings": document_embeddings,
            "input_ids": torch.tensor(
                [
                    [7, 8, 0, 9],
                    [0, 8, 7, 6],
                    [1, 2, 3, 0],
                    [2, 0, 4, 5],
                    [6, 7, 8, 9],
                    [0, 0, 1, 2],
                ]
            ),
            "attention_mask": torch.tensor(
                [
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 0],
                    [1, 1, 1, 1],
                    [1, 1, 0, 0],
                    [1, 1, 1, 1],
                ]
            ),
        },
    ]
    labels = torch.randn(2, 3)
    return sentence_features, labels


def test_distillation_pool_factor_one_does_not_call_pooling() -> None:
    sentence_features, labels = make_sentence_features()
    model = DummyModel()

    loss_fn = Distillation(
        model=model,
        normalize_scores=False,
        pool_factor=1,
    )
    loss = loss_fn(sentence_features=sentence_features, labels=labels)
    loss.backward()

    assert model.pool_calls == 0


@pytest.mark.parametrize(
    ("pool_method", "use_sklearn"),
    [
        ("span", False),
        ("hierarchical", False),
        ("kmeans", False),
        ("kmeans", True),
    ],
)
def test_distillation_pooling_forwards_pooling_config(
    pool_method: str, use_sklearn: bool
) -> None:
    sentence_features, labels = make_sentence_features()
    model = DummyModel()

    loss_fn = Distillation(
        model=model,
        normalize_scores=False,
        pool_factor=2,
        pool_method=pool_method,
        protected_tokens=1,
        use_sklearn=use_sklearn,
    )
    loss = loss_fn(sentence_features=sentence_features, labels=labels)
    loss.backward()

    assert model.pool_calls == 1
    assert model.last_pool_kwargs == {
        "pool_factor": 2,
        "protected_tokens": 1,
        "pool_method": pool_method,
        "use_sklearn": use_sklearn,
    }


def test_distillation_pooling_applies_skiplist_attention_mask() -> None:
    sentence_features, labels = make_sentence_features()
    model = DummyModel()

    loss_fn = Distillation(
        model=model,
        normalize_scores=False,
        pool_factor=2,
    )
    loss = loss_fn(sentence_features=sentence_features, labels=labels)
    loss.backward()

    expected_mask = torch.logical_and(
        sentence_features[1]["input_ids"] != 0,
        sentence_features[1]["attention_mask"].bool(),
    )
    expected_lengths = expected_mask.sum(dim=1).tolist()
    assert model.last_pooled_input_lengths == expected_lengths


def test_distillation_pooling_keeps_gradient_path_to_documents() -> None:
    sentence_features, labels = make_sentence_features()
    model = DummyModel()

    loss_fn = Distillation(
        model=model,
        normalize_scores=False,
        pool_factor=2,
        pool_method="span",
    )
    loss = loss_fn(sentence_features=sentence_features, labels=labels)
    loss.backward()

    document_grad = sentence_features[1]["token_embeddings"].grad
    assert document_grad is not None
    assert torch.isfinite(document_grad).all()
    assert document_grad.abs().sum().item() > 0.0


def test_distillation_pool_factors_samples_only_from_allowed_set() -> None:
    sentence_features, labels = make_sentence_features()
    model = DummyModel()
    allowed_factors = [1, 2, 4]

    loss_fn = Distillation(
        model=model,
        normalize_scores=False,
        pool_factors=allowed_factors,
        pool_factor_seed=1234,
    )
    sampled_factors = []
    for _ in range(20):
        loss = loss_fn(sentence_features=sentence_features, labels=labels)
        loss.backward()
        sampled_factors.append(loss_fn.last_pool_factor)

    assert set(sampled_factors).issubset(set(allowed_factors))


def test_distillation_pool_factors_sampling_is_deterministic_with_seed() -> None:
    sentence_features, labels = make_sentence_features()
    model_a = DummyModel()
    model_b = DummyModel()

    loss_fn_a = Distillation(
        model=model_a,
        normalize_scores=False,
        pool_factors=[1, 2, 3, 4],
        pool_factor_seed=77,
    )
    loss_fn_b = Distillation(
        model=model_b,
        normalize_scores=False,
        pool_factors=[1, 2, 3, 4],
        pool_factor_seed=77,
    )

    sampled_a = []
    sampled_b = []
    for _ in range(25):
        loss_a = loss_fn_a(sentence_features=sentence_features, labels=labels)
        loss_a.backward()
        sampled_a.append(loss_fn_a.last_pool_factor)

        loss_b = loss_fn_b(sentence_features=sentence_features, labels=labels)
        loss_b.backward()
        sampled_b.append(loss_fn_b.last_pool_factor)

    assert sampled_a == sampled_b


def test_distillation_pool_factors_with_one_skip_pooling_for_sampled_ones() -> None:
    sentence_features, labels = make_sentence_features()
    model = DummyModel()

    loss_fn = Distillation(
        model=model,
        normalize_scores=False,
        pool_factors=[1, 2],
        pool_factor_seed=99,
        pool_method="span",
    )

    seen_factors = set()
    previous_pool_calls = 0
    for _ in range(30):
        loss = loss_fn(sentence_features=sentence_features, labels=labels)
        loss.backward()
        sampled_factor = loss_fn.last_pool_factor
        seen_factors.add(sampled_factor)

        if sampled_factor == 1:
            assert model.pool_calls == previous_pool_calls
        else:
            assert model.pool_calls == previous_pool_calls + 1
        previous_pool_calls = model.pool_calls

    assert seen_factors == {1, 2}


def test_distillation_pool_factors_validation() -> None:
    model = DummyModel()

    with pytest.raises(ValueError, match="pool_factors must be non-empty"):
        Distillation(
            model=model,
            pool_factors=[],
        )

    with pytest.raises(ValueError, match="All pool factors must be >= 1"):
        Distillation(
            model=model,
            pool_factors=[0, 2],
        )
