from __future__ import annotations

import sys
import types

import pytest
import torch

from pylate.models.colbert import ColBERT


def _make_model() -> ColBERT:
    return ColBERT.__new__(ColBERT)


def test_span_pooling_keeps_protected_tokens_and_window_averages() -> None:
    model = _make_model()
    document_embeddings = [
        torch.tensor(
            [
                [100.0, 100.0],
                [1.0, 1.0],
                [3.0, 3.0],
                [5.0, 5.0],
                [7.0, 7.0],
                [9.0, 9.0],
            ]
        )
    ]

    pooled = model.pool_embeddings_span(
        documents_embeddings=document_embeddings,
        pool_factor=2,
        protected_tokens=1,
    )[0]

    expected = torch.tensor(
        [
            [2.0, 2.0],
            [6.0, 6.0],
            [9.0, 9.0],
            [100.0, 100.0],
        ]
    )
    assert torch.allclose(pooled, expected)


def test_hierarchical_pooling_handles_single_token_to_pool() -> None:
    model = _make_model()
    document_embeddings = [torch.tensor([[100.0, 100.0], [1.0, 1.0]])]

    pooled = model.pool_embeddings_hierarchical(
        documents_embeddings=document_embeddings,
        pool_factor=4,
        protected_tokens=1,
    )[0]

    expected = torch.tensor([[1.0, 1.0], [100.0, 100.0]])
    assert torch.allclose(pooled, expected)


def test_pooling_method_validation() -> None:
    model = _make_model()
    document_embeddings = [torch.tensor([[1.0, 1.0], [2.0, 2.0]])]

    with pytest.raises(ValueError, match="Invalid pool_method"):
        model._pool_document_embeddings(
            documents_embeddings=document_embeddings,
            pool_factor=2,
            protected_tokens=1,
            pool_method="unknown",
        )


def test_pooling_factor_validation() -> None:
    model = _make_model()
    document_embeddings = [torch.tensor([[1.0, 1.0], [2.0, 2.0]])]

    with pytest.raises(ValueError, match="pool_factor must be >= 1"):
        model._pool_document_embeddings(
            documents_embeddings=document_embeddings,
            pool_factor=0,
            protected_tokens=1,
            pool_method="span",
        )


def test_pooling_protected_tokens_validation() -> None:
    model = _make_model()
    document_embeddings = [torch.tensor([[1.0, 1.0], [2.0, 2.0]])]

    with pytest.raises(ValueError, match="protected_tokens must be >= 0"):
        model._pool_document_embeddings(
            documents_embeddings=document_embeddings,
            pool_factor=2,
            protected_tokens=-1,
            pool_method="span",
        )


def test_kmeans_pooling_with_fastkmeans_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyFastKMeans:
        def __init__(self, dim: int, n_clusters: int, **kwargs) -> None:
            self.dim = dim
            self.n_clusters = n_clusters
            self.centroids = None

        def train(self, sample) -> None:
            self.centroids = sample[[0, -1]][: self.n_clusters]

        def predict(self, sample):
            # Deterministic split into two groups for the test data
            return (sample[:, 0] >= 8.0).astype("int64")

    fake_module = types.ModuleType("fastkmeans")
    fake_module.FastKMeans = DummyFastKMeans
    monkeypatch.setitem(sys.modules, "fastkmeans", fake_module)

    model = _make_model()
    document_embeddings = [
        torch.tensor(
            [
                [100.0, 100.0],
                [1.0, 1.0],
                [2.0, 2.0],
                [8.0, 8.0],
                [9.0, 9.0],
            ]
        )
    ]

    pooled = model.pool_embeddings_kmeans(
        documents_embeddings=document_embeddings,
        pool_factor=2,
        protected_tokens=1,
    )[0]

    expected = torch.tensor([[1.5, 1.5], [8.5, 8.5], [100.0, 100.0]])
    assert torch.allclose(pooled, expected)


def test_kmeans_pooling_with_sklearn_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyFastKMeans:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("fastkmeans backend should not be used")

    class DummySklearnKMeans:
        def __init__(self, n_clusters: int, **kwargs) -> None:
            self.n_clusters = n_clusters

        def fit_predict(self, sample):
            return (sample[:, 0] >= 8.0).astype("int64")

    fake_fastkmeans = types.ModuleType("fastkmeans")
    fake_fastkmeans.FastKMeans = DummyFastKMeans
    monkeypatch.setitem(sys.modules, "fastkmeans", fake_fastkmeans)

    fake_sklearn = types.ModuleType("sklearn")
    fake_sklearn_cluster = types.ModuleType("sklearn.cluster")
    fake_sklearn_cluster.KMeans = DummySklearnKMeans
    fake_sklearn.cluster = fake_sklearn_cluster
    monkeypatch.setitem(sys.modules, "sklearn", fake_sklearn)
    monkeypatch.setitem(sys.modules, "sklearn.cluster", fake_sklearn_cluster)

    model = _make_model()
    document_embeddings = [
        torch.tensor(
            [
                [100.0, 100.0],
                [1.0, 1.0],
                [2.0, 2.0],
                [8.0, 8.0],
                [9.0, 9.0],
            ]
        )
    ]

    pooled = model.pool_embeddings_kmeans(
        documents_embeddings=document_embeddings,
        pool_factor=2,
        protected_tokens=1,
        use_sklearn=True,
    )[0]

    expected = torch.tensor([[1.5, 1.5], [8.5, 8.5], [100.0, 100.0]])
    assert torch.allclose(pooled, expected)
