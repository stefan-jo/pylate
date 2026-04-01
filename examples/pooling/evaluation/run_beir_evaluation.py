#!/usr/bin/env python3
"""Run BEIR evaluation with PLAID and save results.

Evaluates a ColBERT model on a single BEIR dataset using a PLAID index,
and writes metric scores plus a human-readable summary.
Supports optional document embedding pooling via pool_factor and pool_method.
Results are saved as JSON (full scores and run config) and as a summary text file.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch

from eval_shared import (
    PooledColBERT,
    model_size_mb,
    scores_to_serializable,
)
from pylate import evaluation, indexes, models, retrieve

ALL_BEIR_DATASETS = [
    "arguana",
    "climate-fever",
    "dbpedia-entity",
    "fever",
    "fiqa",
    "hotpotqa",
    "msmarco",
    "nfcorpus",
    "nq",
    "quora",
    "scidocs",
    "scifact",
    "trec-covid",
    "webis-touche2020",
    "cqadupstack/android",
    "cqadupstack/english",
    "cqadupstack/gaming",
    "cqadupstack/gis",
    "cqadupstack/mathematica",
    "cqadupstack/physics",
    "cqadupstack/programmers",
    "cqadupstack/stats",
    "cqadupstack/tex",
    "cqadupstack/unix",
    "cqadupstack/webmasters",
    "cqadupstack/wordpress",
]

BEIR_QUERY_LENGTHS = {
    "quora": 32,
    "climate-fever": 64,
    "nq": 32,
    "msmarco": 32,
    "hotpotqa": 32,
    "nfcorpus": 32,
    "scifact": 48,
    "trec-covid": 48,
    "fiqa": 32,
    "arguana": 64,
    "scidocs": 48,
    "dbpedia-entity": 32,
    "webis-touche2020": 32,
    "fever": 32,
    "cqadupstack/android": 32,
    "cqadupstack/english": 32,
    "cqadupstack/gaming": 32,
    "cqadupstack/gis": 32,
    "cqadupstack/mathematica": 32,
    "cqadupstack/physics": 32,
    "cqadupstack/programmers": 32,
    "cqadupstack/stats": 32,
    "cqadupstack/tex": 32,
    "cqadupstack/unix": 32,
    "cqadupstack/webmasters": 32,
    "cqadupstack/wordpress": 32,
}

BEIR_METRICS = ["ndcg@10", "mrr@10", "map@100", "recall@10", "recall@100"]
DEFAULT_BATCH_SIZE = 16
DEFAULT_TOP_K = 20
DEFAULT_DOCUMENT_LENGTH = 180


def _run_config_suffix(k: int, document_length: int) -> str:
    if k == DEFAULT_TOP_K and document_length == DEFAULT_DOCUMENT_LENGTH:
        return ""
    return f"_k{k}_d{document_length}"


def parse_scores(
    scores: dict[str, dict[str, float]],
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Split per-dataset scores and compute means across datasets."""
    dataset_results: dict[str, dict[str, float]] = {}
    metric_values: dict[str, list[float]] = defaultdict(list)
    for dataset_name, dataset_scores in scores.items():
        dataset_results[dataset_name] = {}
        for metric_name, metric_value in dataset_scores.items():
            metric_value_float = float(metric_value)
            dataset_results[dataset_name][metric_name] = metric_value_float
            metric_values[metric_name].append(metric_value_float)

    mean_results = {
        metric_name: sum(values) / len(values)
        for metric_name, values in metric_values.items()
        if values
    }
    return dataset_results, mean_results


def load_beir_dataset(dataset_name: str) -> tuple[list[dict[str, str]], dict[str, str], dict]:
    """Load one BEIR dataset, including cqadupstack subsets."""
    if dataset_name.startswith("cqadupstack/"):
        from beir import util

        util.download_and_unzip(
            url="https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/cqadupstack.zip",
            out_dir="./evaluation_datasets/",
        )
        return evaluation.load_custom_dataset(
            f"evaluation_datasets/{dataset_name}",
            split="test",
        )

    split = "dev" if dataset_name == "msmarco" else "test"
    return evaluation.load_beir(dataset_name=dataset_name, split=split)


def write_summary(
    path: Path,
    scores: dict[str, dict[str, float]],
    evaluation_time_seconds: float,
    model_name: str,
    pool_factor: int,
    pool_method: str,
    dataset_name: str,
    query_length: int,
    document_length: int,
    batch_size: int,
    k: int,
    use_triton: bool = False,
    use_sklearn: bool = False,
) -> None:
    """Write a human-readable summary to a text file."""
    dataset_results, mean_results = parse_scores(scores)
    with open(path, "w") as f:
        f.write("BEIR PLAID Evaluation Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model:        {model_name}\n")
        f.write(f"pool_factor:  {pool_factor}\n")
        f.write(f"pool_method:  {pool_method}\n")
        f.write(f"use_triton:   {use_triton}\n")
        f.write(f"use_sklearn:  {use_sklearn}\n")
        f.write(f"dataset:      {dataset_name}\n")
        f.write(f"query_length: {query_length}\n")
        f.write(f"doc_length:   {document_length}\n")
        f.write(f"batch_size:   {batch_size}\n")
        f.write(f"top_k:        {k}\n")
        f.write(f"Time:         {evaluation_time_seconds:.2f} s ({evaluation_time_seconds/60:.2f} min)\n\n")
        f.write("Results:\n")
        for metric in BEIR_METRICS:
            if metric in mean_results:
                f.write(f"  {metric:<15}: {mean_results[metric]:.4f}\n")
        if "ndcg@10" in dataset_results.get(dataset_name, {}):
            f.write("\nndcg@10:\n")
            f.write(f"  {dataset_name:<20}: {dataset_results[dataset_name]['ndcg@10']:.4f}\n")
    print(f"Summary written to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run BEIR evaluation with PLAID and save results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mixedbread-ai/mxbai-edge-colbert-v0-32m",
        help="HuggingFace model name or path for ColBERT.",
    )
    parser.add_argument(
        "--pool-factor",
        type=int,
        default=1,
        help="Document embedding pool factor (1 = no pooling).",
    )
    parser.add_argument(
        "--pool-method",
        type=str,
        default="hierarchical",
        choices=["hierarchical", "span", "kmeans"],
        help="Document embedding pooling method.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for encoding queries/documents (reduce if OOM).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Top-k retrieved documents per query for evaluation.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="nfcorpus",
        help=(
            "BEIR dataset name. "
            f"Available: {', '.join(ALL_BEIR_DATASETS)}"
        ),
    )
    parser.add_argument(
        "--document-length",
        type=int,
        default=DEFAULT_DOCUMENT_LENGTH,
        help="Maximum document token length.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory to write results JSON and summary.",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Do not convert model to fp16.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (e.g. cuda, cpu). Auto-detected if not set.",
    )
    parser.add_argument(
        "--use-triton",
        action="store_true",
        help="Use Triton backend for k-means pooling (faster on modern GPUs).",
    )
    parser.add_argument(
        "--use-sklearn",
        action="store_true",
        help="Use sklearn KMeans backend for k-means pooling.",
    )
    args = parser.parse_args()
    use_triton: bool = args.use_triton
    use_sklearn: bool = args.use_sklearn
    dataset_name = args.dataset_name.strip().lower()
    if dataset_name not in ALL_BEIR_DATASETS:
        raise ValueError(
            f"Unknown dataset: {dataset_name!r}. "
            f"Available: {ALL_BEIR_DATASETS}"
        )
    query_length = BEIR_QUERY_LENGTHS.get(dataset_name, 32)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Dataset: {dataset_name}")
    print(f"query_length={query_length}, document_length={args.document_length}")

    print(f"Loading model: {args.model}")
    base_model = models.ColBERT(
        model_name_or_path=args.model,
        device=device,
        truncation=True,
        query_length=query_length,
        document_length=args.document_length,
    )

    if not args.no_fp16:
        print(f"Model size before .half(): {model_size_mb(base_model):.2f} MB")
        base_model = base_model.half()
        print(f"Model size after .half(): {model_size_mb(base_model):.2f} MB")

    model = base_model
    if args.pool_factor != 1:
        model = PooledColBERT(
            base_model,
            pool_factor=args.pool_factor,
            pool_method=args.pool_method,
            use_triton=use_triton,
            use_sklearn=use_sklearn,
        )

    print("=" * 60)
    triton_info = f", use_triton={use_triton}" if args.pool_method == "kmeans" else ""
    sklearn_info = (
        f", use_sklearn={use_sklearn}" if args.pool_method == "kmeans" else ""
    )
    print(
        "Starting BEIR evaluation with PLAID "
        f"(pool_factor={args.pool_factor}, pool_method={args.pool_method}{triton_info}{sklearn_info}, k={args.k})"
    )
    print("=" * 60)

    start_time = time.time()
    scores: dict[str, dict[str, float]] = {}
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        index_folder = args.output_dir / "indexes"
        safe_name = args.model.replace("/", "_")

        print(f"\nEvaluating dataset: {dataset_name}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        documents, queries, qrels = load_beir_dataset(dataset_name)
        print(
            f"Loaded {len(documents)} documents and {len(queries)} queries "
            f"(qrels={len(qrels)}), query_length={query_length}"
        )

        index = indexes.PLAID(
            index_folder=str(index_folder),
            index_name=(
                f"{dataset_name.replace('/', '_')}_{safe_name}_"
                f"{args.pool_method}"
            ),
            override=True,
            device=device,
            use_triton=use_triton,
        )
        retriever = retrieve.ColBERT(index=index)

        documents_embeddings = model.encode(
            sentences=[document["text"] for document in documents],
            batch_size=args.batch_size,
            is_query=False,
            show_progress_bar=True,
        )

        index.add_documents(
            documents_ids=[document["id"] for document in documents],
            documents_embeddings=documents_embeddings,
        )

        queries_embeddings = model.encode(
            sentences=list(queries.values()),
            is_query=True,
            batch_size=args.batch_size,
            show_progress_bar=True,
        )

        dataset_scores = retriever.retrieve(
            queries_embeddings=queries_embeddings,
            k=args.k,
        )

        # Remove self-match query IDs when they appear as document IDs (e.g. FiQA).
        for (query_id, _), query_scores in zip(queries.items(), dataset_scores):
            query_scores[:] = [
                score for score in query_scores if score["id"] != query_id
            ]

        evaluation_scores = evaluation.evaluate(
            scores=dataset_scores,
            qrels=qrels,
            queries=list(queries.keys()),
            metrics=BEIR_METRICS,
        )
        scores[dataset_name] = scores_to_serializable(evaluation_scores)
        print(
            f"ndcg@10={scores[dataset_name].get('ndcg@10', 0.0):.4f}, "
            f"mrr@10={scores[dataset_name].get('mrr@10', 0.0):.4f}"
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n⚠️  GPU Out of Memory detected!")
            print(f"Try reducing --batch-size from {args.batch_size} to {args.batch_size//2}.")
            raise
        raise
    evaluation_time = time.time() - start_time
    print(f"\nEvaluation completed in {evaluation_time:.2f} s ({evaluation_time/60:.2f} min)")

    safe_name = args.model.replace("/", "_")
    prefix = (
        f"beir_{safe_name}_pool{args.pool_factor}_{args.pool_method}"
        f"_ds_{dataset_name.replace('/', '_')}"
        f"{_run_config_suffix(args.k, args.document_length)}"
    )

    _, mean_results = parse_scores(scores)
    mean_ndcg_at_10 = mean_results.get("ndcg@10")

    results_data = {
        "model": args.model,
        "pool_factor": args.pool_factor,
        "pool_method": args.pool_method,
        "use_triton": use_triton,
        "use_sklearn": use_sklearn,
        "dataset_name": dataset_name,
        "query_length": query_length,
        "document_length": args.document_length,
        "batch_size": args.batch_size,
        "k": args.k,
        "evaluation_time_seconds": evaluation_time,
        "mean_ndcg_at_10": mean_ndcg_at_10,
        "scores": scores,
    }
    json_path = args.output_dir / f"{prefix}.json"
    with open(json_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"Results JSON written to {json_path}")

    summary_path = args.output_dir / f"{prefix}_summary.txt"
    write_summary(
        summary_path,
        scores,
        evaluation_time,
        args.model,
        args.pool_factor,
        args.pool_method,
        dataset_name,
        query_length,
        args.document_length,
        args.batch_size,
        args.k,
        use_triton=use_triton,
        use_sklearn=use_sklearn,
    )

    _, mean_results = parse_scores(scores)
    print("\nMean Results (ndcg@10, mrr@10):", end=" ")
    print(f"{mean_results.get('ndcg@10', 0):.4f} / {mean_results.get('mrr@10', 0):.4f}")


if __name__ == "__main__":
    main()
