#!/usr/bin/env python3
"""Run NanoBEIR evaluation and save results.

Evaluates a ColBERT model on all NanoBEIR datasets (small-scale BEIR benchmarks)
and writes the full metric scores plus a human-readable summary. Supports
optional document embedding pooling via pool_factor and pool_method. Results are saved as JSON
(full scores and run config) and as a summary text file.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from eval_shared import (
    PooledColBERT,
    dataset_suffix,
    model_size_mb,
    normalize_cli_list,
    scores_to_serializable,
)
from pylate import evaluation, models

ALL_NANOBEIR_DATASETS = [
    "arguana",
    "climatefever",
    "dbpedia",
    "fever",
    "fiqa2018",
    "hotpotqa",
    "msmarco",
    "nfcorpus",
    "nq",
    "quoraretrieval",
    "scidocs",
    "scifact",
    "touche2020",
]

def parse_scores(scores: dict) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Split raw evaluator scores into per-dataset and mean results."""
    dataset_results: dict[str, dict[str, float]] = {}
    mean_results: dict[str, float] = {}
    for key, value in scores.items():
        if "NanoBEIR_mean" in key:
            metric_name = key.replace("NanoBEIR_mean_MaxSim_", "")
            mean_results[metric_name] = float(value)
        elif "Nano" in key and "mean" not in key:
            parts = key.split("_MaxSim_")
            if len(parts) == 2:
                dataset_name = parts[0].replace("Nano", "")
                metric_name = parts[1]
                if dataset_name not in dataset_results:
                    dataset_results[dataset_name] = {}
                dataset_results[dataset_name][metric_name] = float(value)
    return dataset_results, mean_results


def write_summary(
    path: Path,
    scores: dict,
    evaluation_time_seconds: float,
    model_name: str,
    pool_factor: int,
    pool_method: str,
    dataset_names: list[str],
    batch_size: int,
    use_triton: bool | None = None,
) -> None:
    """Write a human-readable summary to a text file."""
    dataset_results, mean_results = parse_scores(scores)
    with open(path, "w") as f:
        f.write("NanoBEIR Full Evaluation Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model:        {model_name}\n")
        f.write(f"pool_factor:  {pool_factor}\n")
        f.write(f"pool_method:  {pool_method}\n")
        if use_triton is not None:
            f.write(f"use_triton:   {use_triton}\n")
        f.write(f"datasets:     {', '.join(dataset_names)}\n")
        f.write(f"batch_size:   {batch_size}\n")
        f.write(f"Time:         {evaluation_time_seconds:.2f} s ({evaluation_time_seconds/60:.2f} min)\n\n")
        f.write("Mean Results Across All Datasets:\n")
        for metric in ["ndcg@10", "mrr@10", "map@100", "recall@10", "recall@100", "accuracy@10"]:
            if metric in mean_results:
                f.write(f"  {metric:<15}: {mean_results[metric]:.4f}\n")
        f.write("\nPer-Dataset Results (ndcg@10):\n")
        for dataset_name in sorted(dataset_results.keys()):
            if "ndcg@10" in dataset_results[dataset_name]:
                f.write(f"  {dataset_name:<20}: {dataset_results[dataset_name]['ndcg@10']:.4f}\n")
    print(f"Summary written to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full NanoBEIR evaluation and save results.",
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
        "--use-triton",
        action="store_true",
        help="Use Triton backend for k-means pooling (faster on modern GPUs).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for encoding (reduce if OOM).",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional NanoBEIR subset (space/comma separated). "
            f"Available: {', '.join(ALL_NANOBEIR_DATASETS)}"
        ),
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
    args = parser.parse_args()
    use_triton: bool | None = True if args.use_triton else None
    dataset_names = normalize_cli_list(args.datasets, lowercase=True, unique=False)
    if dataset_names is not None:
        invalid_datasets = sorted(set(dataset_names) - set(ALL_NANOBEIR_DATASETS))
        if invalid_datasets:
            raise ValueError(
                f"Unknown datasets: {invalid_datasets}. "
                f"Available: {ALL_NANOBEIR_DATASETS}"
            )
        selected_datasets = dataset_names
    else:
        selected_datasets = ALL_NANOBEIR_DATASETS

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        "Datasets: "
        + ("all" if dataset_names is None else ", ".join(selected_datasets))
    )

    print(f"Loading model: {args.model}")
    model = models.ColBERT(
        model_name_or_path=args.model,
        device=device,
        truncation=True,
        query_length=32,
        document_length=180,
    )

    if not args.no_fp16:
        print(f"Model size before .half(): {model_size_mb(model):.2f} MB")
        model = model.half()
        print(f"Model size after .half(): {model_size_mb(model):.2f} MB")

    if args.pool_factor != 1:
        model = PooledColBERT(
            model,
            pool_factor=args.pool_factor,
            pool_method=args.pool_method,
            use_triton=use_triton,
        )

    evaluator_kwargs = {
        "batch_size": args.batch_size,
        "show_progress_bar": True,
    }
    if dataset_names is not None:
        evaluator_kwargs["dataset_names"] = dataset_names

    evaluator = evaluation.NanoBEIREvaluator(**evaluator_kwargs)

    print("=" * 60)
    triton_info = f", use_triton={use_triton}" if args.pool_method == "kmeans" else ""
    print(
        "Starting evaluation on all NanoBEIR datasets "
        f"(pool_factor={args.pool_factor}, pool_method={args.pool_method}{triton_info})"
    )
    print("=" * 60)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    start_time = time.time()
    try:
        scores = evaluator(model)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n⚠️  GPU Out of Memory detected!")
            print(f"Try reducing --batch-size from {args.batch_size} to {args.batch_size//2}.")
            raise
        raise
    evaluation_time = time.time() - start_time
    print(f"\nEvaluation completed in {evaluation_time:.2f} s ({evaluation_time/60:.2f} min)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = args.model.replace("/", "_")
    prefix = (
        f"nanobeir_{safe_name}_pool{args.pool_factor}_{args.pool_method}"
        f"{dataset_suffix(dataset_names)}"
    )

    _, mean_results = parse_scores(scores)
    mean_ndcg_at_10 = mean_results.get("ndcg@10")

    results_data = {
        "model": args.model,
        "pool_factor": args.pool_factor,
        "pool_method": args.pool_method,
        "use_triton": use_triton,
        "dataset_names": selected_datasets,
        "batch_size": args.batch_size,
        "evaluation_time_seconds": evaluation_time,
        "mean_ndcg_at_10": mean_ndcg_at_10,
        "scores": scores_to_serializable(scores),
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
        selected_datasets,
        args.batch_size,
        use_triton=use_triton,
    )

    dataset_results, mean_results = parse_scores(scores)
    print("\nMean Results (ndcg@10, mrr@10):", end=" ")
    print(f"{mean_results.get('ndcg@10', 0):.4f} / {mean_results.get('mrr@10', 0):.4f}")


if __name__ == "__main__":
    main()
