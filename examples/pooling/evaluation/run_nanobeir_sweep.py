#!/usr/bin/env python3
"""Run a NanoBEIR pooling sweep and aggregate retrieval metrics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import pandas as pd
import torch
from tqdm.auto import tqdm

from eval_shared import (
    build_sweep_run_dir,
    dataset_suffix,
    discover_sweep_result_records,
    normalize_cli_list,
    run_eval_subprocess,
    safe_name,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_MODEL = "mixedbread-ai/mxbai-edge-colbert-v0-32m"
DEFAULT_POOL_FACTORS = [1, 2, 3, 4, 5, 6]
DEFAULT_BATCH_SIZE = 16
DEFAULT_QUERY_LENGTH = 48
DEFAULT_DOCUMENT_LENGTH = 180
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
METHODS = [
    ("hier", "hierarchical"),
    ("kmeans", "kmeans"),
    ("slice", "span"),
]


def _parse_selected_methods(method_values: list[str] | None) -> list[tuple[str, str]]:
    if method_values is None:
        return METHODS.copy()

    requested_methods = normalize_cli_list(
        method_values, lowercase=True, unique=False
    )
    if requested_methods is None:
        return METHODS.copy()

    alias_to_cli = {
        "hier": "hierarchical",
        "hierarchical": "hierarchical",
        "kmeans": "kmeans",
        "slice": "span",
        "span": "span",
    }
    method_names = {method_cli: method_name for method_name, method_cli in METHODS}

    selected_methods: list[tuple[str, str]] = []
    seen: set[str] = set()
    invalid_methods: list[str] = []
    for requested_method in requested_methods:
        method_cli = alias_to_cli.get(requested_method)
        if method_cli is None:
            invalid_methods.append(requested_method)
            continue
        if method_cli in seen:
            continue
        seen.add(method_cli)
        selected_methods.append((method_names[method_cli], method_cli))

    if invalid_methods:
        raise ValueError(
            f"Unknown pooling methods: {sorted(set(invalid_methods))}. "
            "Available: hier, kmeans, slice (aliases: hierarchical, span)."
        )
    if not selected_methods:
        raise ValueError("No pooling methods selected.")

    return selected_methods


def _run_config_suffix(query_length: int, document_length: int) -> str:
    if (
        query_length == DEFAULT_QUERY_LENGTH
        and document_length == DEFAULT_DOCUMENT_LENGTH
    ):
        return ""
    return f"_q{query_length}_d{document_length}"


def _run_single_evaluation(
    eval_script: Path,
    repo_root: Path,
    output_dir: Path,
    model_name: str,
    pool_factor: int,
    method_cli: str,
    device: str,
    batch_size: int,
    query_length: int,
    document_length: int,
    no_fp16: bool,
    dataset_names: list[str] | None,
    use_triton: bool = False,
    use_sklearn: bool = False,
) -> None:
    cli_args = [
        "--model",
        model_name,
        "--pool-factor",
        str(pool_factor),
        "--pool-method",
        method_cli,
        "--output-dir",
        str(output_dir),
        "--device",
        device,
        "--batch-size",
        str(batch_size),
        "--query-length",
        str(query_length),
        "--document-length",
        str(document_length),
    ]
    if no_fp16:
        cli_args.append("--no-fp16")
    if use_triton:
        cli_args.append("--use-triton")
    if use_sklearn:
        cli_args.append("--use-sklearn")
    if dataset_names:
        cli_args.extend(["--datasets", *dataset_names])

    run_eval_subprocess(
        eval_script=eval_script,
        repo_root=repo_root,
        cli_args=cli_args,
    )


def _result_json_path(
    output_dir: Path,
    model_name: str,
    pool_factor: int,
    method_cli: str,
    dataset_names: list[str] | None,
    query_length: int,
    document_length: int,
) -> Path:
    model_name_safe = safe_name(value=model_name)
    return output_dir / (
        f"nanobeir_{model_name_safe}_pool{pool_factor}_{method_cli}"
        f"{dataset_suffix(dataset_names)}"
        f"{_run_config_suffix(query_length, document_length)}.json"
    )


def _parse_nanobeir_scores(
    scores: dict,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Split raw evaluator scores into per-dataset and mean results."""
    dataset_results: dict[str, dict[str, float]] = {}
    mean_results: dict[str, float] = {}

    for key, value in scores.items():
        if key.startswith("NanoBEIR_mean_MaxSim_"):
            metric_name = key.replace("NanoBEIR_mean_MaxSim_", "")
            mean_results[metric_name] = float(value)
        elif key.startswith("Nano") and "mean" not in key:
            parts = key.split("_MaxSim_")
            if len(parts) != 2:
                continue
            dataset_name = parts[0].replace("Nano", "", 1)
            metric_name = parts[1]
            if dataset_name not in dataset_results:
                dataset_results[dataset_name] = {}
            dataset_results[dataset_name][metric_name] = float(value)

    return dataset_results, mean_results


def _write_overview_artifacts(
    output_dir: Path,
    rows: list[dict],
    per_dataset_rows: list[dict],
    pool_factors: list[int],
    methods: list[tuple[str, str]],
) -> None:
    df = pd.DataFrame(rows)
    df = df.sort_values(["pool_factor", "method_order"]).reset_index(drop=True)

    per_dataset_df = pd.DataFrame(per_dataset_rows)
    per_dataset_df = per_dataset_df.sort_values(
        ["dataset_name", "pool_factor", "method_order"]
    ).reset_index(drop=True)

    per_dataset_csv_path = output_dir / "overview_per_dataset_metrics_long.csv"
    per_dataset_df.drop(columns=["method_order"]).to_csv(
        per_dataset_csv_path, index=False
    )

    long_csv_path = output_dir / "overview_mean_metrics_long.csv"
    df.drop(columns=["method_order"]).to_csv(long_csv_path, index=False)

    ndcg_pivot = (
        df.pivot(index="pool_factor", columns="pool_method", values="mean_ndcg@10")
        .reindex(index=pool_factors, columns=[method_name for method_name, _ in methods])
        .reset_index()
    )
    ndcg_table_md_path = output_dir / "overview_mean_ndcg_at_10_table.md"
    with open(ndcg_table_md_path, "w") as f:
        f.write(ndcg_pivot.to_markdown(index=False, floatfmt=".6f"))
        f.write("\n")

    plt.figure(figsize=(10, 6))
    for method_name, _ in methods:
        method_df = df[df["pool_method"] == method_name].sort_values("pool_factor")
        plt.plot(
            method_df["pool_factor"],
            method_df["mean_ndcg@10"],
            marker="o",
            linewidth=2,
            label=method_name,
        )
    plt.title("NanoBEIR Mean ndcg@10 by Pool Factor and Pool Method")
    plt.xlabel("Pool Factor")
    plt.ylabel("Mean ndcg@10")
    plt.xticks(pool_factors)
    plt.grid(alpha=0.3)
    plt.legend(title="Pool Method")
    plt.tight_layout()

    ndcg_plot_path = output_dir / "overview_mean_ndcg_at_10_plot.png"
    plt.savefig(ndcg_plot_path, dpi=200)
    plt.close()

    best_idx = df["mean_ndcg@10"].idxmax()
    best_row = df.loc[best_idx]
    summary = {
        "best_pool_factor": int(best_row["pool_factor"]),
        "best_pool_method": best_row["pool_method"],
        "best_mean_ndcg_at_10": float(best_row["mean_ndcg@10"]),
        "n_runs": int(len(df)),
        "per_dataset_long_csv": str(per_dataset_csv_path.name),
        "mean_long_csv": str(long_csv_path.name),
        "table_markdown": str(ndcg_table_md_path.name),
        "plot_png": str(ndcg_plot_path.name),
    }
    summary_path = output_dir / "overview_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved per-dataset metrics CSV: {per_dataset_csv_path}")
    print(f"Saved mean metrics CSV: {long_csv_path}")
    print(f"Saved overview table Markdown: {ndcg_table_md_path}")
    print(f"Saved overview plot: {ndcg_plot_path}")
    print(f"Saved overview summary: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a NanoBEIR pooling sweep and aggregate retrieval metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Model name/path passed to run_nanobeir_evaluation.py.",
    )
    parser.add_argument(
        "--pool-factors",
        type=int,
        nargs="+",
        default=DEFAULT_POOL_FACTORS,
        help="Pool factors to evaluate in order.",
    )
    parser.add_argument(
        "--pool-methods",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional pooling methods to sweep (space/comma separated). "
            "Available: hier, kmeans, slice (aliases: hierarchical, span)."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("examples/pooling/evaluation/results"),
        help="Root folder where a timestamped sweep directory will be created.",
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
        "--device",
        type=str,
        default="cuda",
        help="CUDA device string, e.g. cuda or cuda:0.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size passed to each evaluation run.",
    )
    parser.add_argument(
        "--query-length",
        type=int,
        default=DEFAULT_QUERY_LENGTH,
        help="Query token length passed to each evaluation run.",
    )
    parser.add_argument(
        "--document-length",
        type=int,
        default=DEFAULT_DOCUMENT_LENGTH,
        help="Document token length passed to each evaluation run.",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Pass --no-fp16 to each evaluation run.",
    )
    parser.add_argument(
        "--use-triton",
        action="store_true",
        help="Pass --use-triton to each evaluation run (k-means pooling).",
    )
    parser.add_argument(
        "--use-sklearn",
        action="store_true",
        help="Pass --use-sklearn to each evaluation run (k-means pooling).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs when the expected JSON result file already exists.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this sweep, but none is available.")
    if not args.device.startswith("cuda"):
        raise ValueError(f"Expected a CUDA device, got: {args.device!r}")

    use_triton: bool = args.use_triton
    use_sklearn: bool = args.use_sklearn
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
    selected_methods = _parse_selected_methods(args.pool_methods)

    this_file = Path(__file__).resolve()
    eval_script = this_file.with_name("run_nanobeir_evaluation.py")
    repo_root = this_file.parents[3]
    output_dir = build_sweep_run_dir(
        output_root=args.output_root,
        run_dir_prefix="nanobeir_pooling_sweep",
        reuse_latest=args.skip_existing,
    )

    print(f"Using GPU device: {args.device}")
    print("Datasets: " + ("all" if dataset_names is None else ", ".join(selected_datasets)))
    print(
        "Pool methods: "
        + ", ".join(method_name for method_name, _ in selected_methods)
    )
    print(
        "Run config: "
        f"batch_size={args.batch_size}, query_length={args.query_length}, "
        f"document_length={args.document_length}"
    )
    print(f"Sweep output directory: {output_dir}")
    print(f"Evaluation script: {eval_script}")

    combinations = [
        (pool_factor, method_idx, method_name, method_cli)
        for pool_factor in args.pool_factors
        for method_idx, (method_name, method_cli) in enumerate(selected_methods)
    ]

    run_records = []
    progress = tqdm(combinations, desc="NanoBEIR Sweep", unit="run")
    for pool_factor, method_idx, method_name, method_cli in progress:
        progress.set_postfix(pool_factor=pool_factor, method=method_name)
        result_path = _result_json_path(
            output_dir=output_dir,
            model_name=args.model,
            pool_factor=pool_factor,
            method_cli=method_cli,
            dataset_names=dataset_names,
            query_length=args.query_length,
            document_length=args.document_length,
        )
        if args.skip_existing and result_path.exists():
            run_records.append(
                {
                    "pool_factor": pool_factor,
                    "method_order": method_idx,
                    "pool_method": method_name,
                    "pool_method_cli": method_cli,
                    "dataset_names": selected_datasets,
                    "results_json": result_path,
                }
            )
            continue

        _run_single_evaluation(
            eval_script=eval_script,
            repo_root=repo_root,
            output_dir=output_dir,
            model_name=args.model,
            pool_factor=pool_factor,
            method_cli=method_cli,
            device=args.device,
            batch_size=args.batch_size,
            query_length=args.query_length,
            document_length=args.document_length,
            no_fp16=args.no_fp16,
            dataset_names=dataset_names,
            use_triton=use_triton,
            use_sklearn=use_sklearn,
        )
        run_records.append(
            {
                "pool_factor": pool_factor,
                "method_order": method_idx,
                "pool_method": method_name,
                "pool_method_cli": method_cli,
                "dataset_names": selected_datasets,
                "results_json": result_path,
            }
        )

    if args.skip_existing:
        discovered_records = discover_sweep_result_records(
            output_dir=output_dir,
            file_pattern=f"nanobeir_{safe_name(value=args.model)}_pool*_*.json",
            model_name=args.model,
            methods=selected_methods,
            filters={
                "dataset_names": selected_datasets,
                "query_length": args.query_length,
                "document_length": args.document_length,
            },
        )
        if discovered_records:
            run_records = discovered_records
            print(
                f"Aggregating {len(run_records)} existing/new results from {output_dir}"
            )

    if not run_records:
        raise FileNotFoundError(
            f"No result JSON files found in {output_dir} for model {args.model!r} "
            f"and datasets {selected_datasets}."
        )

    rows = []
    per_dataset_rows = []
    load_progress = tqdm(run_records, desc="Loading Results", unit="file")
    for record in load_progress:
        result_path = Path(record["results_json"])
        if not result_path.exists():
            raise FileNotFoundError(f"Missing result JSON: {result_path}")

        with open(result_path) as f:
            data = json.load(f)

        dataset_results, mean_results = _parse_nanobeir_scores(
            scores=data.get("scores", {})
        )

        rows.append(
            {
                "pool_factor": int(record["pool_factor"]),
                "method_order": int(record["method_order"]),
                "pool_method": record["pool_method"],
                "pool_method_cli": record["pool_method_cli"],
                "mean_ndcg@10": mean_results.get("ndcg@10"),
                "mean_mrr@10": mean_results.get("mrr@10"),
                "mean_map@100": mean_results.get("map@100"),
                "mean_recall@10": mean_results.get("recall@10"),
                "mean_recall@100": mean_results.get("recall@100"),
                "mean_accuracy@10": mean_results.get("accuracy@10"),
                "evaluation_time_seconds": float(data["evaluation_time_seconds"]),
                "results_json": result_path.name,
            }
        )

        for dataset_name, dataset_metrics in dataset_results.items():
            per_dataset_rows.append(
                {
                    "dataset_name": dataset_name,
                    "pool_factor": int(record["pool_factor"]),
                    "method_order": int(record["method_order"]),
                    "pool_method": record["pool_method"],
                    "pool_method_cli": record["pool_method_cli"],
                    "ndcg@10": dataset_metrics.get("ndcg@10"),
                    "mrr@10": dataset_metrics.get("mrr@10"),
                    "map@100": dataset_metrics.get("map@100"),
                    "recall@10": dataset_metrics.get("recall@10"),
                    "recall@100": dataset_metrics.get("recall@100"),
                    "accuracy@10": dataset_metrics.get("accuracy@10"),
                    "results_json": result_path.name,
                }
            )

    _write_overview_artifacts(
        output_dir=output_dir,
        rows=rows,
        per_dataset_rows=per_dataset_rows,
        pool_factors=sorted({int(record["pool_factor"]) for record in run_records}),
        methods=selected_methods,
    )


if __name__ == "__main__":
    main()
