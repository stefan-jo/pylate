#!/usr/bin/env python3
"""Run a NanoBEIR pooling sweep and aggregate retrieval metrics."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import pandas as pd
import torch
from tqdm.auto import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_MODEL = "mixedbread-ai/mxbai-edge-colbert-v0-32m"
DEFAULT_POOL_FACTORS = [1, 2, 3, 4, 5, 7, 10, 15, 20]
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


def _safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "_")


def _normalize_dataset_names(dataset_names: list[str] | None) -> list[str] | None:
    """Normalize dataset input from CLI (supports comma-separated tokens)."""
    if dataset_names is None:
        return None

    normalized = []
    for token in dataset_names:
        for name in token.split(","):
            clean_name = name.strip().lower()
            if clean_name:
                normalized.append(clean_name)

    return normalized or None


def _dataset_suffix(dataset_names: list[str] | None) -> str:
    """Return a filename suffix that captures dataset subset selection."""
    if not dataset_names:
        return ""
    return "_ds_" + "-".join(dataset_names)


def _build_run_dir(output_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"nanobeir_pooling_sweep_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _run_single_evaluation(
    eval_script: Path,
    repo_root: Path,
    output_dir: Path,
    model_name: str,
    pool_factor: int,
    method_cli: str,
    device: str,
    no_fp16: bool,
    dataset_names: list[str] | None,
    use_triton: bool | None = None,
) -> None:
    command = [
        sys.executable,
        str(eval_script),
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
    ]
    if no_fp16:
        command.append("--no-fp16")
    if use_triton:
        command.append("--use-triton")
    if dataset_names:
        command.extend(["--datasets", *dataset_names])

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(repo_root)
        if not existing_pythonpath
        else f"{repo_root}:{existing_pythonpath}"
    )

    subprocess.run(command, cwd=repo_root, check=True, env=env)


def _result_json_path(
    output_dir: Path,
    model_name: str,
    pool_factor: int,
    method_cli: str,
    dataset_names: list[str] | None,
) -> Path:
    safe_name = _safe_model_name(model_name=model_name)
    return output_dir / (
        f"nanobeir_{safe_name}_pool{pool_factor}_{method_cli}"
        f"{_dataset_suffix(dataset_names)}.json"
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
        .reindex(index=pool_factors, columns=[method_name for method_name, _ in METHODS])
        .reset_index()
    )
    ndcg_table_md_path = output_dir / "overview_mean_ndcg_at_10_table.md"
    with open(ndcg_table_md_path, "w") as f:
        f.write(ndcg_pivot.to_markdown(index=False, floatfmt=".6f"))
        f.write("\n")

    plt.figure(figsize=(10, 6))
    for method_name, _ in METHODS:
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
        "--skip-existing",
        action="store_true",
        help="Skip runs when the expected JSON result file already exists.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this sweep, but none is available.")
    if not args.device.startswith("cuda"):
        raise ValueError(f"Expected a CUDA device, got: {args.device!r}")

    use_triton: bool | None = True if args.use_triton else None
    dataset_names = _normalize_dataset_names(args.datasets)
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

    this_file = Path(__file__).resolve()
    eval_script = this_file.with_name("run_nanobeir_evaluation.py")
    repo_root = this_file.parents[3]
    output_dir = _build_run_dir(output_root=args.output_root)

    print(f"Using GPU device: {args.device}")
    print("Datasets: " + ("all" if dataset_names is None else ", ".join(selected_datasets)))
    print(f"Sweep output directory: {output_dir}")
    print(f"Evaluation script: {eval_script}")

    combinations = [
        (pool_factor, method_idx, method_name, method_cli)
        for pool_factor in args.pool_factors
        for method_idx, (method_name, method_cli) in enumerate(METHODS)
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
        )
        if args.skip_existing and result_path.exists():
            run_records.append(
                {
                    "pool_factor": pool_factor,
                    "method_order": method_idx,
                    "pool_method": method_name,
                    "pool_method_cli": method_cli,
                    "dataset_names": selected_datasets,
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
            no_fp16=args.no_fp16,
            dataset_names=dataset_names,
            use_triton=use_triton,
        )
        run_records.append(
            {
                "pool_factor": pool_factor,
                "method_order": method_idx,
                "pool_method": method_name,
                "pool_method_cli": method_cli,
                "dataset_names": selected_datasets,
            }
        )

    rows = []
    per_dataset_rows = []
    load_progress = tqdm(run_records, desc="Loading Results", unit="file")
    for record in load_progress:
        result_path = _result_json_path(
            output_dir=output_dir,
            model_name=args.model,
            pool_factor=record["pool_factor"],
            method_cli=record["pool_method_cli"],
            dataset_names=dataset_names,
        )
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
        pool_factors=args.pool_factors,
    )


if __name__ == "__main__":
    main()
