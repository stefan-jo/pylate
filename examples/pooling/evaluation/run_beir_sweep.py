#!/usr/bin/env python3
"""Run a BEIR pooling sweep and aggregate retrieval metrics.

This script executes ``run_beir_evaluation.py`` for combinations of:
- datasets
- pool factors
- pooling methods

It writes:
- per-dataset metrics (long CSV)
- mean metrics aggregated across datasets (long CSV)
- ndcg@10 overview table (Markdown)
- ndcg@10 overview plot
- summary JSON
"""

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
METHODS = [
    ("hier", "hierarchical"),
    ("kmeans", "kmeans"),
    ("slice", "span"),
]
METRICS = ["ndcg@10", "mrr@10", "map@100", "recall@10", "recall@100"]


def _safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "_")


def _safe_dataset_name(dataset_name: str) -> str:
    return dataset_name.replace("/", "_")


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


def _build_run_dir(output_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"beir_pooling_sweep_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _run_single_evaluation(
    eval_script: Path,
    repo_root: Path,
    output_dir: Path,
    model_name: str,
    pool_factor: int,
    method_cli: str,
    dataset_name: str,
    device: str,
    no_fp16: bool,
    use_triton: bool | None = None,
    batch_size: int = 16,
    k: int = 20,
    document_length: int = 180,
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
        "--dataset-name",
        dataset_name,
        "--output-dir",
        str(output_dir),
        "--device",
        device,
        "--batch-size",
        str(batch_size),
        "--k",
        str(k),
        "--document-length",
        str(document_length),
    ]
    if no_fp16:
        command.append("--no-fp16")
    if use_triton:
        command.append("--use-triton")

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
    dataset_name: str,
) -> Path:
    safe_name = _safe_model_name(model_name=model_name)
    safe_dataset = _safe_dataset_name(dataset_name=dataset_name)
    return output_dir / (
        f"beir_{safe_name}_pool{pool_factor}_{method_cli}_ds_{safe_dataset}.json"
    )


def _extract_metric(data: dict, metric: str, dataset_name: str) -> float | None:
    scores = data.get("scores", {})
    if not isinstance(scores, dict):
        return None

    dataset_scores = scores.get(dataset_name)
    if isinstance(dataset_scores, dict) and metric in dataset_scores:
        return float(dataset_scores[metric])

    for value in scores.values():
        if isinstance(value, dict) and metric in value:
            return float(value[metric])

    return None


def _write_overview_artifacts(
    output_dir: Path,
    rows: list[dict],
    pool_factors: list[int],
    selected_datasets: list[str],
) -> None:
    df = pd.DataFrame(rows)
    df = df.sort_values(["dataset_name", "pool_factor", "method_order"]).reset_index(
        drop=True
    )

    per_dataset_csv_path = output_dir / "overview_per_dataset_metrics_long.csv"
    df.drop(columns=["method_order"]).to_csv(per_dataset_csv_path, index=False)

    agg_df = df.groupby(
        ["pool_factor", "method_order", "pool_method", "pool_method_cli"],
        as_index=False,
    )[METRICS].mean()
    agg_df = agg_df.rename(columns={metric: f"mean_{metric}" for metric in METRICS})
    agg_df = agg_df.sort_values(["pool_factor", "method_order"]).reset_index(drop=True)

    long_csv_path = output_dir / "overview_mean_metrics_long.csv"
    agg_df.drop(columns=["method_order"]).to_csv(long_csv_path, index=False)

    ndcg_pivot = (
        agg_df.pivot(index="pool_factor", columns="pool_method", values="mean_ndcg@10")
        .reindex(index=pool_factors, columns=[method_name for method_name, _ in METHODS])
        .reset_index()
    )
    ndcg_table_md_path = output_dir / "overview_mean_ndcg_at_10_table.md"
    with open(ndcg_table_md_path, "w") as f:
        f.write(ndcg_pivot.to_markdown(index=False, floatfmt=".6f"))
        f.write("\n")

    plt.figure(figsize=(10, 6))
    for method_name, _ in METHODS:
        method_df = agg_df[agg_df["pool_method"] == method_name].sort_values(
            "pool_factor"
        )
        plt.plot(
            method_df["pool_factor"],
            method_df["mean_ndcg@10"],
            marker="o",
            linewidth=2,
            label=method_name,
        )
    plt.title("BEIR Mean ndcg@10 by Pool Factor and Pool Method")
    plt.xlabel("Pool Factor")
    plt.ylabel("Mean ndcg@10")
    plt.xticks(pool_factors)
    plt.grid(alpha=0.3)
    plt.legend(title="Pool Method")
    plt.tight_layout()

    ndcg_plot_path = output_dir / "overview_mean_ndcg_at_10_plot.png"
    plt.savefig(ndcg_plot_path, dpi=200)
    plt.close()

    best_idx = agg_df["mean_ndcg@10"].idxmax()
    best_row = agg_df.loc[best_idx]
    summary = {
        "best_pool_factor": int(best_row["pool_factor"]),
        "best_pool_method": best_row["pool_method"],
        "best_mean_ndcg_at_10": float(best_row["mean_ndcg@10"]),
        "n_datasets": int(len(selected_datasets)),
        "n_runs": int(len(df)),
        "n_combinations": int(len(agg_df)),
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
        description="Run a BEIR pooling sweep and aggregate retrieval metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Model name/path passed to run_beir_evaluation.py.",
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
            "Optional BEIR subset (space/comma separated). "
            f"Available: {', '.join(ALL_BEIR_DATASETS)}"
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device string passed to each run (e.g. cuda, cuda:0, cpu).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size passed to each evaluation run.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=20,
        help="Top-k passed to each evaluation run.",
    )
    parser.add_argument(
        "--document-length",
        type=int,
        default=180,
        help="Document length passed to each evaluation run.",
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

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested CUDA device {args.device!r}, but CUDA is not available."
        )

    use_triton: bool | None = True if args.use_triton else None
    dataset_names = _normalize_dataset_names(args.datasets)
    if dataset_names is not None:
        invalid_datasets = sorted(set(dataset_names) - set(ALL_BEIR_DATASETS))
        if invalid_datasets:
            raise ValueError(
                f"Unknown datasets: {invalid_datasets}. "
                f"Available: {ALL_BEIR_DATASETS}"
            )
        selected_datasets = dataset_names
    else:
        selected_datasets = ALL_BEIR_DATASETS

    this_file = Path(__file__).resolve()
    eval_script = this_file.with_name("run_beir_evaluation.py")
    repo_root = this_file.parents[3]
    output_dir = _build_run_dir(output_root=args.output_root)

    print(f"Using device: {args.device}")
    print("Datasets: " + ("all" if dataset_names is None else ", ".join(selected_datasets)))
    print(f"Sweep output directory: {output_dir}")
    print(f"Evaluation script: {eval_script}")

    combinations = [
        (dataset_name, pool_factor, method_idx, method_name, method_cli)
        for dataset_name in selected_datasets
        for pool_factor in args.pool_factors
        for method_idx, (method_name, method_cli) in enumerate(METHODS)
    ]

    run_records = []
    progress = tqdm(combinations, desc="BEIR Sweep", unit="run")
    for dataset_name, pool_factor, method_idx, method_name, method_cli in progress:
        progress.set_postfix(
            dataset=dataset_name, pool_factor=pool_factor, method=method_name
        )

        result_path = _result_json_path(
            output_dir=output_dir,
            model_name=args.model,
            pool_factor=pool_factor,
            method_cli=method_cli,
            dataset_name=dataset_name,
        )
        if args.skip_existing and result_path.exists():
            run_records.append(
                {
                    "dataset_name": dataset_name,
                    "pool_factor": pool_factor,
                    "method_order": method_idx,
                    "pool_method": method_name,
                    "pool_method_cli": method_cli,
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
            dataset_name=dataset_name,
            device=args.device,
            no_fp16=args.no_fp16,
            use_triton=use_triton,
            batch_size=args.batch_size,
            k=args.k,
            document_length=args.document_length,
        )
        run_records.append(
            {
                "dataset_name": dataset_name,
                "pool_factor": pool_factor,
                "method_order": method_idx,
                "pool_method": method_name,
                "pool_method_cli": method_cli,
                "results_json": result_path,
            }
        )

    rows = []
    load_progress = tqdm(run_records, desc="Loading Results", unit="file")
    for record in load_progress:
        result_path = Path(record["results_json"])
        if not result_path.exists():
            raise FileNotFoundError(f"Missing result JSON: {result_path}")

        with open(result_path) as f:
            data = json.load(f)

        ndcg_at_10 = _extract_metric(
            data=data,
            metric="ndcg@10",
            dataset_name=record["dataset_name"],
        )
        if ndcg_at_10 is None:
            raise ValueError(
                f"Could not find ndcg@10 in result file: {result_path}"
            )

        rows.append(
            {
                "dataset_name": record["dataset_name"],
                "pool_factor": int(record["pool_factor"]),
                "method_order": int(record["method_order"]),
                "pool_method": record["pool_method"],
                "pool_method_cli": record["pool_method_cli"],
                "ndcg@10": float(ndcg_at_10),
                "mrr@10": _extract_metric(
                    data=data,
                    metric="mrr@10",
                    dataset_name=record["dataset_name"],
                ),
                "map@100": _extract_metric(
                    data=data,
                    metric="map@100",
                    dataset_name=record["dataset_name"],
                ),
                "recall@10": _extract_metric(
                    data=data,
                    metric="recall@10",
                    dataset_name=record["dataset_name"],
                ),
                "recall@100": _extract_metric(
                    data=data,
                    metric="recall@100",
                    dataset_name=record["dataset_name"],
                ),
                "evaluation_time_seconds": float(data["evaluation_time_seconds"]),
                "results_json": result_path.name,
            }
        )

    _write_overview_artifacts(
        output_dir=output_dir,
        rows=rows,
        pool_factors=args.pool_factors,
        selected_datasets=selected_datasets,
    )


if __name__ == "__main__":
    main()
