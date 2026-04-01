#!/usr/bin/env python3
"""Aggregate per-dataset BEIR sweep metrics from JSON result files.

This script is a scriptified version of
``analyze_results_model_metrics_from_json.ipynb`` tailored to JSON files
produced by ``run_beir_sweep.py``.

By default it reads:
- ``examples/pooling/evaluation/results/beir_selection/beir_pooling_sweep_20260216_112107``
- ``examples/pooling/evaluation/results/scifact-pool-kmeans-all/beir_selection/beir_pooling_sweep_20260220_101133``

And writes one combined long CSV with per-dataset rows for each model/method/pool
combination.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from eval_shared import normalize_cli_list

DEFAULT_METRICS = ["ndcg@10", "mrr@10", "map@100", "recall@10", "recall@100"]
DEFAULT_POOL_FACTORS = [1, 2, 3, 4, 5, 6]
DEFAULT_OUTPUT_PATH = Path(
    "examples/pooling/evaluation/results/beir_combined_model_metrics_v4.csv"
)
DEFAULT_RESULT_SOURCES = [
    Path(
        "examples/pooling/evaluation/results/beir_selection/"
        "beir_pooling_sweep_20260216_112107"
    ),
    Path(
        "examples/pooling/evaluation/results/scifact-pool-kmeans-all/"
        "beir_selection/beir_pooling_sweep_20260220_101133"
    ),
    Path(
        "examples/pooling/evaluation/results/scifact-pool-kmeans-all-v2/beir_selection/beir_pooling_sweep_20260324_053021"
    ),
    Path(
        "examples/pooling/evaluation/results/scifact-pool-hier-all/beir_selection/beir_pooling_sweep_20260323_171646"
    ),
    Path(
        "examples/pooling/evaluation/results/scifact-pool-hier-all-v2/beir_selection/beir_pooling_sweep_20260323_185250"
    ),
    Path(
        "examples/pooling/evaluation/results/scifact-pool-span-all/beir_selection/beir_pooling_sweep_20260325_081536"
    ),
    Path(
        "examples/pooling/evaluation/results/scifact-pool-span-all-v2/beir_selection/beir_pooling_sweep_20260325_085201"
    ),
    Path(
        "examples/pooling/evaluation/results/scifact-pool-kmeans-2/beir_selection/beir_pooling_sweep_20260325_074735"
    ),
    Path(
        "examples/pooling/evaluation/results/fiqa-pool-kmeans-all/beir_selection/beir_pooling_sweep_20260324_112146"
    ),
    Path(
        "examples/pooling/evaluation/results/fiqa-pool-hier-all/beir_selection_v2/beir_pooling_sweep_20260325_210237"
    ),
    Path(
        "examples/pooling/evaluation/results/fiqa-pool-span-all/beir_selection/beir_pooling_sweep_20260325_183140"
    ),
]


def normalize_method(pool_method_cli: Any, pool_method: Any, use_sklearn: bool) -> str:
    """Normalize pooling method labels across result payload variants."""
    raw_value = pool_method_cli if pd.notna(pool_method_cli) else pool_method
    method = str(raw_value if raw_value is not None else "").strip().lower()
    mapping = {"slice": "span", "hier": "hierarchical"}
    method = mapping.get(method, method)
    if method == "kmeans" and bool(use_sklearn):
        method = "kmeans_sk"
    return method if method else "unknown"


def parse_scores(scores: dict[str, Any], metrics: list[str]) -> dict[str, dict[str, float]]:
    """Extract per-dataset metric values from a score payload."""
    dataset_results: dict[str, dict[str, float]] = {}
    if not isinstance(scores, dict):
        return dataset_results

    for dataset_name, dataset_metrics in scores.items():
        if not isinstance(dataset_metrics, dict):
            continue
        metric_row: dict[str, float] = {}
        for metric in metrics:
            if metric not in dataset_metrics:
                continue
            try:
                metric_row[metric] = float(dataset_metrics[metric])
            except (TypeError, ValueError):
                metric_row[metric] = np.nan
        if metric_row:
            dataset_results[str(dataset_name)] = metric_row

    return dataset_results


def iter_result_payloads(results_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Load valid BEIR result payloads from one results directory."""
    payloads: list[tuple[Path, dict[str, Any]]] = []
    for json_path in sorted(results_dir.glob("*.json")):
        if json_path.name == "overview_summary.json":
            continue

        try:
            payload = json.loads(json_path.read_text())
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON file: {json_path.name}")
            continue

        if not isinstance(payload, dict):
            continue
        if "scores" not in payload or "model" not in payload:
            continue

        payloads.append((json_path, payload))

    return payloads


def as_int_or_nan(value: Any) -> int | float:
    """Convert to int where possible; otherwise return NaN."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return np.nan


def extract_model_name(model_path: str) -> str:
    """Derive a short model identifier from a full model path."""
    normalized = str(model_path).replace("\\", "/").strip("/")
    parts = [part for part in normalized.split("/") if part]
    if len(parts) >= 2:
        return parts[-2]
    return str(model_path)


def build_per_model_table(
    payloads: list[tuple[Path, dict[str, Any]]],
    source_dir: Path,
    target_model_path: str,
    metrics: list[str],
    pool_factors_to_include: set[int] | None,
) -> pd.DataFrame:
    """Build one long-form per-model table from JSON payloads."""
    rows: list[dict[str, Any]] = []

    for json_path, payload in payloads:
        if str(payload.get("model")) != target_model_path:
            continue

        dataset_results = parse_scores(payload.get("scores", {}), metrics)
        method = normalize_method(
            payload.get("pool_method_cli"),
            payload.get("pool_method"),
            bool(payload.get("use_sklearn", False)),
        )
        pool_factor = as_int_or_nan(payload.get("pool_factor"))
        if pool_factors_to_include is not None:
            if pd.isna(pool_factor) or int(pool_factor) not in pool_factors_to_include:
                continue

        for dataset_name, metric_values in dataset_results.items():
            row = {
                "model_name": extract_model_name(target_model_path),
                "dataset": str(dataset_name),
                "method": method,
                "pool_factor": pool_factor,
            }
            for metric in metrics:
                row[metric] = metric_values.get(metric, np.nan)
            row["model_path"] = target_model_path
            row["results_dir"] = str(source_dir)
            row["results_json"] = json_path.name
            rows.append(row)

    ordered_cols = [
        "model_name",
        "dataset",
        "method",
        "pool_factor",
        *metrics,
        "model_path",
        "results_dir",
        "results_json",
    ]
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=ordered_cols)

    out = (
        out[ordered_cols]
        .sort_values(
            [
                "model_name",
                "dataset",
                "method",
                "pool_factor",
                "results_dir",
                "results_json",
            ]
        )
        .reset_index(drop=True)
    )
    return out


def _parse_result_sources(values: list[str] | None) -> list[Path]:
    """Parse and normalize one-or-many source directory CLI args."""
    if values is None:
        sources = DEFAULT_RESULT_SOURCES
    else:
        normalized = normalize_cli_list(values, lowercase=False, unique=False)
        if normalized is None:
            raise ValueError("At least one --result-sources directory must be provided.")
        sources = [Path(value) for value in normalized]

    # Preserve input order while removing duplicates.
    seen_sources: set[str] = set()
    unique_sources: list[Path] = []
    for source_path in sources:
        key = str(source_path)
        if key in seen_sources:
            continue
        seen_sources.add(key)
        unique_sources.append(source_path)
    return unique_sources


def _parse_pool_factors(values: list[int] | None) -> set[int] | None:
    """Normalize pool factor filtering."""
    if values is None:
        return None
    return {int(value) for value in values}


def main() -> None:
    """Run aggregation and write a combined CSV."""
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate per-dataset metrics from BEIR sweep JSON files "
            "(run_beir_sweep.py outputs)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--result-sources",
        nargs="+",
        default=None,
        help=(
            "One or more BEIR sweep result directories. "
            "Accepts space/comma-separated values."
        ),
    )
    parser.add_argument(
        "--pool-factors",
        type=int,
        nargs="+",
        default=DEFAULT_POOL_FACTORS,
        help="Pool factors to include. Use --all-pool-factors to disable filtering.",
    )
    parser.add_argument(
        "--all-pool-factors",
        action="store_true",
        help="Disable pool factor filtering and keep all available pool factors.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help=(
            "Metrics to extract. "
            "Accepts space/comma-separated values (for example: ndcg@10,mrr@10)."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output CSV path for the combined long table.",
    )
    args = parser.parse_args()

    result_sources = _parse_result_sources(args.result_sources)
    if not result_sources:
        raise ValueError("No result directories were provided.")

    metrics = normalize_cli_list(args.metrics, lowercase=False, unique=True)
    if not metrics:
        raise ValueError("No metrics were provided.")

    pool_factor_set: set[int] | None
    if args.all_pool_factors:
        pool_factor_set = None
    else:
        pool_factor_set = _parse_pool_factors(args.pool_factors)

    per_model_tables: list[pd.DataFrame] = []
    for results_dir in result_sources:
        if not results_dir.exists():
            raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

        payloads = iter_result_payloads(results_dir)
        if not payloads:
            print(f"No usable JSON result files found in: {results_dir}. Skipping.")
            continue

        discovered_models = sorted({str(payload.get("model")) for _, payload in payloads})
        print(f"\nResults directory: {results_dir}")
        print(f"Usable JSON files: {len(payloads)}")
        print(f"Discovered model paths: {discovered_models}")
        if pool_factor_set is not None:
            print(f"Pool factors filter: {sorted(pool_factor_set)}")
        else:
            print("Pool factors filter: <all>")

        for model_path in discovered_models:
            model_df = build_per_model_table(
                payloads=payloads,
                source_dir=results_dir,
                target_model_path=model_path,
                metrics=metrics,
                pool_factors_to_include=pool_factor_set,
            )
            if model_df.empty:
                print(
                    f"No metric rows found for model path: {model_path} in {results_dir}"
                )
                continue

            print(
                f"Per-model table for '{model_path}' from "
                f"'{results_dir.name}': {len(model_df)} rows"
            )
            per_model_tables.append(model_df)

    if not per_model_tables:
        raise ValueError("No per-model tables were generated from result sources.")

    combined_metrics = (
        pd.concat(per_model_tables, ignore_index=True)
        .sort_values(
            [
                "model_name",
                "dataset",
                "method",
                "pool_factor",
                "results_dir",
                "results_json",
            ]
        )
        .reset_index(drop=True)
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_metrics.to_csv(args.output_path, index=False)

    print(f"\nSaved combined table to: {args.output_path}")
    print(f"Combined rows: {len(combined_metrics)}")


if __name__ == "__main__":
    main()
