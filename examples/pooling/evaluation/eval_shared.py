"""Shared helpers for pooling evaluation scripts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import hashlib
from typing import Any, Mapping

import torch


def safe_name(value: str) -> str:
    """Normalize a string for filenames/paths."""
    return value.replace("/", "_").replace(" ", "_")


def model_size_mb(model: torch.nn.Module) -> float:
    """Return model size in MB (accounting for dtype)."""
    total_bytes = 0
    for parameter in model.parameters():
        if parameter.dtype == torch.float32:
            elem_size = 4
        elif parameter.dtype in (torch.float16, torch.bfloat16):
            elem_size = 2
        else:
            elem_size = parameter.element_size()
        total_bytes += parameter.numel() * elem_size
    return total_bytes / (1024**2)


class PooledColBERT:
    """Wrapper that applies pooling args when encoding documents."""

    def __init__(
        self,
        model,
        pool_factor: int = 1,
        pool_method: str = "hierarchical",
        use_triton: bool | None = None,
        use_sklearn: bool = False,
    ) -> None:
        self.model = model
        self.pool_factor = pool_factor
        self.pool_method = pool_method
        self.use_triton = use_triton
        self.use_sklearn = use_sklearn

    def encode(self, sentences, is_query: bool = True, **kwargs):
        """Encode with pooling args applied for documents."""
        if not is_query:
            kwargs["pool_factor"] = self.pool_factor
            kwargs["pool_method"] = self.pool_method
            if self.use_triton is not None:
                kwargs["use_triton"] = self.use_triton
            kwargs["use_sklearn"] = self.use_sklearn
        return self.model.encode(sentences, is_query=is_query, **kwargs)

    def __getattr__(self, name):
        """Delegate all other attributes to the underlying model."""
        return getattr(self.model, name)


def scores_to_serializable(scores: Mapping[str, float | int]) -> dict[str, float]:
    """Convert score mappings to JSON-serializable float values."""
    return {key: float(value) for key, value in scores.items()}


def normalize_cli_list(
    values: list[str] | None,
    *,
    lowercase: bool = True,
    unique: bool = False,
) -> list[str] | None:
    """Normalize list CLI args that may include comma-separated values."""
    if values is None:
        return None

    normalized = []
    seen = set()
    for token in values:
        for entry in token.split(","):
            clean_entry = entry.strip()
            if not clean_entry:
                continue
            if lowercase:
                clean_entry = clean_entry.lower()
            if unique:
                if clean_entry in seen:
                    continue
                seen.add(clean_entry)
            normalized.append(clean_entry)

    return normalized or None


def dataset_suffix(
    dataset_names: list[str] | None,
    *,
    safe: bool = False,
    max_length: int | None = None,
) -> str:
    """Build a filename suffix representing dataset subset selection."""
    if not dataset_names:
        return ""

    names = [safe_name(name) if safe else name for name in dataset_names]
    joined = "-".join(names)

    if max_length is not None and len(joined) > max_length:
        digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:10]
        return f"_ds_{len(dataset_names)}_{digest}"

    return "_ds_" + joined


def find_latest_sweep_dir(output_root: Path, run_dir_prefix: str) -> Path | None:
    """Find the most recent existing sweep directory under *output_root*."""
    if not output_root.is_dir():
        return None
    candidates = sorted(
        (
            directory
            for directory in output_root.iterdir()
            if directory.is_dir() and directory.name.startswith(f"{run_dir_prefix}_")
        ),
        key=lambda directory: directory.name,
        reverse=True,
    )
    return candidates[0] if candidates else None


def build_sweep_run_dir(
    output_root: Path,
    run_dir_prefix: str,
    *,
    reuse_latest: bool = False,
) -> Path:
    """Build or reuse a timestamped sweep run directory."""
    if reuse_latest:
        latest = find_latest_sweep_dir(
            output_root=output_root, run_dir_prefix=run_dir_prefix
        )
        if latest is not None:
            return latest
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"{run_dir_prefix}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_eval_subprocess(eval_script: Path, repo_root: Path, cli_args: list[str]) -> None:
    """Run an evaluation script in a subprocess with repo-root on PYTHONPATH."""
    command = [sys.executable, str(eval_script), *cli_args]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(repo_root)
        if not existing_pythonpath
        else f"{repo_root}:{existing_pythonpath}"
    )
    subprocess.run(command, cwd=repo_root, check=True, env=env)


def discover_sweep_result_records(
    output_dir: Path,
    *,
    file_pattern: str,
    model_name: str,
    methods: list[tuple[str, str]],
    filters: Mapping[str, Any] | None = None,
    include_fields: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """Discover sweep result JSON files and convert them into sortable run records."""
    method_lookup = {
        method_cli: (method_idx, method_name)
        for method_idx, (method_name, method_cli) in enumerate(methods)
    }
    records: list[dict[str, Any]] = []

    for result_path in sorted(output_dir.glob(file_pattern)):
        with open(result_path) as f:
            data = json.load(f)

        if data.get("model") != model_name:
            continue

        if filters is not None and any(
            data.get(field_name) != expected_value
            for field_name, expected_value in filters.items()
        ):
            continue

        method_cli = data.get("pool_method")
        if method_cli not in method_lookup:
            continue
        method_order, method_name = method_lookup[method_cli]

        try:
            pool_factor = int(data["pool_factor"])
        except (KeyError, TypeError, ValueError):
            continue

        record: dict[str, Any] = {
            "pool_factor": pool_factor,
            "method_order": method_order,
            "pool_method": method_name,
            "pool_method_cli": method_cli,
            "results_json": result_path,
        }

        is_valid_record = True
        for field_name in include_fields:
            if field_name not in data:
                is_valid_record = False
                break
            record[field_name] = data[field_name]
        if not is_valid_record:
            continue

        records.append(record)

    records.sort(key=lambda record: (record["pool_factor"], record["method_order"]))
    return records
