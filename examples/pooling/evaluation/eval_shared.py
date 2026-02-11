"""Shared helpers for pooling evaluation scripts."""

from __future__ import annotations

import hashlib
from typing import Mapping

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
    ) -> None:
        self.model = model
        self.pool_factor = pool_factor
        self.pool_method = pool_method
        self.use_triton = use_triton

    def encode(self, sentences, is_query: bool = True, **kwargs):
        """Encode with pooling args applied for documents."""
        if not is_query:
            kwargs["pool_factor"] = self.pool_factor
            kwargs["pool_method"] = self.pool_method
            if self.use_triton is not None:
                kwargs["use_triton"] = self.use_triton
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
