#!/usr/bin/env python3
"""Rerank mined BEIR training tuples and build a final training split."""

from __future__ import annotations

import argparse
import inspect
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict, load_from_disk
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from pylate import evaluation

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

DEFAULT_PROMPT = (
    "Given a query A and a passage B, determine whether the passage contains "
    "an answer to the query by providing a prediction of either 'Yes' or 'No'."
)


def load_beir_train_split(dataset_name: str) -> tuple[list[dict[str, str]], dict[str, str], dict]:
    """Load one BEIR dataset train split, including cqadupstack subsets."""
    if dataset_name.startswith("cqadupstack/"):
        from beir import util as beir_util

        beir_util.download_and_unzip(
            url=(
                "https://public.ukp.informatik.tu-darmstadt.de/"
                "thakur/BEIR/datasets/cqadupstack.zip"
            ),
            out_dir="./evaluation_datasets/",
        )
        return evaluation.load_custom_dataset(
            f"evaluation_datasets/{dataset_name}",
            split="train",
        )

    return evaluation.load_beir(dataset_name=dataset_name, split="train")


def infer_dataset_name(input_path: Path) -> str | None:
    """Infer BEIR dataset name from mined dataset path."""
    stem = input_path.name.removesuffix("_mined")
    if stem in ALL_BEIR_DATASETS:
        return stem

    # Handle cqadupstack paths like .../cqadupstack/android_mined
    if input_path.parent.name == "cqadupstack":
        candidate = f"cqadupstack/{stem}"
        if candidate in ALL_BEIR_DATASETS:
            return candidate

    return None


def get_llm_inputs(
    pairs: list[tuple[str, str]],
    tokenizer,
    max_length: int,
    prompt: str = DEFAULT_PROMPT,
) -> dict[str, torch.Tensor]:
    """Build LLM reranker inputs following BGE reranker docs."""
    sep = "\n"
    prompt_inputs = tokenizer(
        prompt,
        return_tensors=None,
        add_special_tokens=False,
    )["input_ids"]
    sep_inputs = tokenizer(
        sep,
        return_tensors=None,
        add_special_tokens=False,
    )["input_ids"]

    if tokenizer.bos_token_id is None:
        raise ValueError("Tokenizer does not define bos_token_id required by LLM reranker format.")

    prepared_inputs = []
    for query, passage in pairs:
        query_inputs = tokenizer(
            f"A: {query}",
            return_tensors=None,
            add_special_tokens=False,
            max_length=max_length * 3 // 4,
            truncation=True,
        )
        passage_inputs = tokenizer(
            f"B: {passage}",
            return_tensors=None,
            add_special_tokens=False,
            max_length=max_length,
            truncation=True,
        )

        item = tokenizer.prepare_for_model(
            [tokenizer.bos_token_id] + query_inputs["input_ids"],
            sep_inputs + passage_inputs["input_ids"],
            truncation="only_second",
            max_length=max_length,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=False,
        )
        item["input_ids"] = item["input_ids"] + sep_inputs + prompt_inputs
        item["attention_mask"] = [1] * len(item["input_ids"])
        prepared_inputs.append(item)

    return tokenizer.pad(
        prepared_inputs,
        padding=True,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )


def score_pairs(
    pairs: list[tuple[str, str]],
    model_name: str,
    batch_size: int,
    max_length: int,
    device: str,
    trust_remote_code: bool,
) -> list[float]:
    """Score query-document pairs with a HF reranker model."""
    if not pairs:
        return []

    print(f"Loading reranker '{model_name}' on device={device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model_load_kwargs = {"trust_remote_code": trust_remote_code}
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError(f"CUDA device requested ({device}) but torch.cuda.is_available() is False.")
        model_load_kwargs["dtype"] = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        model_load_kwargs["low_cpu_mem_usage"] = True

    is_seq_cls = any(
        "SequenceClassification" in arch
        for arch in (config.architectures or [])
    )

    if is_seq_cls:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            **model_load_kwargs,
        )
        scoring_mode = "sequence_classification"
        print("Using sequence-classification scoring.")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_load_kwargs,
        )
        scoring_mode = "causal_lm_yes_logit"
        supports_logits_to_keep = "logits_to_keep" in inspect.signature(model.forward).parameters
        yes_tokens = tokenizer("Yes", add_special_tokens=False)["input_ids"]
        if not yes_tokens:
            raise ValueError("Could not tokenize 'Yes' for LLM reranker scoring.")
        yes_token_id = yes_tokens[0]
        if supports_logits_to_keep:
            print("Using causal-LM Yes-logit scoring with logits_to_keep=1.")
        else:
            print("Using causal-LM Yes-logit scoring.")

    model = model.to(device)
    model.eval()

    scores: list[float] = []
    total = len(pairs)
    with torch.inference_mode():
        for start in tqdm(
            range(0, total, batch_size),
            desc="Scoring pairs",
            total=(total + batch_size - 1) // batch_size,
        ):
            end = min(start + batch_size, total)
            batch_pairs = pairs[start:end]

            if scoring_mode == "sequence_classification":
                inputs = tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=max_length,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                batch_scores = model(**inputs, return_dict=True).logits.view(-1).float()
            else:
                inputs = get_llm_inputs(
                    pairs=batch_pairs,
                    tokenizer=tokenizer,
                    max_length=max_length,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                model_kwargs = {
                    "return_dict": True,
                    "use_cache": False,
                }
                if supports_logits_to_keep:
                    model_kwargs["logits_to_keep"] = 1
                logits = model(**inputs, **model_kwargs).logits
                batch_scores = logits[:, -1, yes_token_id].view(-1).float()

            scores.extend(batch_scores.detach().cpu().tolist())

    return scores


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rerank mined BEIR tuples and build filtered final training examples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="Path to mined DatasetDict saved with train split.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help=(
            "BEIR dataset name for loading raw query/doc text. "
            "If unset, inferred from input path name by removing '_mined'."
        ),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="BAAI/bge-reranker-v2-gemma",
        help="HF reranker model name.",
    )
    parser.add_argument(
        "--negative-threshold-share",
        type=float,
        default=0.95,
        help=(
            "Drop negatives with score > share * min_positive_score for that query."
        ),
    )
    parser.add_argument("--n-positives", type=int, default=1)
    parser.add_argument("--n-hard-negatives", type=int, default=5)
    parser.add_argument("--n-medium-negatives", type=int, default=5)
    parser.add_argument("--n-random-negatives", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device (cuda/cpu). Auto-selects cuda if available.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading tokenizer/model.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output directory for final DatasetDict(train).",
    )
    parser.add_argument(
        "--save-every-queries",
        type=int,
        default=1000,
        help="Save intermediate output every N processed queries. Set 0 to disable.",
    )
    args = parser.parse_args()

    if not args.input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {args.input_path}")

    if not (0.0 <= args.negative_threshold_share <= 1.0):
        raise ValueError("--negative-threshold-share must be between 0 and 1.")
    if args.save_every_queries < 0:
        raise ValueError("--save-every-queries must be >= 0.")

    for name, value in [
        ("--n-positives", args.n_positives),
        ("--n-hard-negatives", args.n_hard_negatives),
        ("--n-medium-negatives", args.n_medium_negatives),
        ("--n-random-negatives", args.n_random_negatives),
        ("--batch-size", args.batch_size),
        ("--max-length", args.max_length),
    ]:
        if value <= 0:
            raise ValueError(f"{name} must be > 0.")

    dataset_name = args.dataset_name
    if dataset_name is None:
        inferred = infer_dataset_name(args.input_path)
        if inferred is None:
            raise ValueError(
                "Could not infer dataset name from input path. "
                "Please provide --dataset-name."
            )
        dataset_name = inferred

    dataset_name = dataset_name.strip().lower()
    if dataset_name not in ALL_BEIR_DATASETS:
        raise ValueError(
            f"Unknown dataset: {dataset_name!r}. Available: {ALL_BEIR_DATASETS}"
        )

    if args.output_path is None:
        output_path = args.input_path.parent / f"{dataset_name}_scored"
    else:
        output_path = args.output_path / f"{dataset_name}_scored"
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading mined data from {args.input_path}...")
    mined_data = load_from_disk(str(args.input_path))
    if isinstance(mined_data, DatasetDict):
        if "train" not in mined_data:
            raise ValueError("Input DatasetDict must contain a 'train' split.")
        train_ds = mined_data["train"]
    else:
        train_ds = mined_data

    required_columns = {
        "query_id",
        "positive_ids",
        "hard_negative_ids",
        "random_negative_ids",
    }
    missing = required_columns - set(train_ds.column_names)
    if missing:
        raise ValueError(f"Input dataset is missing required columns: {sorted(missing)}")

    print(f"Loading BEIR train split '{dataset_name}' to recover texts...")
    try:
        corpus, queries, _ = load_beir_train_split(dataset_name)
    except Exception as exc:
        print(
            f"Could not load train split for dataset '{dataset_name}'. Error: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    doc_by_id = {doc["id"]: doc["text"] for doc in corpus}
    print(f"Recovered texts: queries={len(queries)}, documents={len(doc_by_id)}")

    # Flatten all pairs once for efficient batched scoring.
    pair_index_to_meta: list[tuple[str, str, str]] = []
    text_pairs: list[tuple[str, str]] = []

    print("Building query-document pairs...")
    for row in train_ds:
        query_id = str(row["query_id"])
        query_text = queries.get(query_id)
        if query_text is None:
            continue

        for label, field in [
            ("positive", "positive_ids"),
            ("hard_negative", "hard_negative_ids"),
            ("random_negative", "random_negative_ids"),
        ]:
            for doc_id in row[field]:
                doc_id = str(doc_id)
                doc_text = doc_by_id.get(doc_id)
                if doc_text is None:
                    continue
                pair_index_to_meta.append((query_id, doc_id, label))
                text_pairs.append((query_text, doc_text))

    if not text_pairs:
        raise ValueError("No valid (query, doc) pairs found after ID-to-text mapping.")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    scores = score_pairs(
        pairs=text_pairs,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
        trust_remote_code=args.trust_remote_code,
    )

    scored_by_query: dict[str, dict[str, list[tuple[str, float]]]] = defaultdict(
        lambda: {
            "positive": [],
            "hard_negative": [],
            "random_negative": [],
        }
    )

    for (query_id, doc_id, label), score in zip(pair_index_to_meta, scores):
        scored_by_query[query_id][label].append((doc_id, float(score)))

    rng = random.Random(args.seed)

    output_rows = {
        "query_id": [],
        "document_ids": [],
        "scores": [],
        "labels": [],
    }

    print("Applying filtering and final selection...")
    skipped_queries = 0

    for idx, query_id in enumerate(train_ds["query_id"], start=1):
        query_id = str(query_id)
        group = scored_by_query.get(query_id)
        if group is None:
            skipped_queries += 1
            if args.save_every_queries > 0 and idx % args.save_every_queries == 0:
                partial_dataset = Dataset.from_dict(output_rows)
                DatasetDict({"train": partial_dataset}).save_to_disk(str(output_path))
                print(
                    f"Checkpoint: saved {len(partial_dataset)} rows after "
                    f"{idx}/{len(train_ds)} processed queries to {output_path}"
                )
            continue

        positives = sorted(group["positive"], key=lambda x: x[1], reverse=True)
        hard_negs = group["hard_negative"]
        random_negs = group["random_negative"]

        if not positives:
            skipped_queries += 1
            continue

        min_positive_score = min(score for _, score in positives)
        threshold = args.negative_threshold_share * min_positive_score

        hard_negs = [item for item in hard_negs if item[1] <= threshold]
        random_negs = [item for item in random_negs if item[1] <= threshold]

        selected_positives = positives[: args.n_positives]

        hard_sorted = sorted(hard_negs, key=lambda x: x[1], reverse=True)
        selected_hard = hard_sorted[: args.n_hard_negatives]

        remaining_hard = hard_sorted[args.n_hard_negatives :]
        medium_k = min(args.n_medium_negatives, len(remaining_hard))
        selected_medium = rng.sample(remaining_hard, k=medium_k)

        random_k = min(args.n_random_negatives, len(random_negs))
        selected_random = rng.sample(random_negs, k=random_k)

        selected = [
            (doc_id, score, "positive")
            for doc_id, score in selected_positives
        ]
        selected.extend(
            (doc_id, score, "hard_negative")
            for doc_id, score in selected_hard
        )
        selected.extend(
            (doc_id, score, "medium_negative")
            for doc_id, score in selected_medium
        )
        selected.extend(
            (doc_id, score, "random_negative")
            for doc_id, score in selected_random
        )

        selected.sort(key=lambda x: x[1], reverse=True)

        output_rows["query_id"].append(query_id)
        output_rows["document_ids"].append([doc_id for doc_id, _, _ in selected])
        output_rows["scores"].append([score for _, score, _ in selected])
        output_rows["labels"].append([label for _, _, label in selected])

        if args.save_every_queries > 0 and idx % args.save_every_queries == 0:
            partial_dataset = Dataset.from_dict(output_rows)
            DatasetDict({"train": partial_dataset}).save_to_disk(str(output_path))
            print(
                f"Checkpoint: saved {len(partial_dataset)} rows after "
                f"{idx}/{len(train_ds)} processed queries to {output_path}"
            )

    output_dataset = Dataset.from_dict(output_rows)
    output_dict = DatasetDict({"train": output_dataset})

    print(f"Saving final dataset to {output_path}...")
    output_dict.save_to_disk(str(output_path))

    print(
        "Done. "
        f"Saved split='train' with {len(output_dataset)} rows. "
        f"Skipped {skipped_queries} queries (missing text or positives)."
    )


if __name__ == "__main__":
    main()
