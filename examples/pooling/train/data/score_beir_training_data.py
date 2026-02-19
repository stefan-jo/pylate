#!/usr/bin/env python3
"""Rerank mined BEIR training tuples and build a final training split."""

from __future__ import annotations

import argparse
import inspect
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

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
CHECKPOINT_FILENAME = "scores_checkpoint.pt"


def compute_anchor_positive_score(
    positives: list[tuple[str, float]],
    metric: str,
    source: str,
    n_selected: int,
) -> float:
    """Compute anchor score for negative-threshold filtering."""
    if source == "all":
        selected = positives
    elif source == "selected":
        selected = positives[:n_selected]
    else:
        raise ValueError(f"Unknown anchor positive source: {source!r}")

    positive_scores = [score for _, score in selected]
    if not positive_scores:
        raise ValueError("Cannot compute anchor positive score from an empty positives set.")

    if metric == "median":
        return float(median(positive_scores))
    if metric == "min":
        return float(min(positive_scores))
    raise ValueError(f"Unknown anchor positive metric: {metric!r}")


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


def resolve_checkpoint_dir(
    output_path: Path,
    checkpoint_dir: Path | None,
    resume_checkpoint: Path | None,
) -> Path:
    if checkpoint_dir is not None:
        return checkpoint_dir
    if resume_checkpoint is not None:
        return resume_checkpoint
    return output_path.parent / f"{output_path.name}_checkpoints"


def save_scores_checkpoint(checkpoint_dir: Path, scores: list[float], total_pairs: int) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "scores": scores,
            "total_pairs": total_pairs,
        },
        checkpoint_dir / CHECKPOINT_FILENAME,
    )
    print(f"Saved checkpoint to {checkpoint_dir / CHECKPOINT_FILENAME}", flush=True)


def load_scores_checkpoint(checkpoint_dir: Path) -> tuple[list[float], int]:
    checkpoint_path = checkpoint_dir / CHECKPOINT_FILENAME
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    scores = checkpoint.get("scores")
    if not isinstance(scores, list):
        raise ValueError(f"Checkpoint at {checkpoint_path} does not contain a valid 'scores' list.")

    total_pairs = checkpoint.get("total_pairs")
    if total_pairs is None:
        total_pairs = len(scores)

    return [float(score) for score in scores], int(total_pairs)


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
    initial_scores: list[float] | None = None,
    checkpoint_every: int = 0,
    checkpoint_dir: Path | None = None,
) -> list[float]:
    """Score query-document pairs with a HF reranker model."""
    if not pairs:
        return []

    total = len(pairs)
    scores: list[float] = list(initial_scores or [])
    if len(scores) > total:
        raise ValueError(
            f"Resume checkpoint has {len(scores)} scores but only {total} pairs were built."
        )
    if len(scores) == total:
        if checkpoint_dir is not None:
            save_scores_checkpoint(
                checkpoint_dir=checkpoint_dir,
                scores=scores,
                total_pairs=total,
            )
        return scores

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

    resumed_pairs = len(scores)
    remaining = total - resumed_pairs
    steps_since_last_checkpoint = 0
    with torch.inference_mode():
        for start in tqdm(
            range(resumed_pairs, total, batch_size),
            desc="Scoring pairs",
            total=(remaining + batch_size - 1) // batch_size,
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
            steps_since_last_checkpoint += 1
            if (
                checkpoint_dir is not None
                and checkpoint_every > 0
                and steps_since_last_checkpoint >= checkpoint_every
            ):
                save_scores_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    scores=scores,
                    total_pairs=total,
                )
                steps_since_last_checkpoint = 0

    if checkpoint_dir is not None:
        save_scores_checkpoint(
            checkpoint_dir=checkpoint_dir,
            scores=scores,
            total_pairs=total,
        )

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
            "Drop negatives with score > share * anchor_positive_score for that query."
        ),
    )
    parser.add_argument(
        "--anchor-positive-score",
        type=str,
        default="median_all",
        choices=[
            "median_all",
            "median_selected",
            "min_all",
            "min_selected",
        ],
        help=(
            "How to compute anchor_positive_score for filtering negatives. "
            "'all' uses all positives, while 'selected' uses top --n-positives positives."
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
        "--checkpoint-every",
        type=int,
        default=100,
        help="Save a score checkpoint every N scoring steps (batches). Set 0 to disable periodic saves.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help=(
            "Directory where score checkpoints are written. "
            "Defaults to '<output_dir_name>_checkpoints' next to the output path."
        ),
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint directory to resume score computation from.",
    )
    args = parser.parse_args()

    if not args.input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {args.input_path}")

    if not (0.0 <= args.negative_threshold_share <= 1.0):
        raise ValueError("--negative-threshold-share must be between 0 and 1.")
    if args.checkpoint_every < 0:
        raise ValueError("--checkpoint-every must be >= 0.")
    if args.resume_checkpoint is not None and not args.resume_checkpoint.exists():
        raise FileNotFoundError(f"Resume checkpoint directory does not exist: {args.resume_checkpoint}")


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
    checkpoint_dir = resolve_checkpoint_dir(
        output_path=output_path,
        checkpoint_dir=args.checkpoint_dir,
        resume_checkpoint=args.resume_checkpoint,
    )
    print(f"Checkpoint directory: {checkpoint_dir}")

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

    total_pairs = len(text_pairs)
    resume_scores: list[float] = []
    if args.resume_checkpoint is not None:
        resume_scores, resumed_total_pairs = load_scores_checkpoint(args.resume_checkpoint)
        if resumed_total_pairs != total_pairs:
            raise ValueError(
                "Resume checkpoint was created for a different number of pairs: "
                f"{resumed_total_pairs} != {total_pairs}"
            )
        if len(resume_scores) > total_pairs:
            raise ValueError(
                "Resume checkpoint contains more scores than available pairs: "
                f"{len(resume_scores)} > {total_pairs}"
            )
        print(
            "Loaded resume checkpoint with "
            f"{len(resume_scores)}/{total_pairs} scored pairs."
        )

    if len(resume_scores) == total_pairs:
        print("All pairs already scored in checkpoint. Skipping scoring.")
        scores = resume_scores
    else:
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        scores = score_pairs(
            pairs=text_pairs,
            model_name=args.model_name,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=device,
            trust_remote_code=args.trust_remote_code,
            initial_scores=resume_scores,
            checkpoint_every=args.checkpoint_every,
            checkpoint_dir=checkpoint_dir,
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

    for query_id in train_ds["query_id"]:
        query_id = str(query_id)
        group = scored_by_query.get(query_id)
        if group is None:
            skipped_queries += 1
            continue

        positives = sorted(group["positive"], key=lambda x: x[1], reverse=True)
        hard_negs = group["hard_negative"]
        random_negs = group["random_negative"]

        if not positives:
            skipped_queries += 1
            continue

        anchor_metric, anchor_source = args.anchor_positive_score.split("_", maxsplit=1)
        anchor_positive_score = compute_anchor_positive_score(
            positives=positives,
            metric=anchor_metric,
            source=anchor_source,
            n_selected=args.n_positives,
        )
        threshold = args.negative_threshold_share * anchor_positive_score

        hard_negs = [item for item in hard_negs if item[1] <= threshold]
        random_negs = [item for item in random_negs if item[1] <= threshold]

        selected_positives = positives[: args.n_positives]

        hard_sorted = sorted(hard_negs, key=lambda x: x[1], reverse=True)
        selected_hard = hard_sorted[: args.n_hard_negatives]

        remaining_hard = hard_sorted[args.n_hard_negatives :]
        medium_pool = list(remaining_hard)
        rng.shuffle(medium_pool)
        medium_k = min(args.n_medium_negatives, len(medium_pool))
        selected_medium = medium_pool[:medium_k]
        medium_pool = medium_pool[medium_k:]

        random_pool = list(random_negs)
        rng.shuffle(random_pool)
        random_k = min(args.n_random_negatives, len(random_pool))
        selected_random = random_pool[:random_k]
        random_pool = random_pool[random_k:]

        requested_negatives = (
            args.n_hard_negatives + args.n_medium_negatives + args.n_random_negatives
        )
        selected_negatives = (
            len(selected_hard) + len(selected_medium) + len(selected_random)
        )
        remaining_needed = requested_negatives - selected_negatives

        if remaining_needed > 0:
            extra_medium = min(remaining_needed, len(medium_pool))
            selected_medium.extend(medium_pool[:extra_medium])
            remaining_needed -= extra_medium

        if remaining_needed > 0:
            extra_random = min(remaining_needed, len(random_pool))
            selected_random.extend(random_pool[:extra_random])

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
