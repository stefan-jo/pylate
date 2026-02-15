#!/usr/bin/env python3
"""Train ColBERT with distillation on a scored BEIR training split."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

from pylate import evaluation, losses, models, utils

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

DEFAULT_TRAIN_PATH = Path("examples/pooling/train/data/outputs/scifact_scored")


def normalize_cli_list(values: list[str] | None, lowercase: bool = True) -> list[str] | None:
    if not values:
        return None

    items: list[str] = []
    for value in values:
        for chunk in value.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            items.append(chunk.lower() if lowercase else chunk)

    return items or None


def infer_dataset_name(input_path: Path) -> str | None:
    """Infer a BEIR dataset name from a scored input folder path."""
    stem = input_path.name.removesuffix("_scored")
    if stem in ALL_BEIR_DATASETS:
        return stem

    if input_path.parent.name == "cqadupstack":
        candidate = f"cqadupstack/{stem}"
        if candidate in ALL_BEIR_DATASETS:
            return candidate

    return None


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


def load_train_split(path: Path) -> Dataset:
    dataset = load_from_disk(str(path))
    if isinstance(dataset, DatasetDict):
        if "train" not in dataset:
            raise ValueError("Input DatasetDict must contain a 'train' split.")
        return dataset["train"]
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune an existing ColBERT checkpoint with distillation on a scored "
            "BEIR training split."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=DEFAULT_TRAIN_PATH,
        help="Path to the scored dataset saved with datasets.save_to_disk.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help=(
            "BEIR dataset name for loading train query/document texts. If omitted, "
            "inferred from --train-path by stripping '_scored'."
        ),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="mixedbread-ai/mxbai-edge-colbert-v0-32m",
        help="ColBERT checkpoint to continue fine-tuning.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/pooling/train/output"),
        help="Base output directory for checkpoints and final model.",
    )
    parser.add_argument("--fixed-n-ways", type=int, default=16)
    parser.add_argument("--num-train-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to keep. "
            "Use None (default) to keep all epoch checkpoints."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help=(
            "If > 0, stop after this many update steps (useful for quick smoke tests). "
            "If -1, train for full epochs."
        ),
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=1.0,
        help=(
            "Fraction of filtered training rows to keep for debugging. "
            "Applied after filtering/capping and before training transform. "
            "Must be in (0, 1]."
        ),
    )
    parser.add_argument("--query-length", type=int, default=32)
    parser.add_argument("--document-length", type=int, default=180)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable fp16 mixed precision.",
    )
    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable bf16 mixed precision.",
    )
    parser.add_argument(
        "--normalize-scores",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply per-query min/max normalization inside distillation loss.",
    )
    parser.add_argument(
        "--eval-every-epoch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run NanoBEIR evaluation after every epoch.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=16,
        help="Batch size for optional NanoBEIR evaluator.",
    )
    parser.add_argument(
        "--eval-datasets",
        nargs="+",
        default=None,
        help=(
            "Optional NanoBEIR subset for evaluation (space/comma separated). "
            f"Available: {', '.join(ALL_NANOBEIR_DATASETS)}"
        ),
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Override run name. If omitted, a name is built automatically.",
    )
    parser.add_argument(
        "--report-to-wandb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Report to Weights & Biases.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help=(
            "W&B project name. Overrides WANDB_PROJECT/WANDB_PROEJCT when provided."
        ),
    )
    args = parser.parse_args()

    if args.report_to_wandb:
        wandb_project = (
            args.wandb_project
            or os.environ.get("WANDB_PROJECT")
            or os.environ.get("WANDB_PROEJCT")
        )
        missing_wandb_env_vars: list[str] = []
        if not os.environ.get("WANDB_API_KEY"):
            missing_wandb_env_vars.append("WANDB_API_KEY")
        if not wandb_project:
            missing_wandb_env_vars.append("WANDB_PROJECT (or --wandb-project)")

        if missing_wandb_env_vars:
            raise ValueError(
                "--report-to-wandb requires: "
                f"{missing_wandb_env_vars}. "
                "Set WANDB_API_KEY and WANDB_PROJECT, or pass --wandb-project."
            )

        # Normalize to WANDB_PROJECT for wandb/HF integrations downstream.
        os.environ["WANDB_PROJECT"] = wandb_project

    if args.fixed_n_ways <= 0:
        raise ValueError("--fixed-n-ways must be > 0.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0.")
    if args.num_train_epochs <= 0:
        raise ValueError("--num-train-epochs must be > 0.")
    if args.max_steps == 0 or args.max_steps < -1:
        raise ValueError("--max-steps must be -1 or > 0.")
    if args.save_total_limit is not None and args.save_total_limit <= 0:
        raise ValueError("--save-total-limit must be > 0 when provided.")
    if not 0.0 <= args.warmup_ratio <= 1.0:
        raise ValueError("--warmup-ratio must be in [0, 1].")
    if not 0.0 < args.train_fraction <= 1.0:
        raise ValueError("--train-fraction must be in (0, 1].")
    if args.fp16 and args.bf16:
        raise ValueError("Choose only one precision mode: fp16 or bf16.")
    if not args.train_path.exists():
        raise FileNotFoundError(f"Train path does not exist: {args.train_path}")

    dataset_name = args.dataset_name
    if dataset_name is None:
        inferred_dataset = infer_dataset_name(args.train_path)
        if inferred_dataset is None:
            raise ValueError(
                "Could not infer dataset name from --train-path. "
                "Please provide --dataset-name explicitly."
            )
        dataset_name = inferred_dataset
    dataset_name = dataset_name.strip().lower()
    if dataset_name not in ALL_BEIR_DATASETS:
        raise ValueError(
            f"Unknown dataset: {dataset_name!r}. Available: {ALL_BEIR_DATASETS}"
        )

    eval_datasets = normalize_cli_list(args.eval_datasets, lowercase=True)
    if eval_datasets is not None:
        invalid_datasets = sorted(set(eval_datasets) - set(ALL_NANOBEIR_DATASETS))
        if invalid_datasets:
            raise ValueError(
                f"Unknown NanoBEIR datasets: {invalid_datasets}. "
                f"Available: {ALL_NANOBEIR_DATASETS}"
            )

    print(f"Loading scored training data from: {args.train_path}")
    train_dataset = load_train_split(args.train_path)
    print(f"Loaded rows: {len(train_dataset)}")
    print(f"Columns: {train_dataset.column_names}")

    required_columns = {"query_id", "document_ids", "scores", "labels"}
    missing_columns = sorted(required_columns - set(train_dataset.column_names))
    if missing_columns:
        raise ValueError(
            "Input dataset is missing required columns for filtering/capping: "
            f"{missing_columns}"
        )

    initial_rows = len(train_dataset)
    dropped_inconsistent_lengths = 0
    dropped_without_positive = 0
    dropped_too_short = 0
    capped_rows = 0
    capped_removed_documents = 0

    processed_rows = {
        "query_id": [],
        "document_ids": [],
        "scores": [],
    }

    for row in train_dataset:
        query_id = str(row["query_id"])
        document_ids = row["document_ids"]
        scores = row["scores"]
        labels = row["labels"]

        if not (
            len(document_ids) == len(scores)
            and len(document_ids) == len(labels)
        ):
            dropped_inconsistent_lengths += 1
            continue

        if len(document_ids) > args.fixed_n_ways:
            capped_rows += 1
            capped_removed_documents += len(document_ids) - args.fixed_n_ways

        # Docs are already score-ranked in the scored dataset: keep top-k first.
        truncated_document_ids = document_ids[: args.fixed_n_ways]
        truncated_scores = scores[: args.fixed_n_ways]
        truncated_labels = labels[: args.fixed_n_ways]

        if len(truncated_document_ids) < args.fixed_n_ways:
            dropped_too_short += 1
            continue

        if "positive" not in truncated_labels:
            dropped_without_positive += 1
            continue

        processed_rows["query_id"].append(query_id)
        processed_rows["document_ids"].append(
            [str(doc_id) for doc_id in truncated_document_ids]
        )
        processed_rows["scores"].append(
            [float(score) for score in truncated_scores]
        )

    train_dataset = Dataset.from_dict(processed_rows)
    final_rows = len(train_dataset)

    print("Filtering/capping summary:")
    print(f"  initial rows: {initial_rows}")
    print(f"  dropped (inconsistent lengths): {dropped_inconsistent_lengths}")
    print(f"  dropped (no positive label in top-{args.fixed_n_ways}): {dropped_without_positive}")
    print(f"  dropped (< {args.fixed_n_ways} docs): {dropped_too_short}")
    print(
        "  capped (> {fixed} docs): {capped_rows} rows, {removed_docs} docs removed".format(
            fixed=args.fixed_n_ways,
            capped_rows=capped_rows,
            removed_docs=capped_removed_documents,
        )
    )
    print(f"  final rows: {final_rows}")

    if final_rows == 0:
        raise ValueError("No training rows left after filtering/capping.")

    if args.train_fraction < 1.0:
        sampled_rows = max(1, int(final_rows * args.train_fraction))
        if sampled_rows < final_rows:
            train_dataset = train_dataset.shuffle(seed=args.seed).select(range(sampled_rows))
            print(
                "Applied train fraction: "
                f"{sampled_rows}/{final_rows} rows ({args.train_fraction:.2%})"
            )
            final_rows = sampled_rows

    print(f"Loading BEIR train split '{dataset_name}' to resolve texts...")
    corpus, queries, _ = load_beir_train_split(dataset_name=dataset_name)
    print(f"Loaded BEIR train split: queries={len(queries)}, corpus={len(corpus)}")

    queries_dataset = Dataset.from_dict(
        {
            "query_id": [str(query_id) for query_id in queries.keys()],
            "text": list(queries.values()),
        }
    )
    documents_dataset = Dataset.from_dict(
        {
            "document_id": [str(document["id"]) for document in corpus],
            "text": [document["text"] for document in corpus],
        }
    )

    train_dataset.set_transform(
        utils.KDProcessing(
            queries=queries_dataset,
            documents=documents_dataset,
            n_ways=args.fixed_n_ways,
        ).transform
    )

    run_name = (
        args.run_name
        or f"mxbai-colbert-kd-{dataset_name}-{args.learning_rate}-lr-{args.num_train_epochs}-epochs"
    )
    output_dir = args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model_name}")
    model = models.ColBERT(
        model_name_or_path=args.model_name,
        query_length=args.query_length,
        document_length=args.document_length,
        truncation=True,
        device=args.device,
    )

    evaluator = None
    eval_strategy = "no"
    load_best_model_at_end = False
    metric_for_best_model = None
    greater_is_better = None
    if args.eval_every_epoch:
        evaluator_kwargs = {
            "batch_size": args.eval_batch_size,
            "show_progress_bar": True,
        }
        if eval_datasets is not None:
            evaluator_kwargs["dataset_names"] = eval_datasets

        evaluator = evaluation.NanoBEIREvaluator(**evaluator_kwargs)
        eval_strategy = "epoch"
        load_best_model_at_end = True
        metric_for_best_model = "eval_NanoBEIR_mean_MaxSim_ndcg@10"
        greater_is_better = True
        print("NanoBEIR evaluation is enabled and will run after each epoch.")
        print(
            "Best-checkpoint selection enabled on metric: "
            f"{metric_for_best_model} (higher is better)."
        )
    else:
        print("NanoBEIR evaluation is disabled.")

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        eval_strategy=eval_strategy,
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        fp16=args.fp16,
        bf16=args.bf16,
        run_name=run_name,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
        max_steps=args.max_steps,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        report_to=["wandb"] if args.report_to_wandb else [],
    )

    train_loss = losses.Distillation(
        model=model,
        normalize_scores=args.normalize_scores,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=train_loss,
        evaluator=evaluator,
        data_collator=utils.ColBERTCollator(
            tokenize_fn=model.tokenize,
            valid_label_columns=["scores"],
        ),
    )

    print("Starting training...")
    trainer.train()

    final_output_dir = output_dir / "final"
    model.save_pretrained(str(final_output_dir))
    if args.eval_every_epoch:
        print(
            "Saved final model to: {path} (best checkpoint loaded at end by {metric}).".format(
                path=final_output_dir,
                metric=metric_for_best_model,
            )
        )
    else:
        print(f"Saved final model to: {final_output_dir} (last training state).")


if __name__ == "__main__":
    main()
