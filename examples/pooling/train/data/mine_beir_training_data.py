#!/usr/bin/env python3
"""Build BEIR training tuples with positives, hard negatives, and random negatives."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict
from sentence_transformers import SentenceTransformer, util

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


def load_beir_train_split(dataset_name: str) -> tuple[list[dict[str, str]], dict[str, str], dict]:
    """Load one BEIR dataset train split, including cqadupstack subsets."""
    if dataset_name.startswith("cqadupstack/"):
        from beir import util as beir_util

        beir_util.download_and_unzip(
            url="https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/cqadupstack.zip",
            out_dir="./evaluation_datasets/",
        )
        return evaluation.load_custom_dataset(
            f"evaluation_datasets/{dataset_name}",
            split="train",
        )

    return evaluation.load_beir(dataset_name=dataset_name, split="train")


def to_document_text(document: dict[str, str]) -> str:
    return f"{document.get('title', '')} {document.get('text', '')}".strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create BEIR train examples with positive, hard negative, and random negative ids.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help=f"BEIR dataset name. Available: {', '.join(ALL_BEIR_DATASETS)}",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="BAAI/bge-small-en-v1.5",
        help="SentenceTransformers model for hard negative mining.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/pooling/train/data/outputs/"),
        help="Output directory for Hugging Face DatasetDict saved with a single 'train' split.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Embedding batch size.",
    )
    parser.add_argument(
        "--hard-negatives-k",
        type=int,
        default=100,
        help="Number of hard negatives per query after filtering positives.",
    )
    parser.add_argument(
        "--random-negatives-k",
        type=int,
        default=50,
        help="Number of random negatives per query.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for random negatives.",
    )
    parser.add_argument(
        "--sample-queries",
        type=int,
        default=None,
        help=(
            "Optional number of queries (with positives) to keep. "
            "If not set, keep all queries."
        ),
    )
    parser.add_argument(
        "--sample-docs",
        type=int,
        default=None,
        help=(
            "Optional mining corpus size for debugging. "
            "If set, randomly sample this many docs for mining."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for sentence-transformers model (e.g., cuda, cpu).",
    )
    args = parser.parse_args()

    dataset_name = args.dataset_name.strip().lower()
    if dataset_name not in ALL_BEIR_DATASETS:
        raise ValueError(
            f"Unknown dataset: {dataset_name!r}. Available: {ALL_BEIR_DATASETS}"
        )
    if args.sample_queries is not None and args.sample_queries <= 0:
        raise ValueError("--sample-queries must be > 0.")
    if args.sample_docs is not None and args.sample_docs <= 0:
        raise ValueError("--sample-docs must be > 0.")

    print(f"Loading BEIR dataset '{dataset_name}' (split=train)...")
    try:
        corpus, queries, qrels = load_beir_train_split(dataset_name)
    except Exception as exc:
        print(
            f"Could not load train split for dataset '{dataset_name}'. "
            f"Exiting. Error: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    print(
        "Loaded sizes: "
        f"queries={len(queries)}, corpus={len(corpus)}, qrels={len(qrels)}"
    )

    corpus_by_id = {doc["id"]: doc for doc in corpus}

    print("Collecting positives from qrels/queries/corpus...")
    positives_by_query: dict[str, set[str]] = {}
    for query_id, rel_docs in qrels.items():
        if query_id not in queries:
            continue
        pos_ids = {
            doc_id
            for doc_id, rel in rel_docs.items()
            if rel > 0 and doc_id in corpus_by_id
        }
        if pos_ids:
            positives_by_query[query_id] = pos_ids

    if not positives_by_query:
        print("No positive (query, doc) pairs found after filtering. Exiting.")
        raise SystemExit(1)

    print(
        "Positive pairs summary: "
        f"queries_with_positives={len(positives_by_query)}, "
        f"total_positive_pairs={sum(len(v) for v in positives_by_query.values())}"
    )

    rng = random.Random(args.seed)
    initial_query_count = len(positives_by_query)
    initial_doc_count = len(corpus_by_id)

    if args.sample_queries is not None and args.sample_queries < len(positives_by_query):
        sampled_query_ids = set(
            rng.sample(list(positives_by_query.keys()), k=args.sample_queries)
        )
        positives_by_query = {
            query_id: positives_by_query[query_id]
            for query_id in sampled_query_ids
        }
        print(
            "Applied query sampling: "
            f"queries_with_positives={len(positives_by_query)} (from {initial_query_count})"
        )

    if args.sample_docs is not None:
        sampled_doc_count = min(args.sample_docs, len(corpus))
        sampled_doc_ids = set(rng.sample(list(corpus_by_id.keys()), k=sampled_doc_count))
        corpus = [doc for doc in corpus if doc["id"] in sampled_doc_ids]
        corpus_by_id = {doc["id"]: doc for doc in corpus}

        print(
            "Applied doc sampling: "
            f"corpus={len(corpus)} (from {initial_doc_count})"
        )

    query_ids = list(positives_by_query.keys())
    query_texts = [queries[qid] for qid in query_ids]

    corpus_ids = [doc["id"] for doc in corpus]
    corpus_texts = [to_document_text(doc) for doc in corpus]

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading sentence-transformers model '{args.model_name}' on device={device}...")
    model = SentenceTransformer(args.model_name, device=device)

    print("Encoding corpus...")
    corpus_embeddings = model.encode_document(
        corpus_texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    print("Encoding queries...")
    query_embeddings = model.encode_query(
        query_texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    print("Mining hard negatives...")
    max_extra = max(len(positives_by_query[qid]) for qid in query_ids)
    overfetch = min(len(corpus_ids), args.hard_negatives_k + max_extra + 50)
    retrieved = util.semantic_search(
        query_embeddings,
        corpus_embeddings,
        top_k=overfetch,
    )

    hard_negatives_by_query: dict[str, list[str]] = {}
    for query_id, hits in zip(query_ids, retrieved):
        positives = positives_by_query[query_id]
        hard_negative_ids: list[str] = []
        for hit in hits:
            doc_id = corpus_ids[hit["corpus_id"]]
            if doc_id in positives:
                continue
            hard_negative_ids.append(doc_id)
            if len(hard_negative_ids) == args.hard_negatives_k:
                break
        hard_negatives_by_query[query_id] = hard_negative_ids

    print("Sampling random negatives...")
    corpus_id_set = set(corpus_ids)
    random_negatives_by_query: dict[str, list[str]] = {}

    for query_id in query_ids:
        forbidden = positives_by_query[query_id].union(hard_negatives_by_query[query_id])
        candidates = list(corpus_id_set - forbidden)
        sample_size = min(args.random_negatives_k, len(candidates))
        random_negatives_by_query[query_id] = rng.sample(candidates, k=sample_size)

    dataset = Dataset.from_dict(
        {
            "query_id": query_ids,
            "positive_ids": [sorted(positives_by_query[qid]) for qid in query_ids],
            "hard_negative_ids": [hard_negatives_by_query[qid] for qid in query_ids],
            "random_negative_ids": [random_negatives_by_query[qid] for qid in query_ids],
        }
    )
    dataset_dict = DatasetDict({"train": dataset})

    output_dir = args.output_dir / f"{dataset_name}_mined"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving mined data to {output_dir} as DatasetDict(train)...")
    dataset_dict.save_to_disk(str(output_dir))

    print(
        "Done. "
        f"Saved split='train' with {len(query_ids)} rows, up to "
        f"{args.hard_negatives_k} hard and {args.random_negatives_k} random negatives each."
    )


if __name__ == "__main__":
    main()
