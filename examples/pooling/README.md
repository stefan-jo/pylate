# Learn to Pool

This directory contains the code and artifacts for the paper _Learn to Pool: Lightweight Fine-Tuning for Flexible Multi-Vector Compression_.

The project studies whether lightweight pooling-aware fine-tuning can improve ColBERT document compression compared to inference-only pooling. It evaluates three document-side pooling methods, sequential (span), hierarchical, and k-means, across pool factors 2-6 on top of `mxbai-edge-colbert-v0-32m`.

## Summary

- Base model: `mxbai-edge-colbert-v0-32m`
- Pooling methods: sequential (span), hierarchical, and k-means
- Pool factors evaluated: 2-6, corresponding to 50-83% vector reduction
- Pooling is applied only on the document side; queries remain unpooled
- Fine-tuning data: SciFact and FiQA
- Training regimes: fixed pool factor, multi-factor training, and no-pooling fine-tuning as a control
- Evaluation: selected NanoBEIR datasets, followed by BEIR validation
- Metric: NDCG@10

## Main Findings

- Hierarchical pooling is the strongest inference-only pooling method overall.
- K-means is the strongest training method overall, producing the most consistent gains over untrained baselines.
- Fine-tuning without pooling can severely degrade pooled performance.
- Multi-factor fine-tuning produces a single model that works well across different compression levels.
- Pooling-aware fine-tuning on one dataset with k-means generally improves pooled retrieval on other datasets, with only moderate impact on unpooled performance.

## Models And Data

Only the best-performing models were uploaded to Hugging Face. The uploaded datasets can be used together with `examples/pooling/train/train_beir_colbert_distillation.py` to reproduce the full set of fine-tuned models.

### Models

- [SciFact k-means PF1-6](https://huggingface.co/stefan-jo/mxbai-edge-colbert-v0-32m-scifact-kmeans-pf1-6)
- [FiQA k-means PF1-6](https://huggingface.co/stefan-jo/mxbai-edge-colbert-v0-32m-fiqa-kmeans-pf1-6)

### Datasets

- [SciFact training data with mined reranker scores](https://huggingface.co/datasets/stefan-jo/scifact-train-mined-reranker-scores)
- [FiQA training data with mined reranker scores](https://huggingface.co/datasets/stefan-jo/fiqa-train-mined-reranker-scores)

## Paper

- [PDF](https://stefan-jo.github.io/learn-to-pool/downloads/paper.pdf)

## Repository Contents

- `train/`: training scripts, data preparation code, and supporting notebooks
- `evaluation/`: evaluation scripts, sweep outputs, and analysis notebooks

### Data Preparation

- `train/data/mine_beir_training_data.py`: builds BEIR training tuples with positives, hard negatives, and random negatives
- `train/data/score_beir_training_data.py`: reranks mined tuples and produces the final scored training splits used for distillation

### Training

- `train/train_beir_colbert_distillation.py`: main script for pooling-aware ColBERT distillation on a scored BEIR training split

### Evaluation

- `evaluation/run_nanobeir_sweep.py`: runs NanoBEIR pooling sweeps and aggregates retrieval metrics
- `evaluation/run_beir_sweep.py`: runs BEIR pooling sweeps and writes summary tables, plots, and JSON outputs

## Results

These CSV files contain the main aggregate evaluation outputs used for model comparison.

- `evaluation/results/combined_model_metrics.csv`: aggregated NanoBEIR model metrics
- `evaluation/results/beir_combined_model_metrics.csv`: aggregated BEIR model metrics

## Citation

If you use this project, please cite the paper:

```bibtex
@inproceedings{
josef2026learn,
title={Learn to Pool: Lightweight Fine-Tuning for Flexible Multi-Vector Compression},
author={Stefan Josef},
booktitle={The First Late Interaction Workshop (LIR) @ ECIR 2026},
year={2026},
url={https://openreview.net/forum?id=nw2px6ZDC2}
}
```
