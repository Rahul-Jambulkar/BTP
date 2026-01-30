# BTP
CLEF 2025 Subjectivity Detection Task — Transformer-based monolingual, multilingual, and zero-shot models for binary subjectivity classification (SUBJ vs. OBJ) across multiple languages.
# CLEF 2025 Subjectivity Detection

This repository contains the official training and inference code used for the **CLEF 2025 Subjectivity Detection Task**. The system performs **binary subjectivity classification** (SUBJ vs. OBJ) across multiple languages using Transformer-based models from the Hugging Face ecosystem.

The code supports **monolingual**, **multilingual**, and **zero-shot cross-lingual** settings, closely following the experimental setup described in the CLEF 2025 shared task.

---

## Overview

The pipeline is implemented in a single runnable script, `subjectivity_runner.py`, and provides:

* End-to-end data loading, training, evaluation, and prediction
* Automatic language-specific model selection
* Multilingual training with seen languages
* Zero-shot transfer to unseen languages
* Robust training with GPU out-of-memory recovery
* Reproducible experiments via fixed random seeds

The task is framed as **binary sequence classification**:

* `OBJ` (Objective) → label `0`
* `SUBJ` (Subjective) → label `1`

---

## Supported Languages

### Monolingual Training

* English
* Arabic
* German
* Italian
* Bulgarian

### Zero-Shot Only Languages

* Greek
* Polish
* Romanian
* Ukrainian

### Multilingual Training

* Trained jointly on all *seen* languages listed above

---

## Models

The system automatically selects pretrained models depending on the language:

* **Monolingual models** (e.g., RoBERTa, MARBERTv2, GELECTRA)
* **Multilingual models** (e.g., XLM-R, mDeBERTa, InfoXLM)
* Fallback to `microsoft/mdeberta-v3-base` if a model is unavailable

Learning rates are automatically adjusted per model architecture.

---

## Data Format

All datasets are expected to be in **TSV format** with automatic column detection.

### Required files per language

```
train_<lang>.tsv
(dev_<lang>.tsv)
test_<lang>_unlabeled.tsv
(test_<lang>_labeled.tsv)
```

Where `<lang>` is the CLEF language code (e.g., `en`, `ar`, `de`).

### Columns

The script automatically detects:

* Text column (e.g., `text`, `sentence`, `content`)
* Label column (`OBJ` / `SUBJ`, or `0` / `1`)
* Optional ID column

---

## Training Strategy

The system uses a **two-phase fine-tuning strategy**:

1. **Phase 1**: Train on training data and validate on development data
2. **Phase 2**: Continue training on the combined train + dev set

Additional features:

* Class-weighted cross-entropy loss
* Gradient accumulation
* Gradient checkpointing
* Automatic batch size reduction on GPU OOM

---

## Usage

### Monolingual Training

```bash
python subjectivity_runner.py \
  --lang english \
  --data_root data \
  --output_root outputs
```

### Multilingual Training

```bash
python subjectivity_runner.py \
  --lang multilingual \
  --base_model microsoft/infoxlm-large
```

### Zero-Shot Evaluation

```bash
python subjectivity_runner.py \
  --lang greek
```

---

## Outputs

For each run, the following files are generated:

```
outputs/<language>/<model>/
├── final_model/
├── predictions.tsv
├── metrics.txt
└── completed.flag
```

* `predictions.tsv`: Model predictions (`SUBJ` / `OBJ`)
* `metrics.txt`: Macro-F1, accuracy, precision, recall, and confusion matrix

---

## Requirements

* Python ≥ 3.9
* PyTorch
* Transformers
* Datasets
* scikit-learn
* pandas
* numpy

---

## Reproducibility

* Fixed random seed (`SEED = 42`)
* Deterministic training settings where possible
* Cached models and datasets

---

## License

This repository is released for research and evaluation purposes in the context of the **CLEF 2025 Subjectivity Detection Task**.

---

## Citation

If you use this code, please cite the CLEF 2025 Subjectivity Detection shared task and the corresponding system description paper.
