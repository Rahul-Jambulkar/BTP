import os
import sys
import re
import json
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_recall_fscore_support, confusion_matrix

import torch
from torch.nn import CrossEntropyLoss
from datasets import Dataset, concatenate_datasets

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Setup environment
os.environ['TRANSFORMERS_CACHE'] = str(Path.cwd() / 'model_cache')
os.environ['HF_DATASETS_CACHE'] = str(Path.cwd() / 'dataset_cache')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------- Constants ----------------
SEED = 42
DEFAULT_MAX_LENGTH = 128

PAPER_MONO_MODELS = {
    "english": "roberta-large",
    "arabic": "UBC-NLP/MARBERTv2",
    "german": "deepset/gelectra-large",
    "italian": "xlm-roberta-large",
    "bulgarian": "xlm-roberta-large",
}

PAPER_ZERO_SHOT_BASE = {
    "greek": "microsoft/mdeberta-v3-base",
    "polish": "microsoft/mdeberta-v3-base",
    "ukrainian": "xlm-roberta-large",
    "romanian": "xlm-roberta-large",
}

FALLBACK_MODEL = "microsoft/mdeberta-v3-base"

LANG_CODES = {
    "arabic": "ar",
    "bulgarian": "bg",
    "english": "en",
    "german": "de",
    "greek": "gr",
    "italian": "it",
    "polish": "pol",
    "romanian": "ro",
    "ukrainian": "ukr",
    "multilingual": "multilingual",
}

SEEN_LANGS_FOR_MULTI = ["arabic", "bulgarian", "english", "german", "italian"]
ZERO_SHOT_ONLY_LANGS = ["greek", "polish", "romanian", "ukrainian"]

LR_BY_MODEL = {
    "roberta-large": 2.0e-5,
    "xlm-roberta-large": 1.5e-5,
    "microsoft/mdeberta-v3-base": 1.8e-5,
    "UBC-NLP/MARBERTv2": 1.8e-5,
    "deepset/gelectra-large": 1.5e-5,
    "microsoft/infoxlm-large": 1.5e-5,
    "_default": 1.8e-5,
}

def sanitize_model_name(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9\-_\.]", "_", model_name)

def select_lr(model_name: str) -> float:
    for k, v in LR_BY_MODEL.items():
        if k != "_default" and model_name.lower() == k.lower():
            return v
    return LR_BY_MODEL["_default"]

def ensure_model(model_name: str) -> str:
    try:
        _ = AutoConfig.from_pretrained(model_name)
        return model_name
    except Exception as e:
        logger.warning(f"Model {model_name} not available, falling back to {FALLBACK_MODEL}: {e}")
        return FALLBACK_MODEL

def make_output_dirs(base_out: Path, lang: str, model_name: str):
    model_tag = sanitize_model_name(model_name)
    out_dir = base_out / lang / model_tag
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, ckpt_dir

def is_completed(out_dir: Path) -> bool:
    return (out_dir / "completed.flag").exists()

def detect_columns(df: pd.DataFrame, is_labeled: bool):
    cols = list(df.columns)
    text_col = None
    label_col = None
    id_col = None
    for c in cols:
        lc = c.lower()
        if text_col is None and lc in ["text", "sentence", "content", "utterance"]:
            text_col = c
        if is_labeled and label_col is None and lc in ["label", "labels", "gold", "y", "target", "class"]:
            label_col = c
        if id_col is None and lc in ["id", "idx", "index"]:
            id_col = c
    if text_col is None:
        str_cols = [c for c in cols if df[c].dtype == object]
        if not str_cols:
            raise ValueError("No text-like column found")
        best_c, best_len = None, -1
        for c in str_cols:
            try:
                mean_len = df[c].astype(str).str.len().mean()
            except Exception:
                mean_len = 0
            if mean_len > best_len:
                best_len, best_c = mean_len, c
        text_col = best_c
    if is_labeled and label_col is None:
        for c in cols:
            vals = set(str(v).strip().upper() for v in df[c].dropna().unique().tolist()[:200])
            if vals.issubset({"OBJ","SUBJ","0","1"}):
                label_col = c
                break
    return id_col, text_col, label_col

def map_labels(series):
    def f(x):
        if pd.isna(x):
            return None
        s = str(x).strip().upper()
        if s in ["OBJ","0"]:
            return 0
        if s in ["SUBJ","1"]:
            return 1
        try:
            v = int(float(s))
            return 1 if v != 0 else 0
        except:
            return None
    return series.map(f)

def load_tsv(path: Path, is_labeled: bool):
    df = pd.read_csv(path, sep="\t", quoting=3, dtype=str, keep_default_na=False)
    df = df.replace({"": np.nan})
    id_col, text_col, label_col = detect_columns(df, is_labeled)
    if text_col is None:
        raise ValueError(f"No text column in {path}")
    text = df[text_col].astype(str).fillna("")
    ids = df[id_col] if id_col else pd.Series(range(len(df)), name="id")
    labels = None
    if is_labeled:
        if label_col is None:
            raise ValueError(f"No label column in {path}")
        labels = map_labels(df[label_col])
        keep = ~labels.isna() & text.notna()
        text = text[keep]
        ids = ids[keep]
        labels = labels[keep]
    out = {"id": ids, "text": text}
    if is_labeled:
        out["label"] = labels
    return pd.DataFrame(out)

def df_to_ds(df: pd.DataFrame, tokenizer, max_length: int):
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)
    features = {"id": df["id"].astype(str).tolist(), "text": df["text"].astype(str).tolist()}
    if "label" in df.columns and df["label"] is not None and not df["label"].isna().all():
        features["label"] = df["label"].astype(int).tolist()
    ds = Dataset.from_dict(features)
    return ds.map(tokenize, batched=True, remove_columns=["text"])

def evaluate_predictions(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4, zero_division=0)
    return {"f1_macro": f1, "accuracy": acc, "precision_macro": precision, "recall_macro": recall, "report": report}

def write_metrics(path: Path, metrics: dict, labels_true: list, labels_pred: list):
    with open(path, "w", encoding="utf-8") as f:
        f.write("Macro F1: {:.4f}\n".format(metrics.get("f1_macro", float("nan"))))
        f.write("Accuracy: {:.4f}\n".format(metrics.get("accuracy", float("nan"))))
        f.write("Precision (macro): {:.4f}\n".format(metrics.get("precision_macro", float("nan"))))
        f.write("Recall (macro): {:.4f}\n".format(metrics.get("recall_macro", float("nan"))))
        f.write("\nClassification report:\n")
        f.write(metrics.get("report", "") + "\n")
        cm = confusion_matrix(labels_true, labels_pred, labels=[0,1])
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm) + "\n")

class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # FIXED: Added num_items_in_batch parameter for new API
        labels = inputs.get("labels")
        outputs = model(**{k: v for k,v in inputs.items() if k != "labels"})
        logits = outputs.get("logits")
        if labels is not None:
            if self.class_weights is not None:
                loss_fct = CrossEntropyLoss(weight=self.class_weights.to(logits.device))
            else:
                loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            loss = None
        return (loss, outputs) if return_outputs else loss

def compute_class_weights(labels: np.ndarray):
    counts = np.bincount(labels, minlength=2)
    if counts.sum() == 0 or (counts == 0).any():
        return None
    inv = counts.sum() / (2.0 * counts.astype(np.float32))
    return torch.tensor(inv, dtype=torch.float32)

def train_with_oom_retry(trainer_factory):
    bs_candidates = [8, 4, 2, 1]
    for bs in bs_candidates:
        try:
            trainer1, args1 = trainer_factory(phase="phase1", per_device_bs=bs)
            if trainer1.train_dataset is not None and len(trainer1.train_dataset) > 0:
                resume_ckpt = args1.output_dir if (os.path.isdir(args1.output_dir) and any(os.scandir(args1.output_dir))) else None
                trainer1.train(resume_from_checkpoint=resume_ckpt)
            model = trainer1.model
            trainer2, args2 = trainer_factory(phase="phase2", per_device_bs=bs, model=model)
            if trainer2 is not None and trainer2.train_dataset is not None and len(trainer2.train_dataset) > 0:
                resume_ckpt = args2.output_dir if (os.path.isdir(args2.output_dir) and any(os.scandir(args2.output_dir))) else None
                trainer2.train(resume_from_checkpoint=resume_ckpt)
                model = trainer2.model
            return model
        except torch.cuda.OutOfMemoryError:
            logger.warning(f"OOM with BS={bs}, trying smaller")
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"OOM with BS={bs}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                raise
    logger.warning("GPU OOM, falling back to CPU")
    trainer1, _ = trainer_factory(phase="phase1", per_device_bs=1, force_cpu=True)
    if trainer1.train_dataset is not None and len(trainer1.train_dataset) > 0:
        trainer1.train()
    model = trainer1.model
    trainer2, _ = trainer_factory(phase="phase2", per_device_bs=1, force_cpu=True, model=model)
    if trainer2 is not None and trainer2.train_dataset is not None and len(trainer2.train_dataset) > 0:
        trainer2.train()
        model = trainer2.model
    return model

def two_phase_train(tokenizer, model_name, model_config, ds_train, ds_dev, out_dir: Path, ckpt_dir: Path, base_lr: float, max_length: int):
    set_seed(SEED)
    
    def class_weights_for(ds):
        if ds is None or ds.num_rows == 0 or "label" not in ds.features:
            return None
        return compute_class_weights(np.array(ds["label"]))

    cw = class_weights_for(ds_train)

    def trainer_factory(phase, per_device_bs=4, force_cpu=False, model=None):
        if model is None:
            m = AutoModelForSequenceClassification.from_pretrained(model_name, config=model_config)
        else:
            m = model
        try:
            m.gradient_checkpointing_enable()
        except:
            pass
        m.config.use_cache = False

        if phase == "phase1":
            train_ds = ds_train
            eval_ds = ds_dev if (ds_dev is not None and ds_dev.num_rows > 0) else None
            out_dir_phase = ckpt_dir / "phase1"
            lr = base_lr
            epochs = 5
            eval_strat = "epoch" if eval_ds is not None else "no"
            load_best = True if eval_ds is not None else False
        else:
            if ds_dev is not None and ds_dev.num_rows > 0 and ds_train is not None and ds_train.num_rows > 0:
                train_ds = concatenate_datasets([ds_train, ds_dev])
            else:
                train_ds = ds_train if ds_train is not None else ds_dev
            eval_ds = None
            out_dir_phase = ckpt_dir / "phase2"
            lr = base_lr * 0.5
            epochs = 3
            eval_strat = "no"
            load_best = False

        ga_steps = max(1, 16 // per_device_bs)

        args = TrainingArguments(
            output_dir=str(out_dir_phase),
            overwrite_output_dir=False,
            learning_rate=lr,
            per_device_train_batch_size=per_device_bs,
            per_device_eval_batch_size=max(1, min(8, per_device_bs)),
            gradient_accumulation_steps=ga_steps,
            num_train_epochs=epochs,
            weight_decay=0.01,
            eval_strategy=eval_strat,
            save_strategy="epoch",
            save_total_limit=1,
            logging_steps=50,
            load_best_model_at_end=load_best,
            metric_for_best_model="eval_f1_macro" if load_best else None,
            greater_is_better=True,
            warmup_ratio=0.1,
            lr_scheduler_type="linear",
            seed=SEED,
            fp16=(torch.cuda.is_available() and not force_cpu),
            dataloader_num_workers=0,
            eval_accumulation_steps=8,
            report_to="none",
            no_cuda=force_cpu,
        )

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            mets = evaluate_predictions(labels, preds)
            return {"eval_f1_macro": mets["f1_macro"], "eval_accuracy": mets["accuracy"]}

        trainer = WeightedTrainer(
            class_weights=cw,
            model=m,
            args=args,
            train_dataset=train_ds if (train_ds is not None and train_ds.num_rows > 0) else None,
            eval_dataset=eval_ds,
            processing_class=tokenizer,  # Use processing_class instead of tokenizer
            compute_metrics=compute_metrics if eval_ds is not None else None,
        )
        return trainer, args

    model = train_with_oom_retry(trainer_factory)
    
    final_dir = out_dir / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    return model

def predict_and_save(model, tokenizer, ds_test, df_test_unlab, df_test_lab, out_dir: Path):
    trainer = Trainer(model=model, processing_class=tokenizer)
    preds = trainer.predict(ds_test)
    pred_ids = np.argmax(preds.predictions, axis=-1)

    out_pred = pd.DataFrame({
        "id": df_test_unlab["id"].astype(str),
        "pred_label": ["SUBJ" if p==1 else "OBJ" for p in pred_ids],
    })
    out_pred.to_csv(out_dir / "predictions.tsv", sep="\t", index=False, encoding="utf-8")

    if df_test_lab is not None and len(df_test_lab) == len(out_pred):
        y_true = map_labels(df_test_lab["label"]).astype(int).tolist()
        y_pred = pred_ids.tolist()
        metrics = evaluate_predictions(y_true, y_pred)
        write_metrics(out_dir / "metrics.txt", metrics, y_true, y_pred)
    else:
        with open(out_dir / "metrics.txt", "w", encoding="utf-8") as f:
            f.write("No labeled test or size mismatch.\n")

    (out_dir / "completed.flag").write_text("ok", encoding="utf-8")

def build_seen_multilingual_df(data_root: Path):
    dfs = []
    for lang in SEEN_LANGS_FOR_MULTI:
        code = LANG_CODES[lang]
        lang_dir = data_root / lang
        if not lang_dir.exists():
            continue
        for fname in [f"train_{code}.tsv", f"dev_{code}.tsv"]:
            fpath = lang_dir / fname
            if fpath.exists():
                try:
                    df = load_tsv(fpath, is_labeled=True)
                    dfs.append(df[["id","text","label"]])
                except Exception as e:
                    logger.warning(f"Could not load {fpath}: {e}")
    if not dfs:
        raise RuntimeError("No seen data found.")
    all_df = pd.concat(dfs, axis=0, ignore_index=True)
    all_df = all_df.dropna(subset=["label"])
    return all_df

def run_monolingual(language: str, data_root: Path, outputs_root: Path, max_length: int):
    code = LANG_CODES[language]
    lang_dir = data_root / language
    if not lang_dir.exists():
        raise RuntimeError(f"Folder not found: {lang_dir}")

    base_model = ensure_model(PAPER_MONO_MODELS.get(language, FALLBACK_MODEL))
    out_dir, ckpt_dir = make_output_dirs(outputs_root, language, base_model)
    
    if is_completed(out_dir):
        logger.info(f"Already completed for {language}")
        return

    logger.info(f"Training {language} with {base_model}")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    config = AutoConfig.from_pretrained(base_model, num_labels=2, label2id={"OBJ":0,"SUBJ":1}, id2label={0:"OBJ",1:"SUBJ"})
    lr = select_lr(base_model)

    train_path = lang_dir / f"train_{code}.tsv"
    dev_path = lang_dir / f"dev_{code}.tsv"
    dev_test_path = lang_dir / f"dev_test_{code}.tsv"
    test_unl_path = lang_dir / f"test_{code}_unlabeled.tsv"
    test_lab_path = lang_dir / f"test_{code}_labeled.tsv"

    df_train = load_tsv(train_path, True) if train_path.exists() else None
    df_dev = load_tsv(dev_path, True) if dev_path.exists() else None

    if df_train is None and df_dev is None:
        raise RuntimeError(f"No train/dev for {language}")

    ds_train = df_to_ds(df_train, tokenizer, max_length) if df_train is not None else None
    ds_dev = df_to_ds(df_dev, tokenizer, max_length) if df_dev is not None else None

    model = two_phase_train(tokenizer, base_model, config, ds_train, ds_dev, out_dir, ckpt_dir, lr, max_length)

    if test_unl_path.exists():
        df_unlab = load_tsv(test_unl_path, False)
        ds_test = df_to_ds(df_unlab, tokenizer, max_length)
        df_lab = load_tsv(test_lab_path, True) if test_lab_path.exists() else None
    else:
        df_unlab = load_tsv(dev_test_path, True) if dev_test_path.exists() else None
        ds_test = df_to_ds(df_unlab, tokenizer, max_length) if df_unlab is not None else None
        df_lab = df_unlab

    if ds_test is not None:
        predict_and_save(model, tokenizer, ds_test, df_unlab, df_lab, out_dir)
    else:
        (out_dir / "completed.flag").write_text("ok")

def run_multilingual(data_root: Path, outputs_root: Path, base_model: str, max_length: int):
    base_model = ensure_model(base_model)
    out_dir, ckpt_dir = make_output_dirs(outputs_root, "multilingual", base_model)
    
    if is_completed(out_dir):
        logger.info(f"Multilingual {base_model} done")
        return

    logger.info(f"Training multilingual {base_model}")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    config = AutoConfig.from_pretrained(base_model, num_labels=2, label2id={"OBJ":0,"SUBJ":1}, id2label={0:"OBJ",1:"SUBJ"})
    lr = select_lr(base_model)

    df_all = build_seen_multilingual_df(data_root)
    df_all = df_all.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    split = int(0.9 * len(df_all))
    df_train = df_all.iloc[:split].reset_index(drop=True)
    df_dev = df_all.iloc[split:].reset_index(drop=True)

    ds_train = df_to_ds(df_train, tokenizer, max_length)
    ds_dev = df_to_ds(df_dev, tokenizer, max_length)

    model = two_phase_train(tokenizer, base_model, config, ds_train, ds_dev, out_dir, ckpt_dir, lr, max_length)
    (out_dir / "completed.flag").write_text("ok")

def run_zero_shot(language: str, data_root: Path, outputs_root: Path, max_length: int):
    base_model = ensure_model(PAPER_ZERO_SHOT_BASE.get(language, FALLBACK_MODEL))
    
    multi_out_dir, _ = make_output_dirs(outputs_root, "multilingual", base_model)
    final_multi_model_dir = multi_out_dir / "final_model"
    
    if not final_multi_model_dir.exists():
        logger.info(f"Training multilingual {base_model} first")
        run_multilingual(data_root, outputs_root, base_model, max_length)

    logger.info(f"Zero-shot {language} using {base_model}")
    
    tokenizer = AutoTokenizer.from_pretrained(final_multi_model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(final_multi_model_dir)

    code = LANG_CODES[language]
    lang_dir = data_root / language
    if not lang_dir.exists():
        raise RuntimeError(f"Folder not found: {lang_dir}")

    test_unl_path = lang_dir / f"test_{code}_unlabeled.tsv"
    test_lab_path = lang_dir / f"test_{code}_labeled.tsv"

    df_unlab = load_tsv(test_unl_path, False) if test_unl_path.exists() else None
    df_lab = load_tsv(test_lab_path, True) if test_lab_path.exists() else None
    
    if df_unlab is None:
        raise RuntimeError(f"No test for {language}")

    ds_test = df_to_ds(df_unlab, tokenizer, max_length)
    out_dir, _ = make_output_dirs(outputs_root, language, base_model)
    
    if is_completed(out_dir):
        logger.info(f"Already done for {language}")
        return
        
    predict_and_save(model, tokenizer, ds_test, df_unlab, df_lab, out_dir)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", type=str, required=True, choices=list(LANG_CODES.keys()))
    ap.add_argument("--base_model", type=str, default=None)
    ap.add_argument("--data_root", type=str, default=None)
    ap.add_argument("--output_root", type=str, default=None)
    ap.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH)
    args = ap.parse_args()

    set_seed(SEED)
    
    data_root = Path(args.data_root) if args.data_root else Path.cwd() / "data"
    outputs_root = Path(args.output_root) if args.output_root else Path.cwd() / "outputs"

    logger.info(f"=== Running {args.lang} ===")

    if args.lang == "multilingual":
        base_model = args.base_model or "microsoft/infoxlm-large"
        run_multilingual(data_root, outputs_root, base_model, args.max_length)
    elif args.lang in ZERO_SHOT_ONLY_LANGS:
        run_zero_shot(args.lang, data_root, outputs_root, args.max_length)
    elif args.lang in SEEN_LANGS_FOR_MULTI:
        run_monolingual(args.lang, data_root, outputs_root, args.max_length)
    else:
        raise ValueError(f"Unknown: {args.lang}")

    logger.info(f"✅ Done {args.lang}")

if __name__ == "__main__":
    main()
