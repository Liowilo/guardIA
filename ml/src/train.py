"""Fine-tune multilingual MiniLM for multi-label grooming-pattern detection.

Architecture
------------
- Base: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- Head: linear layer with 5 sigmoids (one per category), BCEWithLogitsLoss via
  Trainer's `problem_type="multi_label_classification"`.

Outputs the best checkpoint (by validation micro-F1) to MODEL_DIR.

Usage
-----
    python -m src.train
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score, hamming_loss
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from .config import (
    BASE_MODEL,
    CATEGORIES,
    DEFAULT_THRESHOLD,
    EVAL_BATCH,
    LR,
    MAX_LEN,
    MODEL_DIR,
    NUM_EPOCHS,
    PAN_LABELED_PATH,
    SEED,
    TRAIN_BATCH,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)


def to_dataset(df: pd.DataFrame) -> Dataset:
    keep = df[["text_es"] + CATEGORIES].rename(columns={"text_es": "text"})
    keep["labels"] = keep[CATEGORIES].values.astype(np.float32).tolist()
    return Dataset.from_pandas(keep[["text", "labels"]], preserve_index=False)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= DEFAULT_THRESHOLD).astype(int)
    labels = labels.astype(int)
    return {
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "hamming_loss": hamming_loss(labels, preds),
        **{
            f"f1_{cat}": f1_score(labels[:, i], preds[:, i], zero_division=0)
            for i, cat in enumerate(CATEGORIES)
        },
    }


def main() -> int:
    if not PAN_LABELED_PATH.exists():
        print(f"[train] missing {PAN_LABELED_PATH}; run src.weak_label first")
        return 1

    df = pd.read_parquet(PAN_LABELED_PATH)

    # Downsample all-zero rows so the model does not collapse to "always safe".
    # Keep all rows with >=1 positive, plus 3x that many all-zero rows.
    any_pos = df[CATEGORIES].sum(axis=1) > 0
    pos_df = df[any_pos]
    neg_df = df[~any_pos]
    target_neg = min(len(neg_df), max(len(pos_df) * 3, 2000))
    neg_df = neg_df.sample(n=target_neg, random_state=SEED)
    df = pd.concat([pos_df, neg_df], ignore_index=True).sample(frac=1, random_state=SEED)
    print(
        f"[train] balanced: pos_rows={len(pos_df):,} neg_rows={len(neg_df):,} "
        f"total={len(df):,}"
    )

    train_df, valtest_df = train_test_split(
        df, test_size=0.2, random_state=SEED, shuffle=True
    )
    val_df, test_df = train_test_split(
        valtest_df, test_size=0.5, random_state=SEED, shuffle=True
    )
    print(f"[train] train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}")

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)

    def tok_fn(batch):
        return tok(batch["text"], truncation=True, max_length=MAX_LEN)

    train_ds = to_dataset(train_df).map(tok_fn, batched=True, remove_columns=["text"])
    val_ds = to_dataset(val_df).map(tok_fn, batched=True, remove_columns=["text"])
    test_ds = to_dataset(test_df).map(tok_fn, batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(CATEGORIES),
        problem_type="multi_label_classification",
        id2label={i: c for i, c in enumerate(CATEGORIES)},
        label2id={c: i for i, c in enumerate(CATEGORIES)},
    )

    args = TrainingArguments(
        output_dir=str(MODEL_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH,
        per_device_eval_batch_size=EVAL_BATCH,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        fp16=torch.cuda.is_available(),
        save_total_limit=2,
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tok,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("[train] test set metrics:")
    print(trainer.evaluate(test_ds))

    # Save the best model + tokenizer to a clean export dir (not the checkpoints).
    final_dir = MODEL_DIR / "final"
    trainer.save_model(str(final_dir))
    tok.save_pretrained(str(final_dir))
    print(f"[train] final model saved to {final_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
