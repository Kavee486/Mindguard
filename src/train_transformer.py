"""
Stage 4 - Fine-tune a transformer (BERT or RoBERTa) for 5-way classification.

Run:
    python src/train_transformer.py --model bert-base-uncased --epochs 3
    python src/train_transformer.py --model roberta-base --epochs 3

This is the GPU-heavy stage. If you hit "CUDA out of memory", lower
--batch_size first (try 8), then --max_len (try 128).

Requires: pip install torch transformers datasets scikit-learn pandas
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_PROCESSED_DIR, MODELS_DIR, LABELS, LABEL2ID, ID2LABEL, MAX_SEQ_LEN, RANDOM_SEED


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bert-base-uncased",
                         help="HF checkpoint, e.g. bert-base-uncased or roberta-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    torch.manual_seed(RANDOM_SEED)

    train_df = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "train.csv")).fillna({"text": ""})
    val_df = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "val.csv")).fillna({"text": ""})
    test_df = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "test.csv")).fillna({"text": ""})

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=args.max_len)

    train_ds = Dataset.from_pandas(train_df[["text", "label"]]).map(tokenize, batched=True)
    val_ds = Dataset.from_pandas(val_df[["text", "label"]]).map(tokenize, batched=True)
    test_ds = Dataset.from_pandas(test_df[["text", "label"]]).map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    model_short_name = args.model.split("/")[-1]
    out_dir = os.path.join(MODELS_DIR, model_short_name)

    training_args = TrainingArguments(
        output_dir=os.path.join(out_dir, "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        logging_steps=50,
        bf16=torch.cuda.is_available(),
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    test_metrics = trainer.evaluate(test_ds)
    print("Test set metrics:", test_metrics)

    final_dir = os.path.join(out_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved fine-tuned model to {final_dir}")


if __name__ == "__main__":
    main()
