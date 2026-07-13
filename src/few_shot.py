"""
Stage 5 - Few-shot / transfer learning on frozen fine-tuned BERT embeddings.

Run:
    python src/few_shot.py --model_dir models/bert-base-uncased/final

This freezes your fine-tuned encoder, extracts the [CLS] embedding for every
post, then trains a tiny logistic-regression classifier on just N examples
per class. Expect the accuracy curve to saturate almost immediately - that's
a real and interesting finding (it measures feature separability of the
already-fine-tuned encoder), not a bug.

Requires: pip install torch transformers scikit-learn pandas matplotlib
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_PROCESSED_DIR, MODELS_DIR, MAX_SEQ_LEN, RANDOM_SEED


@torch.no_grad()
def extract_cls_embeddings(texts, tokenizer, model, device, batch_size=32):
    model.eval()
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, truncation=True, padding=True, max_length=MAX_SEQ_LEN, return_tensors="pt").to(device)
        out = model(**enc)
        cls = out.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(cls)
    return np.vstack(embeddings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="Path to a fine-tuned model, e.g. models/bert-base-uncased/final")
    parser.add_argument("--sample_sizes", nargs="+", type=int, default=[5, 10, 20, 50, 100, 200])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModel.from_pretrained(args.model_dir).to(device)

    train_df = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "train.csv")).fillna({"text": ""})
    test_df = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "test.csv")).fillna({"text": ""})

    print("Extracting embeddings for train set (may take a while)...")
    X_train_full = extract_cls_embeddings(train_df["text"].tolist(), tokenizer, model, device)
    y_train_full = train_df["label"].values

    print("Extracting embeddings for test set...")
    X_test = extract_cls_embeddings(test_df["text"].tolist(), tokenizer, model, device)
    y_test = test_df["label"].values

    rng = np.random.default_rng(RANDOM_SEED)
    results = []

    for n in args.sample_sizes:
        accs, f1s = [], []
        for run in range(5):
            idx = []
            for label in np.unique(y_train_full):
                label_idx = np.where(y_train_full == label)[0]
                chosen = rng.choice(label_idx, size=min(n, len(label_idx)), replace=False)
                idx.extend(chosen)
            idx = np.array(idx)

            clf = LogisticRegression(max_iter=1000)
            clf.fit(X_train_full[idx], y_train_full[idx])
            preds = clf.predict(X_test)
            accs.append(accuracy_score(y_test, preds))
            f1s.append(f1_score(y_test, preds, average="weighted"))

        results.append({"n": n, "acc_mean": np.mean(accs), "f1_mean": np.mean(f1s)})
        print(f"N={n}: acc={np.mean(accs):.4f}  f1={np.mean(f1s):.4f}")

    # 10-shot prototype (NearestCentroid) classifier, for comparison.
    idx = []
    for label in np.unique(y_train_full):
        label_idx = np.where(y_train_full == label)[0]
        chosen = rng.choice(label_idx, size=min(10, len(label_idx)), replace=False)
        idx.extend(chosen)
    idx = np.array(idx)
    proto = NearestCentroid()
    proto.fit(X_train_full[idx], y_train_full[idx])
    proto_preds = proto.predict(X_test)
    print(f"10-shot prototype: acc={accuracy_score(y_test, proto_preds):.4f}  "
          f"f1={f1_score(y_test, proto_preds, average='weighted'):.4f}")

    df_results = pd.DataFrame(results)
    out_dir = os.path.join(MODELS_DIR, "few_shot")
    os.makedirs(out_dir, exist_ok=True)
    df_results.to_csv(os.path.join(out_dir, "few_shot_results.csv"), index=False)

    plt.figure(figsize=(7, 5))
    plt.errorbar(df_results["n"], df_results["acc_mean"], marker="o")
    plt.xscale("log")
    plt.xlabel("Samples per class")
    plt.ylabel("Test accuracy")
    plt.title("Few-Shot Accuracy vs. Samples per Class")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "few_shot_curve.png"), dpi=150)
    print(f"Saved results to {out_dir}")


if __name__ == "__main__":
    main()
