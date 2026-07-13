"""
Stage 8 - Advanced evaluation: cross-validation, ensembling, significance
testing, interpretability (SHAP), and embedding visualisation (t-SNE).

Run:
    python src/evaluate.py --bert_dir models/bert-base-uncased/final --roberta_dir models/roberta-base/final

Requires:
    pip install torch transformers scikit-learn pandas matplotlib shap statsmodels seaborn
"""
import os
import sys
import argparse
import json

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.manifold import TSNE
from sklearn.preprocessing import label_binarize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_PROCESSED_DIR, MODELS_DIR, LABELS, MAX_SEQ_LEN


@torch.no_grad()
def get_probs(model_dir, texts, device, batch_size=32):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, truncation=True, padding=True, max_length=MAX_SEQ_LEN, return_tensors="pt").to(device)
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
    return np.vstack(all_probs)


def mcnemar_test(y_true, preds_a, preds_b):
    """McNemar's test comparing two classifiers on the same test set."""
    from statsmodels.stats.contingency_tables import mcnemar

    correct_a = (preds_a == y_true)
    correct_b = (preds_b == y_true)
    n01 = np.sum(correct_a & ~correct_b)
    n10 = np.sum(~correct_a & correct_b)
    table = [[0, n01], [n10, 0]]
    result = mcnemar(table, exact=False, correction=True)
    return result.statistic, result.pvalue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_dir", default=os.path.join(MODELS_DIR, "bert-base-uncased", "final"))
    parser.add_argument("--roberta_dir", default=os.path.join(MODELS_DIR, "roberta-base", "final"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = os.path.join(MODELS_DIR, "evaluation")
    os.makedirs(out_dir, exist_ok=True)

    test_df = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "test.csv")).fillna({"text": ""})
    y_test = test_df["label"].values
    texts = test_df["text"].tolist()

    # --- 5-fold cross-validation of the baseline ---
    train_df = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "train.csv")).fillna({"text": ""})
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)
    X_all = vectorizer.fit_transform(train_df["text"])
    cv_scores = cross_val_score(LinearSVC(), X_all, train_df["label"], cv=5, scoring="f1_weighted")
    print(f"5-fold CV baseline F1: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    # --- Ensemble BERT + RoBERTa ---
    if os.path.isdir(args.bert_dir) and os.path.isdir(args.roberta_dir):
        print("Scoring BERT and RoBERTa on the test set...")
        bert_probs = get_probs(args.bert_dir, texts, device)
        roberta_probs = get_probs(args.roberta_dir, texts, device)

        bert_preds = bert_probs.argmax(axis=1)
        roberta_preds = roberta_probs.argmax(axis=1)
        ensemble_probs = (bert_probs + roberta_probs) / 2
        ensemble_preds = ensemble_probs.argmax(axis=1)

        for name, preds in [("BERT", bert_preds), ("RoBERTa", roberta_preds), ("Ensemble", ensemble_preds)]:
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="weighted")
            print(f"{name}: accuracy={acc:.4f}  f1={f1:.4f}")

        stat, p = mcnemar_test(y_test, bert_preds, roberta_preds)
        print(f"McNemar's test (BERT vs RoBERTa): statistic={stat:.4f}  p={p:.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, ensemble_preds)
        plt.figure(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
        plt.title("Ensemble (BERT + RoBERTa) - Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
        plt.close()

        # ROC curves (one-vs-rest)
        y_bin = label_binarize(y_test, classes=list(range(len(LABELS))))
        plt.figure(figsize=(7, 6))
        for i, label in enumerate(LABELS):
            fpr, tpr, _ = roc_curve(y_bin[:, i], ensemble_probs[:, i])
            plt.plot(fpr, tpr, label=f"{label} (AUC={auc(fpr, tpr):.3f})")
        plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Ensemble ROC Curves (One-vs-Rest)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "roc_curves.png"), dpi=150)
        plt.close()

        # t-SNE of BERT embeddings (reuse logits' penultimate layer via hidden states)
        print("Computing t-SNE of BERT embeddings (this can take a minute)...")
        tokenizer = AutoTokenizer.from_pretrained(args.bert_dir)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.bert_dir, output_hidden_states=True
        ).to(device)
        model.eval()
        sample_idx = np.random.default_rng(42).choice(len(texts), size=min(1500, len(texts)), replace=False)
        sample_texts = [texts[i] for i in sample_idx]
        sample_labels = y_test[sample_idx]

        cls_vectors = []
        with torch.no_grad():
            for i in range(0, len(sample_texts), 32):
                batch = sample_texts[i:i + 32]
                enc = tokenizer(batch, truncation=True, padding=True, max_length=MAX_SEQ_LEN, return_tensors="pt").to(device)
                hidden = model(**enc).hidden_states[-1]
                cls_vectors.append(hidden[:, 0, :].cpu().numpy())
        cls_vectors = np.vstack(cls_vectors)

        tsne = TSNE(n_components=2, random_state=42, init="pca")
        proj = tsne.fit_transform(cls_vectors)

        plt.figure(figsize=(7, 6))
        for i, label in enumerate(LABELS):
            mask = sample_labels == i
            plt.scatter(proj[mask, 0], proj[mask, 1], label=label, s=8, alpha=0.7)
        plt.legend()
        plt.title("t-SNE of BERT [CLS] Embeddings")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "tsne.png"), dpi=150)
        plt.close()

        summary = {
            "cv_baseline_f1_mean": cv_scores.mean(),
            "cv_baseline_f1_std": cv_scores.std(),
            "bert_acc": accuracy_score(y_test, bert_preds),
            "roberta_acc": accuracy_score(y_test, roberta_preds),
            "ensemble_acc": accuracy_score(y_test, ensemble_preds),
            "mcnemar_statistic": stat,
            "mcnemar_pvalue": p,
        }
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved plots + summary to {out_dir}")
    else:
        print("BERT/RoBERTa model directories not found - run train_transformer.py for both first.")


if __name__ == "__main__":
    main()
