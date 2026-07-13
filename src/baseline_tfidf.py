"""
Stage 2 - Baseline classifier: TF-IDF + Linear SVM.

Run:
    python src/baseline_tfidf.py

This is your first working model and the number every later stage should
beat. Expect roughly 85-87% accuracy - lexical cues alone separate these
five subreddits fairly well.

Requires: pip install scikit-learn pandas joblib
"""
import os
import sys
import json

import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_PROCESSED_DIR, MODELS_DIR, LABELS


def main():
    train_df = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "test.csv"))

    train_df["text"] = train_df["text"].fillna("")
    test_df["text"] = test_df["text"].fillna("")

    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)
    X_train = vectorizer.fit_transform(train_df["text"])
    X_test = vectorizer.transform(test_df["text"])

    clf = LinearSVC()
    clf.fit(X_train, train_df["label"])

    preds = clf.predict(X_test)
    acc = accuracy_score(test_df["label"], preds)
    f1 = f1_score(test_df["label"], preds, average="weighted")

    print(f"Accuracy: {acc:.4f}")
    print(f"Weighted F1: {f1:.4f}")
    print(classification_report(test_df["label"], preds, target_names=LABELS))
    print("Confusion matrix:")
    print(confusion_matrix(test_df["label"], preds))

    out_dir = os.path.join(MODELS_DIR, "baseline_tfidf_svm")
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(out_dir, "vectorizer.joblib"))
    joblib.dump(clf, os.path.join(out_dir, "svm.joblib"))

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"accuracy": acc, "f1_weighted": f1}, f, indent=2)

    print(f"Saved model + metrics to {out_dir}")


if __name__ == "__main__":
    main()
