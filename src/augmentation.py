"""
Stage 7 - Leakage-safe generative augmentation.

Uses the local Ollama LLM to paraphrase posts from under-represented
classes, then retrains the TF-IDF+SVM baseline on original vs. augmented
training data - evaluated on the SAME untouched test set both times, so any
difference you see is real and not an artefact of data leakage.

Run:
    python src/augmentation.py --target_classes aspergers ptsd --n_paraphrases 500

Requires: pip install requests pandas scikit-learn
"""
import os
import sys
import argparse

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_PROCESSED_DIR, MODELS_DIR, LABEL2ID, RANDOM_SEED
from prompting import call_ollama


def paraphrase_prompt(text: str) -> str:
    return (
        "Paraphrase the following personal post in a different way, keeping the "
        "same meaning, tone, and level of detail. Return only the paraphrased text.\n\n"
        f"Post: \"{text[:1200]}\""
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_classes", nargs="+", default=["aspergers", "ptsd"])
    parser.add_argument("--n_paraphrases", type=int, default=500)
    parser.add_argument("--sim_low", type=float, default=0.3)
    parser.add_argument("--sim_high", type=float, default=0.9)
    args = parser.parse_args()

    train_df = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "train.csv")).fillna({"text": ""})
    test_df = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "test.csv")).fillna({"text": ""})

    # --- Baseline (original training data only) ---
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)
    X_train = vectorizer.fit_transform(train_df["text"])
    X_test = vectorizer.transform(test_df["text"])
    clf = LinearSVC()
    clf.fit(X_train, train_df["label"])
    preds = clf.predict(X_test)
    base_acc = accuracy_score(test_df["label"], preds)
    base_f1 = f1_score(test_df["label"], preds, average="weighted")
    print(f"Original train -> accuracy={base_acc:.4f}  f1={base_f1:.4f}")

    # --- Generate paraphrases for target (under-represented) classes ---
    candidates = train_df[train_df["label_name"].isin(args.target_classes)].sample(
        min(args.n_paraphrases, len(train_df)), random_state=RANDOM_SEED
    )

    rng = np.random.default_rng(RANDOM_SEED)
    new_rows = []
    print(f"Generating up to {len(candidates)} paraphrases via Ollama (this can take a while)...")
    for i, row in candidates.iterrows():
        try:
            paraphrase = call_ollama(paraphrase_prompt(row["text"]))
        except Exception as e:
            print(f"  Skipping (Ollama error: {e})")
            continue

        # Cosine similarity filter using the already-fit TF-IDF vectorizer.
        orig_vec = vectorizer.transform([row["text"]])
        para_vec = vectorizer.transform([paraphrase])
        sim = cosine_similarity(orig_vec, para_vec)[0][0]

        if args.sim_low <= sim <= args.sim_high:
            new_rows.append({
                "text": paraphrase,
                "label_name": row["label_name"],
                "label": row["label"],
            })

    print(f"{len(new_rows)} paraphrases passed the similarity filter ({args.sim_low}-{args.sim_high})")

    augmented_train = pd.concat([train_df, pd.DataFrame(new_rows)], ignore_index=True)

    # --- Retrain on augmented data, evaluate on the SAME test set ---
    vectorizer2 = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)
    X_train2 = vectorizer2.fit_transform(augmented_train["text"])
    X_test2 = vectorizer2.transform(test_df["text"])
    clf2 = LinearSVC()
    clf2.fit(X_train2, augmented_train["label"])
    preds2 = clf2.predict(X_test2)
    aug_acc = accuracy_score(test_df["label"], preds2)
    aug_f1 = f1_score(test_df["label"], preds2, average="weighted")
    print(f"Augmented train -> accuracy={aug_acc:.4f}  f1={aug_f1:.4f}")

    print(f"\nDelta: accuracy={aug_acc - base_acc:+.4f}  f1={aug_f1 - base_f1:+.4f}")
    print("A small/neutral delta is a legitimate result when the original data wasn't scarce.")

    out_dir = os.path.join(MODELS_DIR, "augmentation")
    os.makedirs(out_dir, exist_ok=True)
    augmented_train.to_csv(os.path.join(out_dir, "augmented_train.csv"), index=False)


if __name__ == "__main__":
    main()
