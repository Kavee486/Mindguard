"""
Stage 1 - Data download, cleaning, and train/val/test split.

Run:
    python src/data_prep.py

What it does:
  1. Downloads the Reddit mental-health classification dataset from
     HuggingFace (no login needed).
  2. Builds a `text` field as title + body, cleans it, and drops empty /
     too-short posts (mirrors the "body-only filter" from the report).
  3. Saves a class-distribution plot to data/processed/class_distribution.png
  4. Splits 80/10/10 and saves train.csv / val.csv / test.csv.
  5. Also downloads the counseling-conversations dataset used later for
     response-generation fine-tuning, and saves it as counseling.csv.

Requires: pip install datasets pandas scikit-learn matplotlib
"""
import os
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATA_PROCESSED_DIR,
    CLASSIFICATION_DATASET,
    COUNSELING_DATASET,
    LABEL2ID,
    RANDOM_SEED,
)
from utils import clean_text, is_substantive


def load_classification_data() -> pd.DataFrame:
    from datasets import load_dataset

    print(f"Downloading {CLASSIFICATION_DATASET} from HuggingFace...")
    ds = load_dataset(CLASSIFICATION_DATASET)
    split_name = "train" if "train" in ds else list(ds.keys())[0]
    df = ds[split_name].to_pandas()

    # Expected columns: title, body, subreddit (label source).
    df["title"] = df.get("title", "").fillna("")
    df["body"] = df.get("body", "").fillna("")
    df["subreddit"] = df["subreddit"].astype(str)

    # Normalise subreddit names to our fixed label set.
    def normalise_label(s: str) -> str:
        s = s.lower()
        mapping = {
            "adhd": "ADHD",
            "ocd": "OCD",
            "aspergers": "aspergers",
            "asperger": "aspergers",
            "aspergersyndrome": "aspergers",
            "depression": "depression",
            "ptsd": "ptsd",
        }
        return mapping.get(s, s)

    df["label_name"] = df["subreddit"].apply(normalise_label)
    print(f"Raw rows: {len(df)}")

    known_mask = df["label_name"].isin(LABEL2ID.keys())
    dropped = df.loc[~known_mask, "subreddit"].value_counts()
    if len(dropped):
        print(f"Dropping {len(df) - known_mask.sum()} rows with unrecognized subreddit labels:")
        print(dropped.head(20).to_string())
    df = df[known_mask].copy()

    # Body-only filter: keep posts with substantive body text (mirrors the
    # report's finding that ~42% of posts had empty/deleted bodies).
    df["body_clean"] = df["body"].apply(clean_text)
    df = df[df["body_clean"].apply(is_substantive)].copy()

    df["title_clean"] = df["title"].apply(clean_text)
    df["text"] = (df["title_clean"] + " " + df["body_clean"]).str.strip()
    df["label"] = df["label_name"].map(LABEL2ID)

    df = df[["text", "label_name", "label"]].reset_index(drop=True)
    print(f"After body-only filter: {len(df)} posts")
    print(df["label_name"].value_counts())

    return df


def load_counseling_data() -> pd.DataFrame:
    from datasets import load_dataset

    print(f"Downloading {COUNSELING_DATASET} from HuggingFace...")
    ds = load_dataset(COUNSELING_DATASET)
    split_name = "train" if "train" in ds else list(ds.keys())[0]
    df = ds[split_name].to_pandas()
    # Expected columns: Context, Response
    df = df.rename(columns={"Context": "context", "Response": "response"})
    df["context"] = df["context"].apply(clean_text)
    df["response"] = df["response"].fillna("").astype(str).str.strip()
    df = df[(df["context"].str.len() > 0) & (df["response"].str.len() > 0)]
    print(f"Counseling pairs: {len(df)}")
    return df.reset_index(drop=True)


def plot_class_distribution(df: pd.DataFrame, out_path: str):
    counts = df["label_name"].value_counts()
    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar", color="#4C72B0")
    plt.title("Class Distribution - Mental Health Conditions (Body-Only Posts)")
    plt.xlabel("Condition / Label")
    plt.ylabel("Number of Samples")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved class distribution plot to {out_path}")


def main():
    df = load_classification_data()

    plot_class_distribution(
        df, os.path.join(DATA_PROCESSED_DIR, "class_distribution.png")
    )

    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=RANDOM_SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label"], random_state=RANDOM_SEED
    )

    train_df.to_csv(os.path.join(DATA_PROCESSED_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(DATA_PROCESSED_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(DATA_PROCESSED_DIR, "test.csv"), index=False)

    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    counseling_df = load_counseling_data()
    counseling_df.to_csv(
        os.path.join(DATA_PROCESSED_DIR, "counseling.csv"), index=False
    )

    print("Done. Processed files are in data/processed/")


if __name__ == "__main__":
    main()
