"""
Stage 6 - Prompt engineering against a local LLM served through Ollama.

Prerequisite:
    1. Install Ollama: https://ollama.com
    2. Pull a model:   ollama pull qwen2.5:7b   (or qwen2.5:14b if you have the VRAM)
    3. Make sure Ollama is running (it starts a local server on :11434).

Run:
    python src/prompting.py --n_per_class 50

Requires: pip install requests pandas scikit-learn
"""
import os
import sys
import json
import argparse
import time

import pandas as pd
import requests
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_PROCESSED_DIR, MODELS_DIR, LABELS, LABEL2ID, OLLAMA_HOST, PROMPTING_MODEL, RANDOM_SEED

LABEL_LIST_STR = ", ".join(LABELS)


def call_ollama(prompt: str, model: str = PROMPTING_MODEL, temperature: float = 0.0) -> str:
    resp = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": temperature}},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def parse_label(raw_response: str) -> str:
    lower = raw_response.lower()
    for label in LABELS:
        if label.lower() in lower:
            return label
    return LABELS[0]  # fallback if the model didn't answer cleanly


def build_zero_shot_prompt(text: str) -> str:
    return (
        f"Classify the following Reddit post into exactly one of these categories: "
        f"{LABEL_LIST_STR}.\n\nPost: \"{text}\"\n\n"
        f"Answer with only the category name, nothing else."
    )


def build_few_shot_prompt(text: str, examples) -> str:
    exemplar_block = "\n\n".join(
        f"Post: \"{ex['text']}\"\nCategory: {ex['label_name']}" for ex in examples
    )
    return (
        f"Classify Reddit posts into exactly one of: {LABEL_LIST_STR}.\n\n"
        f"Examples:\n{exemplar_block}\n\n"
        f"Now classify this post:\nPost: \"{text}\"\nCategory:"
    )


def build_cot_prompt(text: str) -> str:
    return (
        f"A user wrote the following post. First briefly reason about the emotional "
        f"and behavioural themes present, then decide which single category it belongs "
        f"to: {LABEL_LIST_STR}.\n\nPost: \"{text}\"\n\n"
        f"Respond in this format:\nReasoning: <one or two sentences>\nCategory: <one label>"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_per_class", type=int, default=50, help="Stratified subset size per class")
    parser.add_argument("--n_examples", type=int, default=6, help="Few-shot exemplar count")
    args = parser.parse_args()

    test_df = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "test.csv")).fillna({"text": ""})
    train_df = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "train.csv")).fillna({"text": ""})

    subset = (
        test_df.groupby("label_name", group_keys=False)
        .apply(lambda x: x.sample(min(args.n_per_class, len(x)), random_state=RANDOM_SEED))
        .reset_index(drop=True)
    )
    print(f"Evaluating on {len(subset)} posts ({args.n_per_class} per class)")

    examples = (
        train_df.groupby("label_name", group_keys=False)
        .apply(lambda x: x.sample(1, random_state=RANDOM_SEED))
        .reset_index(drop=True)
        .to_dict("records")[: args.n_examples]
    )

    results = {}
    for strategy, builder in [
        ("zero_shot", lambda t: build_zero_shot_prompt(t)),
        ("few_shot", lambda t: build_few_shot_prompt(t, examples)),
        ("chain_of_thought", lambda t: build_cot_prompt(t)),
    ]:
        print(f"\nRunning strategy: {strategy}")
        preds, gold = [], []
        for i, row in subset.iterrows():
            prompt = builder(row["text"][:1500])  # keep prompts reasonably short
            try:
                raw = call_ollama(prompt)
            except requests.exceptions.RequestException as e:
                print(f"  Ollama call failed ({e}); is `ollama serve` running?")
                raise
            pred_label = parse_label(raw)
            preds.append(LABEL2ID[pred_label])
            gold.append(row["label"])
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(subset)} done")

        acc = accuracy_score(gold, preds)
        f1 = f1_score(gold, preds, average="weighted")
        results[strategy] = {"accuracy": acc, "f1_weighted": f1}
        print(f"  {strategy}: accuracy={acc:.4f}  f1={f1:.4f}")

    out_dir = os.path.join(MODELS_DIR, "prompting")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_dir}/results.json")


if __name__ == "__main__":
    main()
