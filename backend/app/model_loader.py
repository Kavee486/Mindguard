"""
Loads whichever trained models are available on disk. Every model is
optional - the API degrades gracefully and tells you what's missing rather
than crashing, since a beginner working through the stages won't have
everything trained on day one.
"""
import os
import sys
import json
import glob

import joblib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from config import MODELS_DIR, LABELS, MAX_SEQ_LEN  # noqa: E402

_STATE = {
    "baseline_vectorizer": None,
    "baseline_svm": None,
    "bert_tokenizer": None,
    "bert_model": None,
    "roberta_tokenizer": None,
    "roberta_model": None,
    "device": "cpu",
}


def load_all():
    # torch/transformers are only required for the BERT/RoBERTa stages. If
    # they're not installed yet (e.g. you're only trying the baseline model
    # first), skip transformer loading instead of crashing the whole API.
    torch_available = True
    try:
        import torch
        _STATE["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        torch_available = False
        print("[model_loader] torch not installed - BERT/RoBERTa endpoints will be "
              "unavailable until you `pip install torch` (see README.md Section 1).")

    print(f"[model_loader] Using device: {_STATE['device']}")

    baseline_dir = os.path.join(MODELS_DIR, "baseline_tfidf_svm")
    if os.path.exists(os.path.join(baseline_dir, "vectorizer.joblib")):
        _STATE["baseline_vectorizer"] = joblib.load(os.path.join(baseline_dir, "vectorizer.joblib"))
        _STATE["baseline_svm"] = joblib.load(os.path.join(baseline_dir, "svm.joblib"))
        print("[model_loader] Loaded baseline TF-IDF + SVM")
    else:
        print("[model_loader] Baseline model not found - run src/baseline_tfidf.py")

    if not torch_available:
        return

    for name in ["bert-base-uncased", "roberta-base"]:
        model_dir = os.path.join(MODELS_DIR, name, "final")
        key_prefix = "bert" if "bert-base" in name else "roberta"
        if os.path.isdir(model_dir):
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            _STATE[f"{key_prefix}_tokenizer"] = AutoTokenizer.from_pretrained(model_dir)
            _STATE[f"{key_prefix}_model"] = AutoModelForSequenceClassification.from_pretrained(
                model_dir, output_attentions=True
            ).to(_STATE["device"])
            _STATE[f"{key_prefix}_model"].eval()
            print(f"[model_loader] Loaded {name}")
        else:
            print(f"[model_loader] {name} not found at {model_dir} - run src/train_transformer.py --model {name}")


def available_models():
    avail = []
    if _STATE["baseline_svm"] is not None:
        avail.append("baseline")
    if _STATE["bert_model"] is not None:
        avail.append("bert")
    if _STATE["roberta_model"] is not None:
        avail.append("roberta")
    return avail


def predict_baseline(text: str):
    vec = _STATE["baseline_vectorizer"].transform([text])
    # LinearSVC has no predict_proba by default; use decision_function -> softmax as a proxy.
    scores = _STATE["baseline_svm"].decision_function(vec)[0]
    exp = np.exp(scores - np.max(scores))
    probs = exp / exp.sum()
    label_idx = int(np.argmax(probs))
    return LABELS[label_idx], float(probs[label_idx]), {l: float(p) for l, p in zip(LABELS, probs)}


def predict_transformer(text: str, which: str = "bert", return_attention: bool = False):
    import torch

    tokenizer = _STATE[f"{which}_tokenizer"]
    model = _STATE[f"{which}_model"]
    enc = tokenizer(text, truncation=True, padding=True, max_length=MAX_SEQ_LEN, return_tensors="pt").to(_STATE["device"])
    with torch.no_grad():
        out = model(**enc)
    probs = torch.softmax(out.logits, dim=-1)[0].cpu().numpy()
    label_idx = int(np.argmax(probs))

    result = (LABELS[label_idx], float(probs[label_idx]), {l: float(p) for l, p in zip(LABELS, probs)})

    if return_attention:
        # Average last-layer attention over heads, for the [CLS] token's row.
        attentions = out.attentions[-1][0]  # (heads, seq, seq)
        cls_attention = attentions.mean(dim=0)[0].cpu().numpy()  # avg over heads, CLS row
        tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0].cpu().numpy())
        return result, tokens, cls_attention.tolist()

    return result


def predict_ensemble(text: str):
    bert_result = predict_transformer(text, "bert") if _STATE["bert_model"] else None
    roberta_result = predict_transformer(text, "roberta") if _STATE["roberta_model"] else None

    if bert_result and roberta_result:
        avg_probs = {
            l: (bert_result[2][l] + roberta_result[2][l]) / 2 for l in LABELS
        }
        label = max(avg_probs, key=avg_probs.get)
        return label, avg_probs[label], avg_probs
    return bert_result or roberta_result


def get_leaderboard():
    entries = []
    metric_files = glob.glob(os.path.join(MODELS_DIR, "**", "metrics.json"), recursive=True)
    metric_files += glob.glob(os.path.join(MODELS_DIR, "evaluation", "summary.json"))
    for path in metric_files:
        try:
            with open(path) as f:
                data = json.load(f)
            model_name = os.path.basename(os.path.dirname(path))
            entries.append({
                "model": model_name,
                "accuracy": data.get("accuracy") or data.get("ensemble_acc"),
                "f1_weighted": data.get("f1_weighted"),
            })
        except (json.JSONDecodeError, OSError):
            continue
    return entries
