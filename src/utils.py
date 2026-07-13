"""
Shared text-cleaning and small helper functions used by every stage.
"""
import os
import re
import unicodedata


def clean_text(text: str) -> str:
    """Lower-case, strip URLs/mentions/Reddit markup, normalise whitespace.

    This is intentionally simple and shared across every script so that the
    baseline, the transformers, and the LLM prompts all see text cleaned the
    same way.
    """
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.lower()

    # Remove Reddit's own placeholders for missing content.
    text = re.sub(r"\[deleted\]|\[removed\]", " ", text)

    # Strip URLs.
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Strip Reddit user/subreddit mentions (u/name, r/name).
    text = re.sub(r"\b[ur]/\w+", " ", text)

    # Strip basic markdown artefacts (*, _, #, `, >).
    text = re.sub(r"[*_#`>]", " ", text)

    # Collapse whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    return text


def is_substantive(text: str, min_words: int = 3) -> bool:
    """Filter out empty / deleted / too-short posts."""
    if not text:
        return False
    return len(text.split()) >= min_words


def find_latest_checkpoint(models_dir: str, candidates=("bert-base-uncased", "roberta-base")) -> str:
    """Find the most recently modified trained-model checkpoint under models_dir.

    Looks for `<models_dir>/<candidate>/final` directories (the layout used
    by train_transformer.py) and returns the most recently modified one.
    Raises FileNotFoundError with a helpful message if none exist yet.
    """
    found = []
    for name in candidates:
        checkpoint_dir = os.path.join(models_dir, name, "final")
        if os.path.isdir(checkpoint_dir):
            found.append(checkpoint_dir)

    if not found:
        searched = ", ".join(os.path.join(models_dir, name, "final") for name in candidates)
        raise FileNotFoundError(
            "No trained checkpoint found to auto-select. Searched: "
            f"{searched}. Run src/train_transformer.py first, or pass --model_dir explicitly."
        )

    found.sort(key=os.path.getmtime, reverse=True)
    return found[0]
