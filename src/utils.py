"""
Shared text-cleaning and small helper functions used by every stage.
"""
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
