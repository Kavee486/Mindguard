"""
Calls the local Ollama server to generate an empathetic response. Falls back
to a safe canned response if Ollama isn't running or the fine-tuned model
hasn't been served yet, so the rest of the API keeps working.
"""
import os
import sys

import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from config import OLLAMA_HOST, RESPONSE_MODEL_OLLAMA, PROMPTING_MODEL  # noqa: E402

SYSTEM_PROMPT = (
    "You are a warm, supportive listener responding to someone who may be struggling "
    "with a mental-health-related concern. Be empathetic and validating. Offer gentle, "
    "practical coping suggestions where relevant. Never diagnose a condition. Always "
    "encourage the person to seek help from a qualified professional if appropriate. "
    "Keep the response to 3-5 sentences."
)

FALLBACK_RESPONSE = (
    "Thank you for sharing that - it sounds like you're going through a lot right now. "
    "Your feelings are valid, and you don't have to navigate this alone. Consider "
    "reaching out to a mental health professional who can support you properly; if "
    "you're in crisis, please contact a local crisis line or emergency services."
)


def generate_response(user_text: str, predicted_label: str) -> str:
    model_name = RESPONSE_MODEL_OLLAMA

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"(For context, this message was auto-classified as potentially related to: {predicted_label}. "
        f"Do not mention this classification in your reply.)\n\n"
        f"User message: \"{user_text[:1500]}\"\n\nYour supportive response:"
    )

    for model in [model_name, PROMPTING_MODEL]:
        try:
            resp = requests.post(
                f"{OLLAMA_HOST}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.7}},
                timeout=60,
            )
            if resp.status_code == 200:
                text = resp.json().get("response", "").strip()
                if text:
                    return text
        except requests.exceptions.RequestException:
            continue

    return FALLBACK_RESPONSE
