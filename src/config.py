"""
Shared configuration for the MindGuard pipeline.
Edit paths here if you organise your data/models differently.
"""
import os

# ---- Paths ----
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

os.makedirs(DATA_RAW_DIR, exist_ok=True)
os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ---- Labels ----
# Fixed label encoding used throughout the project (must stay consistent
# across every script, or your saved models will predict the wrong names).
LABELS = ["ADHD", "OCD", "aspergers", "depression", "ptsd"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}
NUM_LABELS = len(LABELS)

# ---- HuggingFace dataset identifiers ----
# Both are public, no login/API token required.
CLASSIFICATION_DATASET = "solomonk/reddit_mental_health_posts"
COUNSELING_DATASET = "Amod/mental_health_counseling_conversations"

# ---- Model checkpoints ----
BERT_CHECKPOINT = "bert-base-uncased"
ROBERTA_CHECKPOINT = "roberta-base"
# Smaller instruct model as the default for response fine-tuning so it fits
# on most consumer GPUs. Swap for "mistralai/Mistral-7B-Instruct-v0.2" if you
# have 16GB+ VRAM (or are training via Colab).
RESPONSE_BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
RESPONSE_BASE_MODEL_SMALL = "microsoft/Phi-3-mini-4k-instruct"

# ---- Ollama (local LLM serving) ----
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
PROMPTING_MODEL = os.environ.get("PROMPTING_MODEL", "qwen2.5:7b")
RESPONSE_MODEL_OLLAMA = os.environ.get("RESPONSE_MODEL_OLLAMA", "mindguard-responder")

# ---- Misc ----
RANDOM_SEED = 42
MAX_SEQ_LEN = 256
