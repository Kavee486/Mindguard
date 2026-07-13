"""
MindGuard FastAPI backend.

Run locally:
    uvicorn backend.app.main:app --reload --port 8000

Run via Docker: see docker-compose.yml / README.md.

Endpoints:
    GET  /health        - liveness + which models are loaded
    POST /analyze        - classify text with the best available model + generate a response
    POST /compare         - classify with every available model side by side
    POST /interpret        - BERT attention weights for a piece of text
    GET  /leaderboard       - reported metrics from every trained model
"""
import os
import sys

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from schemas import (
    AnalyzeRequest, AnalyzeResponse, ModelPrediction,
    CompareResponse, InterpretResponse, LeaderboardResponse, LeaderboardEntry,
)
import model_loader
import llm

app = FastAPI(title="MindGuard API", version="0.1.0")

# Comma-separated list of allowed frontend origins, e.g.
# CORS_ALLOWED_ORIGINS="http://localhost:3000,https://mindguard.example.com"
# Defaults to the local Next.js dev server so `docker-compose up` keeps working out of the box.
_cors_origins = os.environ.get("CORS_ALLOWED_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_origins if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    model_loader.load_all()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": model_loader.available_models(),
    }


def _best_available_prediction(text: str):
    avail = model_loader.available_models()
    if "bert" in avail and "roberta" in avail:
        label, conf, probs = model_loader.predict_ensemble(text)
        return label, conf, probs, "ensemble(bert+roberta)"
    if "bert" in avail:
        label, conf, probs = model_loader.predict_transformer(text, "bert")
        return label, conf, probs, "bert"
    if "roberta" in avail:
        label, conf, probs = model_loader.predict_transformer(text, "roberta")
        return label, conf, probs, "roberta"
    if "baseline" in avail:
        label, conf, probs = model_loader.predict_baseline(text)
        return label, conf, probs, "baseline(tfidf+svm)"
    return None


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="`text` must not be empty")

    result = _best_available_prediction(req.text)
    if result is None:
        raise HTTPException(
            status_code=503,
            detail="No trained models found. Run src/baseline_tfidf.py (and/or "
                   "src/train_transformer.py) first, then restart the API.",
        )
    label, conf, probs, model_used = result

    response_text = None
    if req.generate_response:
        response_text = llm.generate_response(req.text, label)

    return AnalyzeResponse(
        text=req.text,
        prediction=ModelPrediction(label=label, confidence=conf, probabilities=probs),
        model_used=model_used,
        response=response_text,
    )


@app.post("/compare", response_model=CompareResponse)
def compare(req: AnalyzeRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="`text` must not be empty")

    predictions = {}
    avail = model_loader.available_models()

    if "baseline" in avail:
        label, conf, probs = model_loader.predict_baseline(req.text)
        predictions["baseline"] = ModelPrediction(label=label, confidence=conf, probabilities=probs)
    if "bert" in avail:
        label, conf, probs = model_loader.predict_transformer(req.text, "bert")
        predictions["bert"] = ModelPrediction(label=label, confidence=conf, probabilities=probs)
    if "roberta" in avail:
        label, conf, probs = model_loader.predict_transformer(req.text, "roberta")
        predictions["roberta"] = ModelPrediction(label=label, confidence=conf, probabilities=probs)
    if "bert" in avail and "roberta" in avail:
        label, conf, probs = model_loader.predict_ensemble(req.text)
        predictions["ensemble"] = ModelPrediction(label=label, confidence=conf, probabilities=probs)

    if not predictions:
        raise HTTPException(status_code=503, detail="No trained models found.")

    return CompareResponse(text=req.text, predictions=predictions)


@app.post("/interpret", response_model=InterpretResponse)
def interpret(req: AnalyzeRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="`text` must not be empty")

    if "bert" not in model_loader.available_models():
        raise HTTPException(
            status_code=503,
            detail="BERT model not loaded - run src/train_transformer.py --model bert-base-uncased first.",
        )
    (label, conf, probs), tokens, weights = model_loader.predict_transformer(
        req.text, "bert", return_attention=True
    )
    return InterpretResponse(text=req.text, tokens=tokens, attention_weights=weights)


@app.get("/leaderboard", response_model=LeaderboardResponse)
def leaderboard():
    entries = model_loader.get_leaderboard()
    return LeaderboardResponse(entries=[LeaderboardEntry(**e) for e in entries])
