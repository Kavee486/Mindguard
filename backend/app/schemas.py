from typing import Dict, List, Optional
from pydantic import BaseModel


class AnalyzeRequest(BaseModel):
    text: str
    generate_response: bool = True


class ModelPrediction(BaseModel):
    label: str
    confidence: float
    probabilities: Dict[str, float]


class AnalyzeResponse(BaseModel):
    text: str
    prediction: ModelPrediction
    model_used: str
    response: Optional[str] = None
    disclaimer: str = (
        "MindGuard is a research/teaching demo, not a medical device. It does not "
        "diagnose. If you are in crisis, please contact a local crisis line or "
        "emergency services."
    )


class CompareResponse(BaseModel):
    text: str
    predictions: Dict[str, ModelPrediction]


class InterpretResponse(BaseModel):
    text: str
    tokens: List[str]
    attention_weights: List[float]


class LeaderboardEntry(BaseModel):
    model: str
    accuracy: Optional[float] = None
    f1_weighted: Optional[float] = None


class LeaderboardResponse(BaseModel):
    entries: List[LeaderboardEntry]
