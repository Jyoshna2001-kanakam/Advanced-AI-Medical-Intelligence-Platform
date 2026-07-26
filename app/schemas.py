"""
schemas.py
----------
Pydantic request/response models.
"""

from typing import Optional, Dict
from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    id: int
    predicted_class: str
    confidence: float
    class_probabilities: Dict[str, float]
    gradcam_url: str
    created_at: str


class ReportRequest(BaseModel):
    patient_note: Optional[str] = Field(
        default=None,
        description="Optional free-text clinical context (age, symptoms, history) to inform the AI report."
    )


class ReportResponse(BaseModel):
    id: int
    predicted_class: str
    confidence: float
    llm_report: str


class HistoryItem(BaseModel):
    id: int
    original_filename: Optional[str]
    predicted_class: str
    confidence: float
    class_probabilities: Optional[Dict[str, float]]
    gradcam_filename: Optional[str]
    llm_report: Optional[str]
    patient_note: Optional[str]
    created_at: Optional[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
