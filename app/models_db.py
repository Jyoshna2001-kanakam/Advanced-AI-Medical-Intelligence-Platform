"""
models_db.py
------------
ORM schema for prediction history storage.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON
from database import Base


class PredictionRecord(Base):
    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True)
    original_filename = Column(String(255), nullable=True)
    predicted_class = Column(String(64), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    class_probabilities = Column(JSON, nullable=True)
    gradcam_filename = Column(String(255), nullable=True)
    llm_report = Column(Text, nullable=True)
    patient_note = Column(Text, nullable=True)  # optional free-text clinical context supplied by the caller
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def to_dict(self):
        return {
            "id": self.id,
            "original_filename": self.original_filename,
            "predicted_class": self.predicted_class,
            "confidence": round(self.confidence, 4),
            "class_probabilities": self.class_probabilities,
            "gradcam_filename": self.gradcam_filename,
            "llm_report": self.llm_report,
            "patient_note": self.patient_note,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
