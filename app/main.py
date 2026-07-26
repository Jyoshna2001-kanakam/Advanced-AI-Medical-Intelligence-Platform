"""
main.py
-------
Advanced AI Medical Intelligence Platform - FastAPI backend.

Endpoints:
    GET  /api/v1/health                      -> service/model health check
    POST /api/v1/predict                     -> upload image, get prediction + Grad-CAM
    POST /api/v1/predict/{id}/report         -> generate/attach an LLM report for a stored prediction
    GET  /api/v1/history                     -> paginated prediction history
    GET  /api/v1/history/{id}                -> single history record
    DELETE /api/v1/history/{id}               -> delete a history record
    GET  /gradcam/{filename}                 -> serve a saved Grad-CAM overlay image
    GET  /                                    -> web UI (static/index.html)
"""

import os
import sys
import io

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from PIL import Image
import torch

sys.path.append(os.path.dirname(__file__))
from database import Base, engine, get_db  # noqa: E402
from models_db import PredictionRecord  # noqa: E402
from schemas import (PredictionResponse, ReportRequest, ReportResponse,  # noqa: E402
                      HistoryItem, HealthResponse)
import inference  # noqa: E402
import llm_report  # noqa: E402

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Advanced AI Medical Intelligence Platform",
    description="Deep learning disease prediction, Grad-CAM explainability, "
                "and LLM-assisted medical reporting via REST API.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GRADCAM_DIR = os.environ.get("GRADCAM_DIR", "gradcam_outputs")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(GRADCAM_DIR, exist_ok=True)

ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


@app.get("/api/v1/health", response_model=HealthResponse)
def health():
    model_loaded = os.path.exists(inference.MODEL_PATH)
    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}. "
                                                      f"Upload a PNG or JPEG image.")
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 10MB).")

    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except Exception:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    try:
        result = inference.predict_and_explain(image)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    record = PredictionRecord(
        original_filename=file.filename,
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        class_probabilities=result["class_probabilities"],
        gradcam_filename=result["gradcam_filename"],
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return PredictionResponse(
        id=record.id,
        predicted_class=record.predicted_class,
        confidence=record.confidence,
        class_probabilities=record.class_probabilities,
        gradcam_url=f"/gradcam/{record.gradcam_filename}",
        created_at=record.created_at.isoformat(),
    )


@app.post("/api/v1/predict/{record_id}/report", response_model=ReportResponse)
def generate_report(record_id: int, payload: ReportRequest, db: Session = Depends(get_db)):
    record = db.query(PredictionRecord).filter(PredictionRecord.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Prediction record not found.")

    report_text = llm_report.generate_report(
        predicted_class=record.predicted_class,
        confidence=record.confidence,
        class_probabilities=record.class_probabilities or {},
        patient_note=payload.patient_note,
    )
    record.llm_report = report_text
    record.patient_note = payload.patient_note
    db.commit()
    db.refresh(record)

    return ReportResponse(
        id=record.id,
        predicted_class=record.predicted_class,
        confidence=record.confidence,
        llm_report=record.llm_report,
    )


@app.get("/api/v1/history", response_model=list[HistoryItem])
def get_history(limit: int = Query(20, le=200), offset: int = Query(0, ge=0),
                db: Session = Depends(get_db)):
    records = (db.query(PredictionRecord)
               .order_by(PredictionRecord.created_at.desc())
               .offset(offset).limit(limit).all())
    return [HistoryItem(**r.to_dict()) for r in records]


@app.get("/api/v1/history/{record_id}", response_model=HistoryItem)
def get_history_item(record_id: int, db: Session = Depends(get_db)):
    record = db.query(PredictionRecord).filter(PredictionRecord.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Prediction record not found.")
    return HistoryItem(**record.to_dict())


@app.delete("/api/v1/history/{record_id}")
def delete_history_item(record_id: int, db: Session = Depends(get_db)):
    record = db.query(PredictionRecord).filter(PredictionRecord.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Prediction record not found.")
    db.delete(record)
    db.commit()
    return {"deleted": record_id}


@app.get("/gradcam/{filename}")
def get_gradcam_image(filename: str):
    path = os.path.join(GRADCAM_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Grad-CAM image not found.")
    return FileResponse(path, media_type="image/png")


# Serve the frontend last so it doesn't shadow API routes
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
