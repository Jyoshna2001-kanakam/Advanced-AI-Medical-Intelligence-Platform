# 🩺 Advanced AI Medical Intelligence Platform

An end-to-end AI application that analyzes chest X-ray images, predicts disease
(Pneumonia vs Normal) using a deep learning model, explains its predictions
with **Grad-CAM**, generates an **LLM-assisted draft medical report**, exposes
everything through a **REST API**, stores prediction history in a
**database**, and ships with a simple **web UI** and **Docker** deployment.

> ⚠️ **Disclaimer:** This is an educational / portfolio project. It is **not**
> a certified medical device and must never be used for real clinical
> diagnosis or treatment decisions without review by a licensed physician.

---

## 1. Project Objective

Build a complete AI application capable of:
- Analyzing medical images
- Predicting disease using Deep Learning
- Explaining predictions using Explainable AI (Grad-CAM)
- Generating AI-assisted medical reports using an LLM
- Providing REST APIs
- Storing prediction history in a database
- Deploying with a user-friendly interface

## 2. Architecture

```
┌──────────────┐     upload image      ┌────────────────────┐
│  Web Frontend │ ───────────────────▶ │   FastAPI Backend   │
│ (HTML/JS UI)  │ ◀─────────────────── │  (REST API layer)   │
└──────────────┘   JSON + Grad-CAM PNG  └─────────┬──────────┘
                                                   │
                     ┌─────────────────────────────┼─────────────────────────┐
                     ▼                             ▼                         ▼
           ┌──────────────────┐          ┌──────────────────┐     ┌──────────────────┐
           │ DenseNet-121 CNN │          │     Grad-CAM      │     │  SQLite / Postgres│
           │ (Disease predict)│──────────▶  Explainability   │     │  Prediction History│
           └──────────────────┘  feature  └──────────────────┘     └──────────────────┘
                                    maps
                     │
                     ▼
           ┌──────────────────┐
           │  Anthropic Claude │
           │  (LLM Report Gen) │
           └──────────────────┘
```

**Design principle:** the CNN is the *only* component that makes the medical
classification — deterministic, auditable, and explainable via Grad-CAM. The
LLM is used strictly for **report writing/communication** from the CNN's
already-computed structured output, never for diagnosis from raw pixels. This
keeps diagnostic responsibility with the validated vision model.

## 3. Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning | PyTorch, TorchVision (DenseNet-121, transfer learning) |
| Explainable AI | Custom Grad-CAM implementation (OpenCV for heatmap overlay) |
| LLM Integration | Anthropic Claude API (`anthropic` Python SDK) |
| API | FastAPI + Uvicorn |
| Database | SQLAlchemy ORM, SQLite (dev) / PostgreSQL (prod, one env var swap) |
| Frontend | HTML5 / CSS / vanilla JavaScript |
| Deployment | Docker, docker-compose |
| Testing | Pytest, httpx |

## 4. Repository Structure

```
medical-ai-platform/
├── app/
│   ├── main.py              # FastAPI app & REST endpoints
│   ├── database.py          # SQLAlchemy engine/session
│   ├── models_db.py         # ORM model (PredictionRecord)
│   ├── schemas.py           # Pydantic request/response schemas
│   ├── inference.py         # Model loading + prediction wrapper
│   ├── gradcam.py           # Grad-CAM implementation
│   ├── llm_report.py        # Claude-based report generation (+ fallback)
│   └── static/index.html    # Web UI
├── ml/
│   ├── model_def.py         # DenseNet-121 architecture definition
│   ├── train.py             # Training script
│   └── generate_synthetic_data.py  # Demo dataset generator
├── models/
│   └── pneumonia_densenet121.pth   # Trained model checkpoint
├── tests/
│   └── test_api.py          # API smoke tests (pytest)
├── docs/
│   └── PROJECT_REPORT.pdf   # Full written project report
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

## 5. Dataset

This repository ships with a **synthetic chest-X-ray-style dataset**
(`ml/generate_synthetic_data.py`) so the entire pipeline — training,
inference, Grad-CAM, API, database — is runnable immediately without any
external download, and so the shipped `models/pneumonia_densenet121.pth`
checkpoint is a genuinely trained artifact rather than a placeholder.

**For a production-grade model**, replace the contents of `data/train/` and
`data/val/` with the public **["Chest X-Ray Images (Pneumonia)"](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)**
dataset (Kermany et al.), keeping the same folder layout:

```
data/
  train/
    NORMAL/
    PNEUMONIA/
  val/
    NORMAL/
    PNEUMONIA/
```

No other code changes are required — `train.py` uses `torchvision.datasets.ImageFolder`,
which works identically with real or synthetic data.

## 6. Model

- **Architecture:** DenseNet-121, ImageNet-pretrained backbone (when network
  access allows the weight download; otherwise trains from a random init with
  a clear console warning — see `ml/model_def.py`), classifier head replaced
  with `Dropout(0.3) → Linear(num_features, 2)`.
- **Why DenseNet-121:** it is the backbone used by CheXNet-style chest X-ray
  models in the medical imaging literature, and its dense connectivity gives
  strong, well-localized final-layer feature maps — important for Grad-CAM
  quality.
- **Training:** `ml/train.py`, Adam optimizer, cross-entropy loss, saves the
  best validation-accuracy checkpoint plus a `training_history.json`.

```bash
python ml/generate_synthetic_data.py     # or plug in the real dataset
python ml/train.py --epochs 5 --batch-size 16 --out models/pneumonia_densenet121.pth
```

Demo run in this repo (synthetic data, 3 epochs, CPU):

```
[epoch 1/3] train_loss=0.3989 train_acc=0.9000 | val_loss=0.2218 val_acc=1.0000
[epoch 2/3] train_loss=0.0998 train_acc=0.9800 | val_loss=0.0021 val_acc=1.0000
[epoch 3/3] train_loss=0.0850 train_acc=0.9700 | val_loss=0.0001 val_acc=1.0000
[done] Best val accuracy: 1.0000
```

## 7. Explainable AI (Grad-CAM)

`app/gradcam.py` implements Grad-CAM from scratch (Selvaraju et al., ICCV
2017): forward + gradient hooks on the last convolutional layer
(`features.norm5`), gradient-weighted channel importance, ReLU, normalize,
resize to the input resolution, and overlay as a JET colormap heatmap. The
API returns this heatmap alongside every prediction so the class output is
never a black box.

## 8. LLM Integration

`app/llm_report.py` calls the Anthropic Claude API with the CNN's structured
output (predicted class, confidence, class probability distribution, and an
optional clinician-supplied note) and asks it to produce a structured,
appropriately-hedged draft report (Findings → Region of Interest → Clinical
Interpretation → Recommended Next Steps → mandatory Disclaimer). If
`ANTHROPIC_API_KEY` is not set (or the call fails), the endpoint falls back to
a deterministic template so the API never breaks in a demo/offline
environment.

## 9. REST API

Interactive OpenAPI docs are auto-served at **`/docs`** once running.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/v1/health` | Service + model load status |
| POST | `/api/v1/predict` | Upload an image → prediction + Grad-CAM |
| POST | `/api/v1/predict/{id}/report` | Generate/attach an LLM report to a stored prediction |
| GET | `/api/v1/history?limit=&offset=` | Paginated prediction history |
| GET | `/api/v1/history/{id}` | Single history record |
| DELETE | `/api/v1/history/{id}` | Delete a history record |
| GET | `/gradcam/{filename}` | Serve a saved Grad-CAM overlay image |

Example:
```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -F "file=@chest_xray.png;type=image/png"

curl -X POST http://localhost:8000/api/v1/predict/1/report \
  -H "Content-Type: application/json" \
  -d '{"patient_note": "45-year-old, cough and fever for 3 days"}'
```

## 10. Database

`prediction_history` table (SQLAlchemy, see `app/models_db.py`): id, original
filename, predicted class, confidence, full class probability JSON, Grad-CAM
filename, LLM report text, optional patient note, timestamp. Defaults to
SQLite (zero config); set `DATABASE_URL` to a PostgreSQL DSN for production —
no other code changes needed.

## 11. Running Locally

```bash
git clone <your-repo-url>
cd medical-ai-platform
pip install -r requirements.txt

# 1. Generate demo data and train (or plug in the real dataset first)
python ml/generate_synthetic_data.py
python ml/train.py --epochs 5 --out models/pneumonia_densenet121.pth

# 2. (optional) enable real LLM reports
cp .env.example .env   # then set ANTHROPIC_API_KEY

# 3. Run the API + web UI
cd app
uvicorn main:app --reload --port 8000
```
Open **http://localhost:8000** for the web UI, or **http://localhost:8000/docs** for the API docs.

## 12. Running with Docker

```bash
docker compose up --build
```
This builds the image, installs dependencies, and serves the API + UI on
`http://localhost:8000`. Set `ANTHROPIC_API_KEY` in your shell or a `.env`
file before running for live LLM reports (otherwise the template fallback is
used).

## 13. Testing

```bash
pytest tests/ -v
```

## 14. Evaluation Criteria Mapping

| Criteria | Where |
|---|---|
| DL Model Performance | `ml/train.py`, `models/training_history.json` |
| Code Quality / Structure | modular `app/` + `ml/` separation, typed schemas |
| Explainable AI | `app/gradcam.py` |
| LLM Integration | `app/llm_report.py` |
| API Development | `app/main.py` |
| Database Design | `app/database.py`, `app/models_db.py` |
| Web Application | `app/static/index.html` |
| Documentation | this README + `docs/PROJECT_REPORT.pdf` |
| Deployment | `Dockerfile`, `docker-compose.yml` |

## 15. Limitations & Future Work

- Ships with a synthetic demo dataset; swap in the real Kaggle Chest X-Ray
  dataset (and more epochs + GPU training) for production-grade accuracy.
- Binary classification only (Normal / Pneumonia); could extend to
  multi-label chest pathology classification (e.g. NIH ChestX-ray14 classes).
- No authentication/authorization layer yet — add OAuth2/JWT before any
  multi-user or public deployment.
- No PACS/DICOM support yet — currently accepts PNG/JPEG only.

## 16. License / Attribution

Educational project. DenseNet-121 architecture: Huang et al., "Densely
Connected Convolutional Networks", CVPR 2017. Grad-CAM: Selvaraju et al.,
ICCV 2017.
