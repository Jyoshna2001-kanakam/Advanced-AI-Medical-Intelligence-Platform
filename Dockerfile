FROM python:3.11-slim

# System deps needed by opencv-python-headless / torchvision image decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY ml/ ./ml/
COPY models/ ./models/

ENV MODEL_PATH=/app/models/pneumonia_densenet121.pth \
    GRADCAM_DIR=/app/gradcam_outputs \
    DATABASE_URL=sqlite:////app/data/medical_ai.db \
    PYTHONUNBUFFERED=1

RUN mkdir -p /app/gradcam_outputs /app/data

EXPOSE 8000

WORKDIR /app/app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
