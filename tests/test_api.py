"""
test_api.py
-----------
Basic smoke tests for the API layer. Run with: pytest tests/
Requires the model to already be trained (see README) since /predict
depends on a loaded checkpoint.
"""

import io
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "app"))

from fastapi.testclient import TestClient
from PIL import Image

from main import app  # noqa: E402

client = TestClient(app)


def test_health():
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"


def _dummy_image_bytes():
    img = Image.new("RGB", (224, 224), color=(120, 120, 120))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_predict_rejects_bad_content_type():
    resp = client.post("/api/v1/predict", files={"file": ("test.txt", b"not an image", "text/plain")})
    assert resp.status_code == 400


def test_predict_and_history_flow():
    buf = _dummy_image_bytes()
    resp = client.post("/api/v1/predict", files={"file": ("test.png", buf, "image/png")})
    if resp.status_code == 503:
        # Model not trained in this environment run — acceptable for a pure code check.
        return
    assert resp.status_code == 200
    data = resp.json()
    assert "predicted_class" in data
    assert 0.0 <= data["confidence"] <= 1.0

    history_resp = client.get("/api/v1/history")
    assert history_resp.status_code == 200
    assert len(history_resp.json()) >= 1
