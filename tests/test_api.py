"""
tests/test_api.py – CI test suite for the banana ripeness FastAPI app.

Tests run against the FastAPI TestClient (no real model weights required)
by patching the model-loading function so that a tiny random model is used.
"""

import io
import types
from unittest.mock import patch, MagicMock

import pytest
import torch
import torch.nn as nn
from PIL import Image
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Minimal stub model (random weights, correct output shape)
# ---------------------------------------------------------------------------

class _StubModel(nn.Module):
    def __init__(self, num_classes: int = 6):
        super().__init__()
        self.fc = nn.Linear(3 * 224 * 224, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.view(x.size(0), -1))


def _make_stub_model() -> nn.Module:
    m = _StubModel()
    m.eval()
    return m


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """Return a TestClient with the model replaced by a stub."""
    with patch("main.get_model", return_value=_make_stub_model()):
        from main import app
        with TestClient(app) as c:
            yield c


def _make_image_bytes(fmt: str = "JPEG") -> bytes:
    """Return raw bytes of a tiny synthetic image."""
    img = Image.new("RGB", (64, 64), color=(200, 180, 50))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestIndex:
    def test_index_returns_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]


class TestPredict:
    VALID_CLASSES = {"freshripe", "freshunripe", "overripe", "ripe", "rotten", "unripe"}

    def test_predict_jpeg(self, client):
        img_bytes = _make_image_bytes("JPEG")
        resp = client.post(
            "/predict",
            files={"file": ("banana.jpg", img_bytes, "image/jpeg")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["predicted_class"] in self.VALID_CLASSES
        assert 0.0 <= data["confidence"] <= 1.0
        assert set(data["all_probabilities"].keys()) == self.VALID_CLASSES
        total = sum(data["all_probabilities"].values())
        assert abs(total - 1.0) < 1e-3

    def test_predict_png(self, client):
        img_bytes = _make_image_bytes("PNG")
        resp = client.post(
            "/predict",
            files={"file": ("banana.png", img_bytes, "image/png")},
        )
        assert resp.status_code == 200
        assert resp.json()["predicted_class"] in self.VALID_CLASSES

    def test_predict_non_image_rejected(self, client):
        resp = client.post(
            "/predict",
            files={"file": ("data.csv", b"col1,col2\n1,2\n", "text/csv")},
        )
        assert resp.status_code == 400

    def test_predict_corrupt_image_rejected(self, client):
        resp = client.post(
            "/predict",
            files={"file": ("bad.jpg", b"not-an-image", "image/jpeg")},
        )
        assert resp.status_code == 400

    def test_predict_no_file_returns_422(self, client):
        resp = client.post("/predict")
        assert resp.status_code == 422
