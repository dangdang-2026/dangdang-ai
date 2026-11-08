import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import settings
from app.food.predictor import Candidate, NotAnImageError
from app.food.router import get_predictor
from app.main import app

TOKEN = "test-token"


class FakePredictor:
    """진짜 모델 없이 주소만 시험한다 (CI 에는 모델 파일이 없다)."""

    def predict(self, image_bytes: bytes) -> list[Candidate]:
        try:
            Image.open(io.BytesIO(image_bytes)).load()
        except OSError as e:
            raise NotAnImageError from e
        return [
            Candidate(name="김밥", confidence=0.9),
            Candidate(name="비빔밥", confidence=0.06),
            Candidate(name="김치볶음밥", confidence=0.02),
        ]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "internal_token", TOKEN)
    app.dependency_overrides[get_predictor] = FakePredictor
    yield TestClient(app)
    app.dependency_overrides.clear()


def jpeg_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (300, 200), "orange").save(buffer, format="JPEG")
    return buffer.getvalue()


def test_photo_returns_three_candidates(client):
    response = client.post(
        "/internal/food-candidates",
        headers={"X-Internal-Token": TOKEN},
        files={"image": ("meal.jpg", jpeg_bytes(), "image/jpeg")},
    )

    assert response.status_code == 200
    candidates = response.json()["candidates"]
    assert len(candidates) == 3
    assert candidates[0] == {"name": "김밥", "confidence": 0.9}


def test_not_a_photo_returns_400(client):
    response = client.post(
        "/internal/food-candidates",
        headers={"X-Internal-Token": TOKEN},
        files={"image": ("memo.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 400


def test_missing_token_returns_401(client):
    response = client.post(
        "/internal/food-candidates",
        files={"image": ("meal.jpg", jpeg_bytes(), "image/jpeg")},
    )

    assert response.status_code == 401
