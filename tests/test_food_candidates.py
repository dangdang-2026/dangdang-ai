import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import settings
from app.food.detector import Box
from app.food.predictor import Candidate, open_image
from app.food.router import get_detector, get_predictor
from app.main import app

TOKEN = "test-token"


WHOLE = [
    Candidate(name="김밥", confidence=0.9),
    Candidate(name="비빔밥", confidence=0.06),
    Candidate(name="김치볶음밥", confidence=0.02),
]
LEFT = Box(x=0.0, y=0.0, width=0.5, height=1.0)
RIGHT = Box(x=0.5, y=0.0, width=0.5, height=1.0)


class FakePredictor:
    """진짜 모델 없이 주소만 시험한다 (CI 에는 모델 파일이 없다).
    사진 전체(300×200)면 김밥, 잘린 사진이면 가운데 색으로 음식을 정한다."""

    def predict(self, image_bytes: bytes) -> list[Candidate]:
        return self.predict_image(open_image(image_bytes))

    def predict_image(self, image: Image.Image) -> list[Candidate]:
        if image.size == (300, 200):
            return WHOLE
        red, _, blue = image.getpixel((image.width // 2, image.height // 2))
        name = "김치찌개" if red > blue else "미역국"
        return [Candidate(name=name, confidence=0.8)]


class FakeDetector:
    def __init__(self, boxes: list[Box]) -> None:
        self.boxes = boxes

    def detect(self, image: Image.Image) -> list[Box]:
        return self.boxes


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "internal_token", TOKEN)
    app.dependency_overrides[get_predictor] = FakePredictor
    app.dependency_overrides[get_detector] = lambda: None
    yield TestClient(app)
    app.dependency_overrides.clear()


def use_detector(boxes: list[Box]) -> None:
    app.dependency_overrides[get_detector] = lambda: FakeDetector(boxes)


def jpeg_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (300, 200), "orange").save(buffer, format="JPEG")
    return buffer.getvalue()


def table_bytes() -> bytes:
    """왼쪽 반은 빨강(김치찌개), 오른쪽 반은 파랑(미역국)"""
    image = Image.new("RGB", (300, 200), "red")
    image.paste((0, 0, 255), (150, 0, 300, 200))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def post(client, data: bytes):
    return client.post(
        "/internal/food-candidates",
        headers={"X-Internal-Token": TOKEN},
        files={"image": ("meal.png", data, "image/png")},
    )


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


def test_without_detector_whole_photo_is_one_item(client):
    body = post(client, jpeg_bytes()).json()

    assert body["items"] == [{"box": None, "candidates": body["candidates"]}]


def test_table_photo_returns_item_per_dish(client):
    use_detector([LEFT, RIGHT])

    body = post(client, table_bytes()).json()

    assert body["candidates"][0]["name"] == "김밥"  # 예전 칸은 사진 전체 그대로
    assert [item["box"] for item in body["items"]] == [LEFT.model_dump(), RIGHT.model_dump()]
    assert [item["candidates"][0]["name"] for item in body["items"]] == ["김치찌개", "미역국"]


def test_one_dish_keeps_box_and_uses_whole_photo_candidates(client):
    use_detector([LEFT])

    body = post(client, table_bytes()).json()

    assert body["items"] == [{"box": LEFT.model_dump(), "candidates": body["candidates"]}]


def test_no_dish_found_is_one_item_without_box(client):
    use_detector([])

    body = post(client, table_bytes()).json()

    assert body["items"] == [{"box": None, "candidates": body["candidates"]}]


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
