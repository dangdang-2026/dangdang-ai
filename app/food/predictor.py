"""food.onnx 를 불러 두고 사진 1장 → 후보 3개를 냄"""

import io
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, computed_field

from app.food.db_names import search_name
from app.food.preprocess import preprocess

TOP_K = 3


class Candidate(BaseModel):
    name: str  # AI Hub 이름 (화면 칩에 보이는 이름)
    confidence: float  # 0~1, 3개를 더해도 1이 안 될 수 있다(나머지 음식 몫)

    # 식약처 음식 DB에서 찾을 이름. 이름 꼴이 다르면 바꿔 준다 (계란찜 → 달걀찜, #44)
    # 백엔드는 이 이름으로 검색한다
    @computed_field(alias="searchName")
    @property
    def search_name(self) -> str:
        return search_name(self.name)


class NotAnImageError(Exception):
    pass


def open_image(image_bytes: bytes) -> Image.Image:
    """디스크에 저장하지 않고 메모리에서만 연다. 폰 사진 회전을 바로잡아 화면에 보이는 방향으로"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except (UnidentifiedImageError, OSError) as e:
        raise NotAnImageError from e
    return ImageOps.exif_transpose(image).convert("RGB")


class FoodPredictor:
    def __init__(self, model_path: Path, labels_path: Path) -> None:
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.labels: list[str] = json.loads(labels_path.read_text("utf-8"))

    def predict(self, image_bytes: bytes) -> list[Candidate]:
        return self.predict_image(open_image(image_bytes))

    def predict_image(self, image: Image.Image) -> list[Candidate]:
        logits = self.session.run(None, {"image": preprocess(image)})[0][0]
        scores = np.exp(logits - logits.max())  # softmax: 점수 → 확률
        scores /= scores.sum()

        top = np.argsort(-scores)[:TOP_K]
        return [Candidate(name=self.labels[i], confidence=round(float(scores[i]), 4)) for i in top]
