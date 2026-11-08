"""food.onnx 를 불러 두고 사진 1장 → 후보 3개를 냄"""

import io
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from app.food.preprocess import preprocess

TOP_K = 3


class Candidate(BaseModel):
    name: str
    confidence: float  # 0~1, 3개를 더해도 1이 안 될 수 있다(나머지 음식 몫)


class NotAnImageError(Exception):
    pass


class FoodPredictor:
    def __init__(self, model_path: Path, labels_path: Path) -> None:
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.labels: list[str] = json.loads(labels_path.read_text("utf-8"))

    def predict(self, image_bytes: bytes) -> list[Candidate]:
        try:
            image = Image.open(io.BytesIO(image_bytes))  # 디스크에 저장하지 않고 메모리에서만
            image.load()
        except (UnidentifiedImageError, OSError) as e:
            raise NotAnImageError from e

        logits = self.session.run(None, {"image": preprocess(image)})[0][0]
        scores = np.exp(logits - logits.max())  # softmax: 점수 → 확률
        scores /= scores.sum()

        top = np.argsort(-scores)[:TOP_K]
        return [Candidate(name=self.labels[i], confidence=round(float(scores[i]), 4)) for i in top]
