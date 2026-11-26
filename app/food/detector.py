"""dish.onnx 를 불러 두고 사진 1장 → 음식 그릇 자리(최대 6개)를 냄

그릇 찾기 모델은 OWL-ViT(글로 찾을 것을 알려 주는 검출 모델)를 「음식이 담긴 그릇」 같은 말로
고정해서 ONNX 로 내보낸 것이다 (training/export_dish_detector.py). 음식 이름은 여기서 정하지 않고,
찾은 자리를 잘라 기존 150종 분류 모델(predictor.py)에 맡긴다.
"""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from pydantic import BaseModel

MAX_DISHES = 6  # 백엔드도 6개까지만 쓴다
MIN_SCORE = 0.1  # OWL-ViT 점수는 낮게 나오는 편 (한상 사진 23장으로 정함)
MIN_AREA = 0.01  # 사진의 1%보다 작은 자리는 버림 (양념 종지, 젓가락 끝)
MAX_AREA = 0.6  # 사진의 60%보다 큰 자리는 상 전체라서 버림
SAME_DISH_IOU = 0.3  # 두 자리가 이만큼 겹치면 같은 그릇
INSIDE_RATIO = 0.8  # 작은 자리가 큰 자리 안에 80% 넘게 들어가면 그릇 안의 일부


class Box(BaseModel):
    """사진 기준 0~1 비율. x·y는 왼쪽 위 모서리"""

    x: float
    y: float
    width: float
    height: float


class DishDetector:
    def __init__(self, model_path: Path, config_path: Path) -> None:
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        config = json.loads(config_path.read_text("utf-8"))
        self.size: int = config["size"]
        # 앞쪽 n개 말이 「음식 그릇」, 나머지는 빈 그릇·컵 등
        self.positive: int = config["positive"]
        self.mean = np.array(config["mean"], dtype=np.float32)
        self.std = np.array(config["std"], dtype=np.float32)

    def detect(self, image: Image.Image) -> list[Box]:
        """image 는 회전을 바로잡은 RGB 사진"""
        width, height = image.size
        side = max(width, height)
        # 오른쪽·아래를 채워 정사각형으로 (모델 입력이 정사각형이고, 좌표를 되돌리기 쉽다)
        square = Image.new("RGB", (side, side), (128, 128, 128))
        square.paste(image, (0, 0))
        array = np.asarray(square.resize((self.size, self.size), Image.Resampling.BICUBIC))
        array = (array.astype(np.float32) / 255.0 - self.mean) / self.std
        boxes, scores = self.session.run(None, {"image": array.transpose(2, 0, 1)[np.newaxis]})

        food = scores[0][:, : self.positive].max(axis=1)
        other = scores[0][:, self.positive :].max(axis=1)
        corners = _to_corners(boxes[0] * side, width, height)  # 정사각형 비율 → 원래 사진 픽셀
        area = (corners[:, 2] - corners[:, 0]) * (corners[:, 3] - corners[:, 1]) / (width * height)
        keep = (food >= MIN_SCORE) & (food > other) & (area > MIN_AREA) & (area < MAX_AREA)
        corners, food = corners[keep], food[keep]

        picked = _remove_overlaps(corners, food)[:MAX_DISHES]
        # 위 줄부터, 같은 줄은 왼쪽부터 (화면에 번호를 붙일 때 읽는 순서)
        picked.sort(key=lambda i: (round(corners[i, 1] / height * 4), corners[i, 0]))
        return [
            Box(
                x=round(float(corners[i, 0] / width), 4),
                y=round(float(corners[i, 1] / height), 4),
                width=round(float((corners[i, 2] - corners[i, 0]) / width), 4),
                height=round(float((corners[i, 3] - corners[i, 1]) / height), 4),
            )
            for i in picked
        ]


def _to_corners(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """(가운데 x, 가운데 y, 폭, 높이) → (왼, 위, 오른, 아래), 사진 밖은 잘라 냄"""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    corners = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    corners[:, [0, 2]] = corners[:, [0, 2]].clip(0, width)
    corners[:, [1, 3]] = corners[:, [1, 3]].clip(0, height)
    return corners


def _remove_overlaps(corners: np.ndarray, scores: np.ndarray) -> list[int]:
    """점수 높은 자리부터 남기고, 그 자리와 겹치는 자리는 지움"""
    order = list(np.argsort(-scores))
    kept: list[int] = []
    while order:
        best = order.pop(0)
        kept.append(int(best))
        order = [i for i in order if not _same_dish(corners[best], corners[i])]
    return kept


def _same_dish(a: np.ndarray, b: np.ndarray) -> bool:
    inter_w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    inter_h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = inter_w * inter_h
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    iou = inter / (area_a + area_b - inter + 1e-9)
    inside = inter / (min(area_a, area_b) + 1e-9)
    return iou >= SAME_DISH_IOU or inside >= INSIDE_RATIO
