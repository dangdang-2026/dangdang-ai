import numpy as np
import pytest
from PIL import Image

from app.food.detector import MAX_DISHES, DishDetector

# 말 3개: 앞 2개는 음식 그릇, 마지막은 컵
POSITIVE = 2


class FakeSession:
    """dish.onnx 대신 정해 둔 자리와 점수를 돌려준다 (자리는 정사각형 기준 cx·cy·w·h 0~1)"""

    def __init__(self, rows: list[tuple[float, float, float, float, float, float, float]]):
        self.boxes = np.array([[r[:4] for r in rows]], dtype=np.float32)
        self.scores = np.array([[r[4:] for r in rows]], dtype=np.float32)

    def run(self, _outputs, _inputs):
        return self.boxes, self.scores


def detector(rows) -> DishDetector:
    d = DishDetector.__new__(DishDetector)  # 모델 파일 없이
    d.session = FakeSession(rows)
    d.size, d.positive = 32, POSITIVE
    d.mean, d.std = np.zeros(3, np.float32), np.ones(3, np.float32)
    return d


SQUARE = Image.new("RGB", (400, 400))


def test_box_is_ratio_of_photo():
    boxes = detector([(0.25, 0.5, 0.3, 0.4, 0.5, 0.1, 0.0)]).detect(SQUARE)

    assert len(boxes) == 1
    assert boxes[0].model_dump() == pytest.approx(
        {"x": 0.1, "y": 0.3, "width": 0.3, "height": 0.4}, abs=1e-4
    )


def test_wide_photo_maps_back_from_padded_square():
    # 800×400 사진은 800×800 정사각형 위쪽에 놓인다. 정사각형 기준 y 0.25 → 사진 기준 0.5
    boxes = detector([(0.5, 0.25, 0.2, 0.2, 0.5, 0.0, 0.0)]).detect(Image.new("RGB", (800, 400)))

    assert boxes[0].model_dump() == pytest.approx(
        {"x": 0.4, "y": 0.3, "width": 0.2, "height": 0.4}, abs=1e-4
    )


def test_drops_low_score_cup_tiny_and_whole_table():
    rows = [
        (0.2, 0.2, 0.2, 0.2, 0.05, 0.05, 0.0),  # 점수 낮음
        (0.5, 0.5, 0.2, 0.2, 0.3, 0.0, 0.6),  # 컵 점수가 더 높음
        (0.8, 0.8, 0.05, 0.05, 0.5, 0.0, 0.0),  # 너무 작음 (0.25%)
        (0.5, 0.5, 0.9, 0.9, 0.5, 0.0, 0.0),  # 상 전체 (81%)
        (0.8, 0.2, 0.2, 0.2, 0.4, 0.0, 0.0),  # 남는 것
    ]

    boxes = detector(rows).detect(SQUARE)

    assert [round(b.x, 2) for b in boxes] == [0.7]


def test_overlapping_and_inner_boxes_become_one_dish():
    rows = [
        (0.3, 0.3, 0.3, 0.3, 0.6, 0.0, 0.0),  # 그릇
        (0.32, 0.3, 0.3, 0.3, 0.5, 0.0, 0.0),  # 거의 같은 자리
        (0.3, 0.3, 0.1, 0.1, 0.4, 0.0, 0.0),  # 그릇 안의 일부
        (0.7, 0.7, 0.2, 0.2, 0.3, 0.0, 0.0),  # 다른 그릇
    ]

    boxes = detector(rows).detect(SQUARE)

    assert len(boxes) == 2


def test_at_most_six_in_reading_order():
    rows = [
        (0.1 + 0.11 * i, 0.8 if i % 2 else 0.2, 0.12, 0.12, 0.2 + i * 0.01, 0.0, 0.0)
        for i in range(8)
    ]

    boxes = detector(rows).detect(SQUARE)

    assert len(boxes) == MAX_DISHES
    ys = [b.y for b in boxes]
    assert ys == sorted(ys)  # 위 줄부터
