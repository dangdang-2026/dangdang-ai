"""Spring 만 부르는 내부 주소: 사진 1장 → 음식 후보 3개, 한상 사진이면 그릇마다 위치와 후보"""

import secrets
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image
from pydantic import BaseModel

from app.core.config import settings
from app.food.detector import Box, DishDetector
from app.food.predictor import Candidate, FoodPredictor, NotAnImageError, open_image

MAX_BYTES = 10 * 1024 * 1024  # 10MB
CROP_MARGIN = 0.1  # 그릇 자리를 자를 때 가장자리를 10% 더 넣는다 (그릇 테두리가 잘리지 않게)

router = APIRouter(prefix="/internal")


class Item(BaseModel):
    box: Box | None  # null 이면 위치를 모름 (사진 전체를 음식 하나로 봄)
    candidates: list[Candidate]


class FoodCandidatesResponse(BaseModel):
    candidates: list[Candidate]  # 사진 전체를 음식 하나로 본 후보 (예전 계약 그대로)
    items: list[Item]  # 사진 속 음식마다 (그릇을 2개 넘게 못 찾으면 1개)


def verify_internal_token(x_internal_token: Annotated[str | None, Header()] = None) -> None:
    if not settings.internal_token or not secrets.compare_digest(
        x_internal_token or "", settings.internal_token
    ):
        raise HTTPException(status_code=401, detail="내부 토큰이 없거나 틀렸습니다")


def get_predictor(request: Request) -> FoodPredictor:
    predictor = request.app.state.food_predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="음식 모델이 아직 준비되지 않았습니다")
    return predictor


def get_detector(request: Request) -> DishDetector | None:
    # 그릇 찾기 모델이 없으면 예전처럼 사진 전체를 음식 하나로 본다 (503 아님)
    return getattr(request.app.state, "dish_detector", None)


@router.post("/food-candidates", dependencies=[Depends(verify_internal_token)])
async def food_candidates(
    image: UploadFile,
    predictor: Annotated[FoodPredictor, Depends(get_predictor)],
    detector: Annotated[DishDetector | None, Depends(get_detector)],
) -> FoodCandidatesResponse:
    image_bytes = await image.read(MAX_BYTES + 1)
    if len(image_bytes) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="사진은 10MB 까지입니다")
    try:
        # 모델 계산은 1~3초 걸려서 다른 요청을 막지 않게 따로 돌린다
        return await run_in_threadpool(analyze, image_bytes, predictor, detector)
    except NotAnImageError:
        raise HTTPException(status_code=400, detail="사진 파일이 아닙니다") from None


def analyze(
    image_bytes: bytes, predictor: FoodPredictor, detector: DishDetector | None
) -> FoodCandidatesResponse:
    photo = open_image(image_bytes)
    whole = predictor.predict_image(photo)
    boxes = detector.detect(photo) if detector is not None else []

    if len(boxes) < 2:  # 그릇이 하나뿐이면 사진 전체로 본 후보가 더 정확하다
        box = boxes[0] if boxes else None
        return FoodCandidatesResponse(candidates=whole, items=[Item(box=box, candidates=whole)])

    items = [Item(box=box, candidates=predictor.predict_image(crop(photo, box))) for box in boxes]
    return FoodCandidatesResponse(candidates=whole, items=items)


def crop(photo: Image.Image, box: Box) -> Image.Image:
    width, height = photo.size
    margin_x, margin_y = box.width * CROP_MARGIN, box.height * CROP_MARGIN
    left = max(0.0, box.x - margin_x) * width
    top = max(0.0, box.y - margin_y) * height
    right = min(1.0, box.x + box.width + margin_x) * width
    bottom = min(1.0, box.y + box.height + margin_y) * height
    return photo.crop((round(left), round(top), round(right), round(bottom)))
