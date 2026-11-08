"""Spring 만 부르는 내부 주소: 사진 1장 → 음식 후보 3개."""

import secrets
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Request, UploadFile
from pydantic import BaseModel

from app.core.config import settings
from app.food.predictor import Candidate, FoodPredictor, NotAnImageError

MAX_BYTES = 10 * 1024 * 1024  # 10MB

router = APIRouter(prefix="/internal")


class FoodCandidatesResponse(BaseModel):
    candidates: list[Candidate]


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


@router.post("/food-candidates", dependencies=[Depends(verify_internal_token)])
async def food_candidates(
        image: UploadFile,
        predictor: Annotated[FoodPredictor, Depends(get_predictor)],
) -> FoodCandidatesResponse:
    image_bytes = await image.read(MAX_BYTES + 1)
    if len(image_bytes) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="사진은 10MB 까지입니다")
    try:
        candidates = predictor.predict(image_bytes)
    except NotAnImageError:
        raise HTTPException(status_code=400, detail="사진 파일이 아닙니다") from None
    return FoodCandidatesResponse(candidates=candidates)