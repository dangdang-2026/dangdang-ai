"""Spring 만 부르는 내부 주소: 음식 이름 여러 개 → 임베딩 벡터 (foods 매칭용).

사용자 정보는 받지 않는다. 음식 이름 글자만.
"""

import logging
import time
from typing import Annotated, Protocol

from fastapi import APIRouter, Depends, Header, HTTPException
from openai import OpenAIError
from pydantic import BaseModel, Field, field_validator

from app.core.config import settings
from app.core.llm import get_openai_client
from app.embedding.embedder import EMBEDDING_DIM, OpenAIEmbedder
from app.food.router import verify_internal_token

log = logging.getLogger(__name__)
router = APIRouter(prefix="/internal")

MAX_TEXTS = 100  # 한 번에 100개까지 (Spring 채우기 작업이 100개씩 보낸다)
MAX_CHARS = 200  # 음식 이름 하나 최대 길이 (넘으면 잘라서 쓴다)
# 폭 없는 글자 (strip이 안 지운다): 복사·붙여넣기로 딸려 와서 빈 이름이 빈 이름으로 안 보인다
ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"))


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> tuple[list[list[float]], int | None]: ...


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=MAX_TEXTS)

    @field_validator("texts")
    @classmethod
    def no_blank(cls, v: list[str]) -> list[str]:
        cleaned = [t.translate(ZERO_WIDTH).strip()[:MAX_CHARS] for t in v]
        if any(not t for t in cleaned):
            raise ValueError("빈 이름이 있습니다")
        return cleaned


class EmbedResponse(BaseModel):
    model: str
    dim: int
    vectors: list[list[float]]  # 보낸 texts 순서 그대로


def get_embedder() -> Embedder | None:
    client = get_openai_client()
    if client is None:  # 키가 없으면 Spring이 글자 검색만 한다
        return None
    return OpenAIEmbedder(client, settings.embedding_model)


@router.post("/embeddings", dependencies=[Depends(verify_internal_token)])
def embed(
    request: EmbedRequest,
    embedder: Annotated[Embedder | None, Depends(get_embedder)],
    x_request_id: Annotated[str | None, Header()] = None,
) -> EmbedResponse:
    if embedder is None:
        raise HTTPException(status_code=503, detail="LLM 키가 없습니다")
    started = time.monotonic()
    try:
        vectors, tokens = embedder.embed(request.texts)
    except (OpenAIError, ValueError) as e:  # 시간 초과·사용량 초과·키 틀림·이상한 답
        log.warning(
            "[임베딩] 실패 requestId=%s 개수=%d %s",
            x_request_id,
            len(request.texts),
            type(e).__name__,
        )
        raise HTTPException(status_code=502, detail="임베딩 호출 실패") from e
    # 음식 이름은 남기지 않는다 — 개수·토큰만
    log.info(
        "[임베딩] requestId=%s 개수=%d tokens=%s %dms",
        x_request_id,
        len(vectors),
        tokens if tokens is not None else "-",
        (time.monotonic() - started) * 1000,
    )
    return EmbedResponse(model=settings.embedding_model, dim=EMBEDDING_DIM, vectors=vectors)
