"""
진료 메모 → 기준에 넣을 후보 3가지 (C-11·C-12).
"""

import logging
import time
from datetime import date
from typing import Annotated, Protocol

from fastapi import APIRouter, Depends, Header, HTTPException
from openai import OpenAIError
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.llm import get_openai_client
from app.food.router import verify_internal_token
from app.visit_note.organize import PROMPT_VERSION, OpenAINoteOrganizer
from app.visit_note.verify import NoteResult, split_memo, verify

log = logging.getLogger(__name__)
router = APIRouter(prefix="/internal")


class NoteOrganizer(Protocol):
    def organize(self, sentences: list[str], visit_date: date) -> dict: ...


class OrganizeRequest(BaseModel):
    # Spring은 원문 3000자까지 받고, 이름·병원·전화번호를 [이름] 같은 자리표시로 바꿔 보낸다
    # → 늘어날 몫만큼 여유
    memo: str = Field(min_length=1, max_length=4000)
    visitDate: date  # 진료한 날 — 「2주 뒤」 계산 기준


def get_note_organizer() -> NoteOrganizer | None:
    client = get_openai_client()
    if client is None:
        return None
    return OpenAINoteOrganizer(client, settings.llm_model_note)


@router.post("/visit-note/organize", dependencies=[Depends(verify_internal_token)])
def organize(
    request: OrganizeRequest,
    organizer: Annotated[NoteOrganizer | None, Depends(get_note_organizer)],
    x_request_id: Annotated[str | None, Header()] = None,
) -> NoteResult:
    sentences, insulin_skipped = split_memo(request.memo)
    if not sentences:  # 약 이야기뿐이면 GPT를 부르지 않는다
        return NoteResult(insulinSkipped=insulin_skipped, promptVersion=PROMPT_VERSION)
    if organizer is None:
        raise HTTPException(status_code=503, detail="LLM 키가 없습니다")
    started = time.monotonic()
    try:
        raw = organizer.organize(sentences, request.visitDate)
    except (OpenAIError, ValueError) as e:
        log.warning(
            "[메모] 실패 requestId=%s %s %s", x_request_id, PROMPT_VERSION, type(e).__name__
        )
        raise HTTPException(status_code=502, detail="GPT 호출 실패") from e
    result = verify(raw, "\n".join(sentences), request.visitDate)
    result.insulinSkipped = insulin_skipped
    result.promptVersion = PROMPT_VERSION
    found = [n for n, v in result.model_dump().items() if isinstance(v, dict)]
    # 메모 내용·값은 남기지 않는다 — 어떤 칸을 찾았는지만
    log.info(
        "[메모] requestId=%s promptVersion=%s 찾음=%s 약빼기=%s %dms",
        x_request_id,
        PROMPT_VERSION,
        ",".join(found) or "-",
        insulin_skipped,
        (time.monotonic() - started) * 1000,
    )
    return result
