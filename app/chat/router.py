"""Spring 만 부르는 내부 주소: 챗봇 질문 → 무엇을 보여 줄지 (Q-02~Q-06).

위험 신호·약 용량은 Spring이 먼저 규칙으로 거른다 (여기까지 오지 않는다).
"""

import logging
import time
from datetime import datetime
from typing import Annotated, Protocol
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, Header, HTTPException
from openai import OpenAIError
from pydantic import BaseModel, Field

from app.chat.brain import ChatReply, OpenAIChatBrain, Turn
from app.chat.prompt import PROMPT_VERSION
from app.core.config import settings
from app.core.llm import get_openai_client
from app.food.router import verify_internal_token

log = logging.getLogger(__name__)
router = APIRouter(prefix="/internal")
KST = ZoneInfo("Asia/Seoul")


class ChatBrain(Protocol):
    def decide(self, message, history, records_allowed, today) -> ChatReply: ...


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=500)
    recordsAllowed: bool = False
    history: list[Turn] = Field(default_factory=list, max_length=6)


def get_chat_brain() -> ChatBrain | None:
    client = get_openai_client()
    if client is None:  # 키가 없으면 챗봇은 못 한다
        return None
    return OpenAIChatBrain(client, settings.llm_model)


@router.post("/chat", dependencies=[Depends(verify_internal_token)], response_model_by_alias=True)
def chat(
        request: ChatRequest,
        brain: Annotated[ChatBrain | None, Depends(get_chat_brain)],
        x_request_id: Annotated[str | None, Header()] = None,
) -> ChatReply:
    if brain is None:
        raise HTTPException(status_code=503, detail="LLM 키가 없습니다")
    started = time.monotonic()
    today = datetime.now(KST).date()
    try:
        reply = brain.decide(request.message, request.history, request.recordsAllowed, today)
    except OpenAIError as e:  # 시간 초과·사용량 초과·키 틀림
        log.warning(
            "[챗봇] 실패 requestId=%s %s %s", x_request_id, PROMPT_VERSION, type(e).__name__
        )
        raise HTTPException(status_code=502, detail="GPT 호출 실패") from e
    # 질문·답 내용은 남기지 않는다 (SYS-03) — Spring 로그와 requestId로 맞춰 본다
    log.info(
        "[챗봇] requestId=%s promptVersion=%s kind=%s %dms",
        x_request_id,
        reply.promptVersion,
        reply.kind,
        (time.monotonic() - started) * 1000,
        )
    return reply
