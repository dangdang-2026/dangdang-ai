"""Spring 만 부르는 내부 주소: 글 제목·본문 -> 올리기 전에 알려 줄 문장들 (U-15 · U-30)."""

from typing import Annotated, Literal, Protocol

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.llm import get_openai_client
from app.food.router import verify_internal_token
from app.moderation.dose_check import OpenAIDoseChecker
from app.moderation.rules import find_hits

router = APIRouter(prefix="/internal")


class DoseChecker(Protocol):
    def confirm_dose(self, sentences: list[str]) -> list[bool] | None: ...


class PostCheckRequest(BaseModel):
    title: str = Field(default="", max_length=100)
    content: str = Field(min_length=1, max_length=5000)
    # Spring이 커뮤니티 AI 확인 동의(COMMUNITY_AI_CHECK)를 보고 정한다
    # False면 규칙만 (해외로 안 보냄)
    # 안 보내면 False — 동의를 모르는 요청(옛 Spring 등)은 GPT로 보내지 않는다
    allowLlm: bool = False


class Issue(BaseModel):
    kind: Literal["DOSE", "URGENT_MEDICAL", "MIND", "LOW_GLUCOSE"]
    quote: str


class PostCheckResponse(BaseModel):
    issues: list[Issue]
    llmChecked: bool  # 약·용량을 GPT로 한 번 더 확인했는지 (로그·발표용)


def get_dose_checker() -> DoseChecker | None:
    client = get_openai_client()
    if client is None:  # 키가 없으면 규칙만으로 동작
        return None
    return OpenAIDoseChecker(client, settings.llm_model)


@router.post("/post-check", dependencies=[Depends(verify_internal_token)])
def post_check(
    request: PostCheckRequest,
    checker: Annotated[DoseChecker | None, Depends(get_dose_checker)],
) -> PostCheckResponse:
    hits = find_hits(request.title, request.content)
    dose_hits = [h for h in hits if h.kind == "DOSE"]

    llm_checked = False
    if dose_hits and checker is not None and request.allowLlm:
        confirmed = checker.confirm_dose([h.quote for h in dose_hits])
        if confirmed is not None:  # 물어봤으면 아니라고 한 문장은 뺀다
            llm_checked = True
            dropped = {h for h, ok in zip(dose_hits, confirmed, strict=True) if not ok}
            hits = [h for h in hits if h not in dropped]

    return PostCheckResponse(
        issues=[Issue(kind=h.kind, quote=h.quote) for h in hits],
        llmChecked=llm_checked,
    )
