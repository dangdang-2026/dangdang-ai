"""GPT에게 「어떤 도구를 쓸지」만 고르게 하고, 결과를 Spring이 그릴 답 모양으로 바꾼다.

- GPT는 숫자를 만들지 않는다: 기록은 기간·시점만 정하고 Spring이 센다 (ASK-03)
- 자료에 없는 정보·용량 숫자가 섞인 답은 버리고 UNKNOWN (ASK-05)
- 오류(OpenAIError)는 그대로 올린다 → router가 502 → Spring이 「답을 만들지 못함」(Q-08)
"""

import json
from datetime import date
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from app.chat.prompt import PROMPT_VERSION, SLOTS, build_system, build_tools
from app.chat.sources import usable_sources
from app.moderation.rules import STRONG_DOSE

MAX_DRAFT = 200
MAX_ANSWER = 400
TIMEOUT_SECONDS = 10.0  # Spring은 15초까지 기다린다


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    text: str = Field(max_length=2000)


class Records(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    from_: date | None = Field(default=None, alias="from")
    to: date | None = None
    slot: str | None = None


class SourceOut(BaseModel):
    title: str
    url: str


class ChatReply(BaseModel):
    kind: Literal["RECORDS", "QUESTION_DRAFT", "INFO", "UNKNOWN"]
    text: str | None = None
    records: Records | None = None
    questionDraft: str | None = None
    sources: list[SourceOut] = Field(default_factory=list)
    reason: str | None = None  # UNKNOWN일 때 이유 (Spring이 버튼을 고르거나 위급 안내로 바꿀 때 씀)
    promptVersion: str = PROMPT_VERSION


# 답할 수 없을 때 보일 글. GPT가 글을 만들지 않고 이유(reason)만 고른다 → 판단·숫자가 섞일 틈이 없다
# HYPO·URGENT·CRISIS·SAFETY는 Spring이 키워드 안내와 같은 위급 안내로 바꿔 끼운다
# → 여기 글은 예전 Spring용 예비
UNKNOWN_TEXTS = {
    "JUDGMENT": ("그건 사람마다 달라서 당당이가 정해 드릴 수 없어요. 진료 때 꼭 물어봐 주세요."),
    "NO_SOURCE": "그 내용은 당당이가 가진 자료에 없어요. 다니는 병원에 물어봐 주세요.",
    "CHITCHAT": "네, 언제든 불러 주세요. 기록 보기나 진료 때 물어볼 말을 도와드릴게요.",
    "RUDE": (
        "도움이 못 돼서 죄송해요. 당당이는 기록 보기와 "
        "진료 때 물어볼 말 만들기를 도와드릴 수 있어요."
    ),
    "OFF_TOPIC": (
        "그건 당당이가 도와드리기 어려워요. 혈당 기록이나 진료 준비라면 언제든 물어봐 주세요."
    ),
    # 기록 동의 없이 기록을 원할 때 (Spring CONSENT_TEXT와 같은 문장). 동의가 있으면 쓰지 않는다
    "NO_CONSENT": (
        "내 기록을 보려면 「기록 사용 동의」가 필요해요. 마이 › AI 설정에서 켤 수 있어요."
    ),
    "DISTRESS": (
        "많이 지치셨죠. 그런 마음이 드는 건 이상한 게 아니에요. "
        "혼자 견디기 힘들면 1577-0199에서 이야기 들어 줘요."
    ),
    "APP_HELP": (
        "앱 사용법은 「마이 › 도움말」에 모아 뒀어요. "
        "기록은 기록 탭, 진료 일정은 진료 탭에서 바꿀 수 있어요."
    ),
    "ABOUT": (
        "당당이는 기록을 보여 주고 진료 때 물어볼 말을 함께 정리하는 AI예요. "
        "몸 상태 판단은 나를 보는 의료진이 가장 정확해서 병원에 물어보시라고 해요."
    ),
    # ↓ 예비 (Spring RiskSignalRules 문장과 뜻을 맞춤)
    "HYPO": (
        "병원에서 안내받은 저혈당 대처를 먼저 해 주세요. "
        "어지럽거나 의식이 흐려지면 바로 119에 전화해 주세요."
    ),
    "URGENT": (
        "지금은 병원에 먼저 연락해 주세요. "
        "피가 많이 나거나, 어지럽거나, 의식이 흐려지면 바로 119에 전화해 주세요."
    ),
    "CRISIS": (
        "혼자 견디지 않아도 돼요. 109(자살예방 상담전화)나 1577-0199(정신건강 위기상담)에서 "
        "24시간 이야기할 수 있어요. 나를 해칠 것 같거나 위급하면 바로 119에 전화해 주세요."
    ),
    "SAFETY": (
        "지금 안전한 게 가장 중요해요. 위험하면 112에 바로 전화해 주세요. "
        "1366(여성긴급전화)은 24시간 상담해 줘요."
    ),
}
DRAFT_TEXT = "진료 때 이렇게 물어보면 어떨까요?"
WORRIED_DRAFT_TEXT = "걱정되셨죠. 진료 때 이렇게 물어보면 어떨까요?"


def unknown(reason: str | None = None) -> ChatReply:
    """이유를 모르거나 이상하면 NO_SOURCE (도구 안 부름·깨진 인자·출처 없음·용량 숫자 등)."""
    reason = reason if reason in UNKNOWN_TEXTS else "NO_SOURCE"
    return ChatReply(kind="UNKNOWN", text=UNKNOWN_TEXTS[reason], reason=reason)


def _parse_date(value) -> date | None:
    try:
        return date.fromisoformat(value) if isinstance(value, str) else None
    except ValueError:
        return None


class OpenAIChatBrain:
    def __init__(self, client: OpenAI, model: str) -> None:
        self.client = client
        self.model = model

    def decide(
        self, message: str, history: list[Turn], records_allowed: bool, today: date
    ) -> ChatReply:
        sources = usable_sources()
        messages = [{"role": "system", "content": build_system(today, records_allowed, sources)}]
        messages += [{"role": t.role, "content": t.text} for t in history]
        messages.append({"role": "user", "content": message})

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=build_tools(records_allowed, bool(sources)),
            tool_choice="required",  # 반드시 도구 하나 → 자유 답(숫자·판단)이 나오지 않게
            parallel_tool_calls=False,
            timeout=TIMEOUT_SECONDS,
        )
        calls = completion.choices[0].message.tool_calls or []
        if not calls:
            return unknown()
        name = calls[0].function.name
        try:
            args = json.loads(calls[0].function.arguments or "{}")
        except ValueError:
            return unknown()
        if not isinstance(args, dict):
            return unknown()
        return self._to_reply(name, args, records_allowed, sources)

    def _to_reply(self, name: str, args: dict, records_allowed: bool, sources) -> ChatReply:
        if name == "show_records" and not records_allowed:
            return unknown("NO_CONSENT")
        if name == "show_records":
            slot = args.get("slot")
            return ChatReply(
                kind="RECORDS",
                records=Records(
                    from_=_parse_date(args.get("from")),
                    to=_parse_date(args.get("to")),
                    slot=slot if slot in SLOTS else None,
                ),
            )
        if name == "draft_question":
            draft = str(args.get("question") or "").strip()[:MAX_DRAFT]
            if not draft:
                return unknown()
            intro = WORRIED_DRAFT_TEXT if args.get("worried") is True else DRAFT_TEXT
            return ChatReply(kind="QUESTION_DRAFT", text=intro, questionDraft=draft)
        if name == "answer_from_sources":
            answer = str(args.get("answer") or "").strip()[:MAX_ANSWER]
            by_id = {s.id: s for s in sources}
            picked = [by_id[i] for i in args.get("source_ids") or [] if i in by_id]
            if not answer or not picked or STRONG_DOSE.search(answer):
                return unknown()  # 출처 없음 · 용량 숫자가 섞임 → 보여 주지 않는다
            return ChatReply(
                kind="INFO",
                text=answer,
                sources=[SourceOut(title=s.title, url=s.url) for s in picked],
            )
        if name == "say_unknown":
            reason = args.get("reason")
            if reason == "NO_CONSENT" and records_allowed:
                reason = None  # 동의했는데 「동의가 필요해요」라고 하면 틀린 안내
            return unknown(reason)
        return unknown()
