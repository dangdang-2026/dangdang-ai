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
    promptVersion: str = PROMPT_VERSION


def unknown() -> ChatReply:
    return ChatReply(kind="UNKNOWN")


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
        if name == "show_records" and records_allowed:
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
            return ChatReply(
                kind="QUESTION_DRAFT", text="진료 때 이렇게 물어보면 어떨까요?", questionDraft=draft
            )
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
        return unknown()
