import json
from datetime import date
from types import SimpleNamespace

import pytest
from openai import OpenAIError

from app.chat import brain as brain_module
from app.chat.brain import OpenAIChatBrain, Turn
from app.chat.sources import Source

TODAY = date(2026, 11, 14)
SOURCE = Source(
    id="S1",
    title="시험 자료",
    url="https://example.org/gdm",
    checked="2026-11-14",
    facts=("시험용 문장입니다.",),
)


class FakeOpenAI:
    """openai.OpenAI 와 같은 모양만 흉내: GPT가 도구 하나를 고른 것처럼 돌려준다."""

    def __init__(self, tool=None, arguments=None, error=None):
        self.tool = tool
        self.arguments = arguments
        self.error = error
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        tool_calls = None
        if self.tool:
            args = self.arguments if isinstance(self.arguments, str) else json.dumps(self.arguments)
            tool_calls = [SimpleNamespace(function=SimpleNamespace(name=self.tool, arguments=args))]
        message = SimpleNamespace(content=None, tool_calls=tool_calls)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def tool_names(call):
    return [t["function"]["name"] for t in call["tools"]]


@pytest.fixture
def no_sources(monkeypatch):
    monkeypatch.setattr(brain_module, "usable_sources", lambda: [])


@pytest.fixture
def one_source(monkeypatch):
    monkeypatch.setattr(brain_module, "usable_sources", lambda: [SOURCE])


def decide(fake, message="질문", records_allowed=True, history=()):
    brain = OpenAIChatBrain(fake, "gpt-5.4-nano")
    return brain.decide(message, list(history), records_allowed, TODAY)


# ── 기록 (Q-03) ──


def test_기록은_기간과_시점만_돌려준다(no_sources):
    fake = FakeOpenAI("show_records", {"from": "2026-11-07", "to": "2026-11-13", "slot": "FASTING"})
    reply = decide(fake, "지난주 공복 혈당 보여 줘")

    assert reply.kind == "RECORDS"
    assert reply.records.model_dump(by_alias=True) == {
        "from": date(2026, 11, 7),
        "to": date(2026, 11, 13),
        "slot": "FASTING",
    }
    assert reply.text is None  # 글은 Spring이 붙인다 (숫자는 GPT가 안 만든다)
    call = fake.calls[0]
    assert call["model"] == "gpt-5.4-nano"
    assert call["tool_choice"] == "required"
    assert "show_records" in tool_names(call)
    assert "오늘 날짜(한국): 2026-11-14" in call["messages"][0]["content"]


def test_기록_동의가_없으면_기록_도구를_주지_않는다(no_sources):
    fake = FakeOpenAI("say_unknown", {})
    reply = decide(fake, "내 기록 보여 줘", records_allowed=False)

    assert reply.kind == "UNKNOWN"
    assert "show_records" not in tool_names(fake.calls[0])
    assert "show_records는 쓸 수 없다" in fake.calls[0]["messages"][0]["content"]


def test_동의가_없는데_기록_도구를_부르면_무시한다(no_sources):
    fake = FakeOpenAI("show_records", {"from": "2026-11-07", "to": "2026-11-13", "slot": None})
    assert decide(fake, records_allowed=False).kind == "UNKNOWN"


def test_이상한_날짜와_시점은_비운다(no_sources):
    fake = FakeOpenAI("show_records", {"from": "지난주", "to": "2026-13-01", "slot": "LUNCH"})
    reply = decide(fake)
    assert reply.records.model_dump(by_alias=True) == {"from": None, "to": None, "slot": None}


# ── 물어볼 말 (Q-04) ──


def test_물어볼_말_초안(no_sources):
    question = "  공복 혈당이 높게 나온 날 무엇을 바꿔야 할까요?  "
    fake = FakeOpenAI("draft_question", {"question": question})
    reply = decide(fake, "공복 혈당 높으면 병원에 뭐라고 물어보지")

    assert reply.kind == "QUESTION_DRAFT"
    assert reply.questionDraft == "공복 혈당이 높게 나온 날 무엇을 바꿔야 할까요?"
    assert reply.text == "진료 때 이렇게 물어보면 어떨까요?"


def test_물어볼_말은_200자로_자른다(no_sources):
    fake = FakeOpenAI("draft_question", {"question": "가" * 300})
    assert len(decide(fake).questionDraft) == 200


def test_빈_초안은_UNKNOWN(no_sources):
    assert decide(FakeOpenAI("draft_question", {"question": " "})).kind == "UNKNOWN"


# ── 일반 정보 (Q-06) ──


def test_승인_자료가_없으면_정보_도구를_주지_않는다(no_sources):
    fake = FakeOpenAI("say_unknown", {})
    decide(fake, "출산하면 낫나요?")
    assert "answer_from_sources" not in tool_names(fake.calls[0])
    assert "(없음" in fake.calls[0]["messages"][0]["content"]


def test_자료_안에서_답하면_출처를_붙인다(one_source):
    args = {"answer": "자료에 따르면 이래요.", "source_ids": ["S1"]}
    fake = FakeOpenAI("answer_from_sources", args)
    reply = decide(fake, "출산하면 낫나요?")

    assert reply.kind == "INFO"
    assert reply.text == "자료에 따르면 이래요."
    assert [s.model_dump() for s in reply.sources] == [
        {"title": "시험 자료", "url": "https://example.org/gdm"}
    ]
    assert "[S1] 시험 자료\n- 시험용 문장입니다." in fake.calls[0]["messages"][0]["content"]
    assert "answer_from_sources" in tool_names(fake.calls[0])


def test_없는_자료_번호면_UNKNOWN(one_source):
    fake = FakeOpenAI("answer_from_sources", {"answer": "그럴듯한 답", "source_ids": ["S9"]})
    assert decide(fake).kind == "UNKNOWN"


def test_답에_약_용량_숫자가_섞이면_버린다(one_source):
    args = {"answer": "인슐린 8단위가 보통이에요.", "source_ids": ["S1"]}
    fake = FakeOpenAI("answer_from_sources", args)
    assert decide(fake).kind == "UNKNOWN"


# ── 그 밖 ──


def test_도구를_안_부르거나_인자가_깨지면_UNKNOWN(no_sources):
    assert decide(FakeOpenAI()).kind == "UNKNOWN"
    assert decide(FakeOpenAI("draft_question", "{깨진 JSON")).kind == "UNKNOWN"
    assert decide(FakeOpenAI("모르는_도구", {})).kind == "UNKNOWN"


def test_앞_대화를_순서대로_붙인다(no_sources):
    fake = FakeOpenAI("say_unknown", {})
    history = [Turn(role="user", text="첫 질문"), Turn(role="assistant", text="첫 답")]
    decide(fake, "두 번째 질문", history=history)

    roles = [(m["role"], m["content"]) for m in fake.calls[0]["messages"][1:]]
    assert roles == [("user", "첫 질문"), ("assistant", "첫 답"), ("user", "두 번째 질문")]


def test_OpenAI_오류는_그대로_올린다(no_sources):
    with pytest.raises(OpenAIError):
        decide(FakeOpenAI(error=OpenAIError("Request timed out.")))
