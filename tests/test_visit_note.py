import json
from datetime import date
from types import SimpleNamespace

import pytest
from openai import OpenAIError

from app.visit_note.organize import SCHEMA, OpenAINoteOrganizer
from app.visit_note.verify import split_memo, verify

VISIT = date(2026, 11, 14)
MEMO = (
    "공복은 95 미만, 식후는 140 미만으로 보자고 하셨어요. "
    "아침 공복이랑 세 끼 식후 1시간에 재래요. "
    "인슐린은 자기 전 4단위로 시작하재요. "
    "다음 진료는 11월 28일 오전 10시."
)


def field(value, evidence, uncertain=False):
    return {"value": value, "evidence": evidence, "uncertain": uncertain}


def empty():
    return field(None, None)


GOOD = {
    "measure_slots": field(
        ["FASTING", "BREAKFAST_AFTER", "LUNCH_AFTER", "DINNER_AFTER"],
        "아침 공복이랑 세 끼 식후 1시간에 재래요.",
    ),
    "fasting_target": field(95, "공복은 95 미만, 식후는 140 미만으로 보자고 하셨어요."),
    "post_meal_target": field(140, "공복은 95 미만, 식후는 140 미만으로 보자고 하셨어요."),
    "post_meal_minutes": field(60, "아침 공복이랑 세 끼 식후 1시간에 재래요."),
    "next_visit_date": field("2026-11-28", "다음 진료는 11월 28일 오전 10시."),
}


def kept_memo():
    sentences, _ = split_memo(MEMO)
    return "\n".join(sentences)


# ── 앞: 약 문장 빼기 ──


def test_인슐린_문장은_GPT에_보내지_않는다():
    sentences, skipped = split_memo(MEMO)
    assert skipped is True
    assert all("인슐린" not in s for s in sentences)
    assert len(sentences) == 3


def test_약_이야기가_없으면_skipped_false():
    assert split_memo("다음 진료는 11월 28일.") == (["다음 진료는 11월 28일."], False)


# ── 뒤: 근거 확인 ──


def test_근거가_메모에_그대로_있으면_FOUND():
    result = verify(GOOD, kept_memo(), VISIT).model_dump(mode="json")

    assert result["fastingTarget"] == {
        "value": 95,
        "evidence": "공복은 95 미만, 식후는 140 미만으로 보자고 하셨어요.",
        "status": "FOUND",
    }
    assert result["postMealTarget"]["value"] == 140
    assert result["postMealMinutes"]["value"] == 60
    assert result["measureSlots"]["value"] == [
        "FASTING",
        "BREAKFAST_AFTER",
        "LUNCH_AFTER",
        "DINNER_AFTER",
    ]
    assert result["nextVisitDate"] == {
        "value": "2026-11-28",
        "evidence": "다음 진료는 11월 28일 오전 10시.",
        "status": "FOUND",
    }


def test_메모에_없는_근거는_버린다():
    raw = {**GOOD, "fasting_target": field(92, "공복은 92 미만이라고 하셨어요.")}
    assert verify(raw, kept_memo(), VISIT).fastingTarget is None


def test_띄어쓰기만_다르면_같은_문장으로_본다():
    raw = {**GOOD, "next_visit_date": field("2026-11-28", "다음 진료는  11월 28일 오전 10시.")}
    assert verify(raw, kept_memo(), VISIT).nextVisitDate.status == "FOUND"


def test_숫자가_근거에_없으면_확인_필요():
    evidence = "공복은 95 미만, 식후는 140 미만으로 보자고 하셨어요."
    raw = {**GOOD, "fasting_target": field(90, evidence)}
    assert verify(raw, kept_memo(), VISIT).fastingTarget.status == "CHECK_NEEDED"


def test_범위_밖_숫자는_확인_필요():
    memo = "공복은 950 미만이래요."
    raw = {"fasting_target": field(950, "공복은 950 미만이래요.")}
    assert verify(raw, memo, VISIT).fastingTarget.status == "CHECK_NEEDED"


def test_애매하면_확인_필요():
    memo = "식후는 140이었나 그랬어요."
    raw = {"post_meal_target": field(140, "식후는 140이었나 그랬어요.", uncertain=True)}
    assert verify(raw, memo, VISIT).postMealTarget.status == "CHECK_NEEDED"


def test_식후_시간은_10분_단위만_FOUND():
    memo = "식후 45분에 재래요."
    raw = {"post_meal_minutes": field(45, "식후 45분에 재래요.")}
    assert verify(raw, memo, VISIT).postMealMinutes.status == "CHECK_NEEDED"


def test_상대_날짜는_계산해도_확인_필요():
    memo = "다음 진료는 2주 뒤예요."
    raw = {"next_visit_date": field("2026-11-28", "다음 진료는 2주 뒤예요.", uncertain=True)}
    item = verify(raw, memo, VISIT).nextVisitDate
    assert item.value == date(2026, 11, 28)
    assert item.status == "CHECK_NEEDED"


def test_지난_날짜나_이상한_날짜():
    memo = "다음 진료는 11월 1일."
    assert verify(
        {"next_visit_date": field("2026-11-01", memo)}, memo, VISIT
    ).nextVisitDate.status == ("CHECK_NEEDED")
    assert verify({"next_visit_date": field("11월 말", memo)}, memo, VISIT).nextVisitDate is None


def test_모르는_시점_이름은_빼고_확인_필요():
    memo = "공복이랑 간식 뒤에 재래요."
    raw = {"measure_slots": field(["FASTING", "SNACK_AFTER"], memo)}
    item = verify(raw, memo, VISIT).measureSlots
    assert item.value == ["FASTING"]
    assert item.status == "CHECK_NEEDED"


def test_근거에_약_이야기가_섞이면_버린다():
    memo = "인슐린 맞고 공복 95 미만."
    raw = {"fasting_target": field(95, memo)}
    assert verify(raw, memo, VISIT).fastingTarget is None


def test_없는_칸이나_이상한_모양은_None():
    result = verify({"fasting_target": "95", "post_meal_target": empty()}, MEMO, VISIT)
    assert result.fastingTarget is None
    assert result.postMealTarget is None
    assert result.measureSlots is None


# ── GPT 부르기 (가짜 OpenAI) ──


class FakeOpenAI:
    def __init__(self, content=None, error=None):
        self.content = content
        self.error = error
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_스키마를_고정해서_부른다():
    fake = FakeOpenAI(json.dumps(GOOD))
    sentences, _ = split_memo(MEMO)
    raw = OpenAINoteOrganizer(fake, "gpt-5.4-mini").organize(sentences, VISIT)

    assert raw == GOOD
    call = fake.calls[0]
    assert call["model"] == "gpt-5.4-mini"
    assert call["response_format"]["json_schema"]["strict"] is True
    assert call["response_format"]["json_schema"]["schema"] is SCHEMA
    assert "진료일 2026-11-14" in call["messages"][0]["content"]
    assert "인슐린" not in call["messages"][1]["content"]


def test_JSON이_아니면_ValueError():
    with pytest.raises(ValueError):
        OpenAINoteOrganizer(FakeOpenAI("정리했어요"), "m").organize(["a"], VISIT)


def test_OpenAI_오류는_그대로_올린다():
    with pytest.raises(OpenAIError):
        OpenAINoteOrganizer(FakeOpenAI(error=OpenAIError("timeout")), "m").organize(["a"], VISIT)
