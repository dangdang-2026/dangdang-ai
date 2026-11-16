"""진료 메모 정리 앞뒤 검사 (VISIT-08).

앞: 인슐린·약 이야기 문장은 GPT에 보내지도 않는다 (정리하지 않는다).
뒤: GPT가 준 값마다 근거 문장이 **메모에 그대로 있는지** 확인한다. 없으면 버린다(지어낸 값).
    숫자가 근거에 없거나·범위 밖·애매하면 「확인 필요」(CHECK_NEEDED)로 표시한다.
"""

import re
from datetime import date
from typing import Literal

from pydantic import BaseModel

from app.moderation.rules import DRUG_NAMES, STRONG_DOSE, split_sentences

SLOTS = ("FASTING", "BREAKFAST_AFTER", "LUNCH_AFTER", "DINNER_AFTER", "BEDTIME")
Status = Literal["FOUND", "CHECK_NEEDED"]


class Item(BaseModel):
    value: list[str] | int | date
    evidence: str  # 메모에서 그대로 옮긴 근거 문장 (화면 C-12에 인용)
    status: Status


class NoteResult(BaseModel):
    measureSlots: Item | None = None
    fastingTarget: Item | None = None
    postMealTarget: Item | None = None
    postMealMinutes: Item | None = None
    nextVisitDate: Item | None = None
    insulinSkipped: bool = False  # 인슐린·약 이야기를 빼고 정리했는지 (화면에 한 줄 안내)
    promptVersion: str = ""


def _is_medicine(sentence: str) -> bool:
    return bool(STRONG_DOSE.search(sentence)) or any(name in sentence for name in DRUG_NAMES)


def split_memo(memo: str) -> tuple[list[str], bool]:
    """GPT에 보낼 문장들, 인슐린·약 문장을 뺐는지."""
    kept, skipped = [], False
    for sentence in split_sentences(memo):
        if _is_medicine(sentence):
            skipped = True
        else:
            kept.append(sentence)
    return kept, skipped


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _grounded(evidence, memo: str) -> str | None:
    """근거가 메모에 그대로 있으면 그 근거, 아니면 None."""
    if not isinstance(evidence, str) or not evidence.strip():
        return None
    ev = _norm(evidence)
    if ev not in _norm(memo) or _is_medicine(ev):
        return None
    return ev


def _number(field: dict, memo: str, low: int, high: int) -> Item | None:
    value, evidence = field.get("value"), _grounded(field.get("evidence"), memo)
    if not isinstance(value, int) or isinstance(value, bool) or evidence is None:
        return None
    ok = low <= value <= high and str(value) in evidence and not field.get("uncertain")
    return Item(value=value, evidence=evidence, status="FOUND" if ok else "CHECK_NEEDED")


def _minutes(field: dict, memo: str) -> Item | None:
    value, evidence = field.get("value"), _grounded(field.get("evidence"), memo)
    if not isinstance(value, int) or isinstance(value, bool) or evidence is None:
        return None
    # 기준 API(CriteriaVersionCreateRequest)와 같은 규칙: 30~180분, 10분 단위
    ok = 30 <= value <= 180 and value % 10 == 0 and not field.get("uncertain")
    return Item(value=value, evidence=evidence, status="FOUND" if ok else "CHECK_NEEDED")


def _slots(field: dict, memo: str) -> Item | None:
    value, evidence = field.get("value"), _grounded(field.get("evidence"), memo)
    if not isinstance(value, list) or evidence is None:
        return None
    slots = [s for s in SLOTS if s in value]  # 모르는 이름은 빼고, 순서는 하루 순서대로
    if not slots:
        return None
    status = "CHECK_NEEDED" if field.get("uncertain") or len(slots) != len(value) else "FOUND"
    return Item(value=slots, evidence=evidence, status=status)


def _visit_date(field: dict, memo: str, visit_date: date) -> Item | None:
    value, evidence = field.get("value"), _grounded(field.get("evidence"), memo)
    if not isinstance(value, str) or evidence is None:
        return None
    try:
        when = date.fromisoformat(value)
    except ValueError:
        return None
    # 「2주 뒤」처럼 날짜 숫자가 근거에 없으면 계산한 날이라 확인이 필요하다
    written = str(when.month) in evidence and str(when.day) in evidence
    ok = when > visit_date and written and not field.get("uncertain")
    return Item(value=when, evidence=evidence, status="FOUND" if ok else "CHECK_NEEDED")


def verify(raw: dict, memo: str, visit_date: date) -> NoteResult:
    def get(name):
        field = raw.get(name)
        return field if isinstance(field, dict) else {}

    return NoteResult(
        measureSlots=_slots(get("measure_slots"), memo),
        fastingTarget=_number(get("fasting_target"), memo, 60, 140),
        postMealTarget=_number(get("post_meal_target"), memo, 80, 250),
        postMealMinutes=_minutes(get("post_meal_minutes"), memo),
        nextVisitDate=_visit_date(get("next_visit_date"), memo, visit_date),
    )
