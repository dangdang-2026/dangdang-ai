"""GPT(Structured Outputs)로 진료 메모에서 3가지만 찾는다 (VISIT-08).

찾는 것: 측정 시점 · 목표 수치 · 다음 진료일.

- 답 모양을 JSON 스키마로 고정한다 (strict) → 필드가 빠지거나 다른 칸이 생기지 않는다
- 값마다 근거 문장(evidence)을 메모에서 그대로 옮기게 한다 → verify.py가 진짜 있는지 확인
- 오류(OpenAIError)·이상한 답(ValueError)은 그대로 올린다 → router가 502
"""

import json
from datetime import date

from openai import OpenAI

from app.visit_note.verify import SLOTS

PROMPT_VERSION = "visit-note-v1"
TIMEOUT_SECONDS = 15.0

SYSTEM = """너는 임신성 당뇨 산모가 진료 뒤 적은 메모에서 아래 5가지만 찾아 옮기는 도우미다.
판단·추천·계산을 하지 않는다. 메모에 없는 값은 만들지 않는다.

- measure_slots: 혈당을 재라고 한 때. FASTING(공복) BREAKFAST_AFTER LUNCH_AFTER DINNER_AFTER
  BEDTIME(자기 전) 중에서. 「세 끼 식후」면 아침·점심·저녁 식후 셋
- fasting_target: 공복 혈당 목표 숫자 (mg/dL, 「95 미만」이면 95)
- post_meal_target: 식후 혈당 목표 숫자
- post_meal_minutes: 식후 몇 분에 재는지 (1시간=60, 2시간=120)
- next_visit_date: 다음 진료 날짜 YYYY-MM-DD. 「2주 뒤」처럼 상대 날짜면
  진료일 {visit_date} 기준으로 계산하고 uncertain=true

규칙:
- evidence에는 그 값이 나온 메모 문장을 **한 글자도 바꾸지 말고 그대로** 옮긴다.
- 메모에 없으면 value=null, evidence=null, uncertain=false.
- 애매하면(「140이었나」「아마」) uncertain=true.
- 약·인슐린 이야기는 절대 옮기지 않는다."""


def _field(value_schema: dict) -> dict:
    return {
        "type": "object",
        "properties": {
            "value": value_schema,
            "evidence": {"type": ["string", "null"]},
            "uncertain": {"type": "boolean"},
        },
        "required": ["value", "evidence", "uncertain"],
        "additionalProperties": False,
    }


NUMBER = {"type": ["integer", "null"]}
SCHEMA = {
    "type": "object",
    "properties": {
        "measure_slots": _field(
            {"type": ["array", "null"], "items": {"type": "string", "enum": list(SLOTS)}}
        ),
        "fasting_target": _field(NUMBER),
        "post_meal_target": _field(NUMBER),
        "post_meal_minutes": _field(NUMBER),
        "next_visit_date": _field({"type": ["string", "null"]}),
    },
    "required": [
        "measure_slots",
        "fasting_target",
        "post_meal_target",
        "post_meal_minutes",
        "next_visit_date",
    ],
    "additionalProperties": False,
}


class OpenAINoteOrganizer:
    def __init__(self, client: OpenAI, model: str) -> None:
        self.client = client
        self.model = model

    def organize(self, sentences: list[str], visit_date: date) -> dict:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM.format(visit_date=visit_date.isoformat())},
                {"role": "user", "content": "\n".join(sentences)},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "visit_note", "strict": True, "schema": SCHEMA},
            },
            timeout=TIMEOUT_SECONDS,
        )
        raw = json.loads(completion.choices[0].message.content or "")
        if not isinstance(raw, dict):
            raise ValueError("JSON 객체가 아님")
        return raw
