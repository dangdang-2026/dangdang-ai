"""챗봇 프롬프트와 GPT가 고를 수 있는 도구 (프롬프트를 고치면 PROMPT_VERSION 숫자를 올린다)."""

from datetime import date

from app.chat.sources import Source

PROMPT_VERSION = "chat-v1"

SLOTS = [
    "FASTING",
    "BREAKFAST_AFTER",
    "LUNCH_AFTER",
    "DINNER_AFTER",
    "BEDTIME",
]

SYSTEM = """너는 임신성 당뇨 산모 앱 「당당」의 도우미 「당당이」다.
답을 주는 AI가 아니라 **진료 때 묻는 걸 돕는 AI**다.
반드시 도구 하나를 골라 부른다. 도구 없이 답하지 않는다.

지켜야 할 것:
- 진단·처방·약이나 인슐린 용량·「괜찮다/위험하다」 판단을 하지 않는다.
- 사용자의 혈당 숫자를 직접 말하거나 해석하지 않는다.
  기록이 필요하면 show_records로 기간과 시점만 정한다(숫자는 앱이 센다).
- 병원·의료진 안내가 언제나 우선이라고 생각한다.
- 짧고 다정한 존댓말. 한 번에 2~3문장.

도구 고르기:
- 「내 기록 보여 줘」「지난주 공복 혈당」처럼 자기 기록을 보고 싶어 하면 → show_records
- 「병원에 뭐라고 물어보지」, 걱정을 진료 때 물어보고 싶어 하면 → draft_question
  (의사에게 할 한 문장 질문, 사용자 말투)
- 일반 정보 질문이고 아래 [자료]에 답이 있으면 → answer_from_sources
  (자료 문장 안에서만, 고른 자료 번호를 같이)
- 그 밖, 자료에 없음, 판단·용량을 묻는 질문 → say_unknown

오늘 날짜(한국): {today}. 「지난주」「어제」는 오늘을 기준으로 날짜로 바꾼다. 기간은 길어도 31일.
{records_note}
[자료]
{sources}"""

NO_RECORDS = (
    "사용자가 기록 사용에 동의하지 않았다. show_records는 쓸 수 없다."
    " 기록을 보여 달라고 하면 say_unknown."
)


def build_system(today: date, records_allowed: bool, sources: list[Source]) -> str:
    if sources:
        blocks = "\n".join(
            f"[{s.id}] {s.title}\n" + "\n".join(f"- {fact}" for fact in s.facts) for s in sources
        )
    else:
        blocks = "(없음 — answer_from_sources는 쓸 수 없다)"
    return SYSTEM.format(
        today=today.isoformat(), records_note="" if records_allowed else NO_RECORDS, sources=blocks
    )


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


SHOW_RECORDS = _tool(
    "show_records",
    "사용자의 혈당 기록을 앱 화면 카드로 보여 준다. 기간과 측정 시점만 정한다.",
    {
        "from": {"type": "string", "description": "시작 날짜 YYYY-MM-DD"},
        "to": {"type": "string", "description": "끝 날짜 YYYY-MM-DD (오늘 이하)"},
        "slot": {
            "type": ["string", "null"],
            "enum": [*SLOTS, None],
            "description": "시점, 전체면 null",
        },
    },
    ["from", "to", "slot"],
)
DRAFT_QUESTION = _tool(
    "draft_question",
    "진료 때 의사에게 물어볼 한 문장을 사용자 말투로 만든다. 판단·처방을 넣지 않는다.",
    {"question": {"type": "string", "description": "물어볼 말 (100자 안팎)"}},
    ["question"],
)
ANSWER_FROM_SOURCES = _tool(
    "answer_from_sources",
    "[자료] 문장 안에서만 짧게 답한다.",
    {
        "answer": {"type": "string", "description": "2~3문장 답"},
        "source_ids": {"type": "array", "items": {"type": "string"}, "description": "쓴 자료 번호"},
    },
    ["answer", "source_ids"],
)
SAY_UNKNOWN = _tool("say_unknown", "답할 수 없을 때. 병원에 물어보라고 안내한다.", {}, [])


def build_tools(records_allowed: bool, has_sources: bool) -> list[dict]:
    tools = [DRAFT_QUESTION, SAY_UNKNOWN]
    if records_allowed:
        tools.insert(0, SHOW_RECORDS)
    if has_sources:
        tools.append(ANSWER_FROM_SOURCES)
    return tools
