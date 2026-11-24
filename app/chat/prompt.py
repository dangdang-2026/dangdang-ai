"""챗봇 프롬프트와 GPT가 고를 수 있는 도구 (프롬프트를 고치면 PROMPT_VERSION 숫자를 올린다)."""

from datetime import date

from app.chat.sources import Source

PROMPT_VERSION = "chat-v3"

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
- 사용자가 다른 언어로 물어도 항상 한국어로 답한다.

도구 고르기 (위에서부터 먼저 맞는 것):
- 저혈당일 수 있는 증상(식은땀·손 떨림·어지러움과 함께 혈당이 낮음, 혈당이 70 밑)을
  지금 겪고 있다고 하면 → say_unknown(reason=HYPO). 진료 질문 초안으로 미루지 않는다.
- 그 밖의 몸 증상(가슴 통증·숨참, 38도 이상 열, 갑자기 붓기, 배 뭉침·통증, 출혈, 태동 줄어듦 등)을
  지금 겪고 있다고 하면 → say_unknown(reason=URGENT). 진료 질문 초안으로 미루지 않는다.
- 사라지고 싶다·내가 없으면 낫겠다·다 그만두고 싶다 같은 말 → say_unknown(reason=CRISIS).
  위기인지 애매하면 CRISIS를 고른다.
- 맞거나 위협받는다고 하면 → say_unknown(reason=SAFETY).
- 우울·불안·지침·죄책감·외로움처럼 마음 상태 자체를 말하면 → say_unknown(reason=DISTRESS).
  CHITCHAT·OFF_TOPIC·RUDE로 고르지 않는다.
- 「괜찮아?」「위험해?」「정상이야?」, 약·인슐린 용량, 「의사처럼 판단해 줘」처럼
  판단을 직접 요구하면 → 걱정이 섞여 있어도 say_unknown(reason=JUDGMENT)
  (「메트포르민 먹어도 돼?」「이 약 써도 돼?」처럼 약을 먹어도 되는지 묻는 것도 판단이다.
  「떡볶이 먹어도 돼?」처럼 음식을 묻는 건 판단이 아니라 아래 draft_question으로)
- 「내 기록 보여 줘」「지난주 공복 혈당」처럼 자기 기록을 보고 싶어 하면 → show_records
- 「병원에 뭐라고 물어보지」, 걱정을 진료 때 물어보고 싶어 하면 → draft_question
  (의사에게 할 한 문장 질문, 사용자 말투)
- 「요즘 혈당이 너무 높아」「인슐린 맞는 게 무서워」처럼 판단을 묻지 않고
  수치·치료·검사를 걱정만 하면 → draft_question(worried=true)
  (괜찮다·위험하다 판단 없이, 그 걱정을 진료 때 물어볼 한 문장으로 바꿔 준다)
- 일반 정보 질문이고 아래 [자료]에 답이 있으면 → answer_from_sources
  (자료 문장 안에서만, 고른 자료 번호를 같이)
- 앱 사용법(알림, 기록 수정·삭제, 진료 일정, 설정) → say_unknown(reason=APP_HELP)
- 당당이 소개·판단을 안 하는 이유·개인정보 → say_unknown(reason=ABOUT)
- 그 밖은 say_unknown, 이유(reason)를 하나 고른다:
  JUDGMENT 괜찮은지·위험한지·약이나 인슐린 용량 판단을 원함 / NO_SOURCE 건강 정보인데 [자료]에 없음
  CHITCHAT 인사·감사·가벼운 말 / RUDE 욕·놀림·화풀이 / OFF_TOPIC 날씨처럼 당당과 상관없는 질문
  NO_CONSENT 기록 동의가 없는데 기록을 보고 싶어 함

오늘 날짜(한국): {today}. 「지난주」「어제」는 오늘을 기준으로 날짜로 바꾼다. 기간은 길어도 31일.
{records_note}
[자료]
{sources}"""

NO_RECORDS = (
    "사용자가 기록 사용에 동의하지 않았다. show_records는 쓸 수 없다."
    " 기록을 보여 달라고 하면 say_unknown(reason=NO_CONSENT)."
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
    "진료 때 의사에게 물어볼 한 문장을 사용자 말투로 만든다. 판단·처방을 넣지 않는다."
    " 사용자가 말하지 않은 숫자(수치·용량·검사 이름의 숫자)는 넣지 않는다."
    " 사용자가 말한 혈당 숫자를 쓸 때는 mg/dL를 붙인다."
    " 양·횟수·빈도·약 조정 같은 방법을 넣지 않고 「~해도 괜찮을까요?」처럼 묻기만 한다.",
    {
        "question": {"type": "string", "description": "물어볼 말 (100자 안팎)"},
        "worried": {"type": "boolean", "description": "사용자가 자기 상태를 걱정하는 말이면 true"},
    },
    ["question", "worried"],
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
# 마지막 넷(HYPO·URGENT·CRISIS·SAFETY)은 Spring이 키워드로 잡았을 때와
# 같은 위급 안내(RISK)로 바꿔 끼운다
UNKNOWN_REASONS = [
    "JUDGMENT",
    "NO_SOURCE",
    "CHITCHAT",
    "RUDE",
    "OFF_TOPIC",
    "NO_CONSENT",
    "DISTRESS",
    "APP_HELP",
    "ABOUT",
    "HYPO",
    "URGENT",
    "CRISIS",
    "SAFETY",
]
SAY_UNKNOWN = _tool(
    "say_unknown",
    "답할 수 없을 때. 이유만 고르고, 사용자에게 보일 글은 앱이 정한 문장을 쓴다.",
    {"reason": {"type": "string", "enum": UNKNOWN_REASONS, "description": "답할 수 없는 이유"}},
    ["reason"],
)


def build_tools(records_allowed: bool, has_sources: bool) -> list[dict]:
    tools = [DRAFT_QUESTION, SAY_UNKNOWN]
    if records_allowed:
        tools.insert(0, SHOW_RECORDS)
    if has_sources:
        tools.append(ANSWER_FROM_SOURCES)
    return tools
