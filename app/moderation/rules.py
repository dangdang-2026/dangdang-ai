"""1단계: 규칙으로 의심 문장을 빠르게 고른다 (약·용량 / 위급한 몸 상태 / 마음 위기).

규칙은 일부러 넓게 잡는다. 약·용량은 2단계(GPT)가 한 번 더 걸러 준다.
위급·마음 위기는 놓치는 것이 더 위험하므로 GPT에 묻지 않고 그대로 알린다.
"""

import re
from dataclasses import dataclass
from typing import Literal

Kind = Literal["DOSE", "URGENT_MEDICAL", "MIND"]

# 임신성 당뇨 산모 글에 자주 나오는 약 이름 (늘리고 싶으면 여기만 고친다)
DRUG_NAMES = (
    "인슐린",
    "메트포르민",
    "글루코파지",
    "다이아벡스",
    "노보래피드",
    "휴마로그",
    "란투스",
    "레버미어",
    "트레시바",
    "투제오",
    "아스피린",
    "타이레놀",
    "철분제",
    "엽산",
)
# 숫자 + 약 단위. mg/dL(혈당 단위)은 뺀다
STRONG_DOSE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:단위|유닛|IU|mg(?!\s*/\s*d[lL])|밀리그램|㎎|㎍|mcg|µg|마이크로그램)"
    # 한글 숫자 (「측정한 단위」처럼 앞에 한글이 붙으면 안 봄) 평가 D08·D09
    r"|(?<![가-힣])(?:한|두|세|네|다섯|여섯|일곱|여덟|아홉|열)\s*(?:단위|유닛)",
    re.I,
)
# 숫자 + 알·정·ml — 음식(계란 2알)에도 쓰여서 약 이름이 같이 있을 때만 본다
WEAK_DOSE = re.compile(r"\d+(?:\.\d+)?\s*(?:알|정|mL|ml|cc)")
# 약 이름 바로 뒤 숫자 (「타이레놀 500」). 「인슐린 30분」「엽산 2026년」은 뺀다 — 평가 D10
DRUG_NUMBER = re.compile(
    "(?:" + "|".join(DRUG_NAMES) + r")\s*\d{2,4}(?!\d|\s*(?:분|시|년|주|일|개월))"
)

URGENT = re.compile(
    r"피가\s*(?:조금\s*|좀\s*)?(?:비쳐|비치|비침|나와|났)|출혈|하혈"
    r"|양수가?\s*(?:터졌|터진|새|샌|흘러)"
    r"|태동이?\s*(?:\S+\s+)?(?:줄었|줄어|없어|없|안\s*느껴|약해)"  # 「태동이 확 줄었어요」 평가 U03
    r"|(?:아기|애기)가?\s*(?:\S+\s+){0,2}(?:안\s*움직|움직이지\s*않)"  # 평가 U08
    r"|피\s*(?:섞인|섞여|묻어)"  # 평가 U06
    r"|식은땀|정신을?\s*잃|의식을?\s*잃|의식이\s*(?:흐려|희미)|쓰러[져졌지]"  # 저혈당·실신 평가 U07
    r"|배가\s*심하게\s*(?:아파|아프|뭉쳐|뭉치)|(?:심한|극심한|참을\s*수\s*없는)\s*통증"
    r"|(?:시야가|눈앞이)\s*(?:흐려|흐릿|침침)|경련"
)
MIND = re.compile(
    r"죽고\s*싶|사라지고\s*싶|없어지고\s*싶|살기\s*싫|살고\s*싶지\s*않|자해|극단적\s*(?:인\s*)?선택"
    # 돌려 말하기 — 평가 M04
    r"|살아서\s*뭐\s*하|내일이\s*안\s*왔으면"
)

SENTENCE_END = re.compile(r"(?<=[.!?。…])\s+|\n+")
MAX_QUOTE = 60  # 화면 인용 칸 길이
MAX_HITS = 5


@dataclass(frozen=True)
class RuleHit:
    kind: Kind
    quote: str  # 걸린 문장 (화면에 「“…”」로 보여 준다)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_END.split(text) if s and s.strip()]


def classify(sentence: str) -> Kind | None:
    """한 문장의 종류. 여러 개면 위험한 쪽(마음 위기 > 위급 > 약·용량)을 고른다."""
    if MIND.search(sentence):
        return "MIND"
    if URGENT.search(sentence):
        return "URGENT_MEDICAL"
    has_drug = any(name in sentence for name in DRUG_NAMES)
    if STRONG_DOSE.search(sentence) or (
        has_drug and (WEAK_DOSE.search(sentence) or DRUG_NUMBER.search(sentence))
    ):
        return "DOSE"
    return None


def find_hits(title: str, content: str) -> list[RuleHit]:
    hits: list[RuleHit] = []
    for sentence in split_sentences(f"{title}\n{content}"):
        kind = classify(sentence)
        if kind is None:
            continue
        hit = RuleHit(kind=kind, quote=sentence[:MAX_QUOTE])
        if hit not in hits:
            hits.append(hit)
        if len(hits) == MAX_HITS:
            break
    return hits
