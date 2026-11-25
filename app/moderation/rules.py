"""1단계: 규칙으로 의심 문장을 빠르게 고른다 (약·용량 / 위급한 몸 상태 / 마음 위기 / 폭력 / 저혈당).

규칙은 일부러 넓게 잡는다. 약·용량은 2단계(GPT)가 한 번 더 걸러 준다.
위급·마음 위기·폭력·저혈당은 놓치는 것이 더 위험하므로 GPT에 묻지 않고 그대로 알린다.
"""

import re
from dataclasses import dataclass
from typing import Literal

Kind = Literal["DOSE", "URGENT_MEDICAL", "MIND", "SAFETY", "LOW_GLUCOSE"]

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

# 말 목록은 챗봇 be RiskSignalRules와 같이 고친다
# MIND·LOW_GLUCOSE(=be HYPO)는 같은 목록. URGENT는 출혈 줄만 같고 나머지는 일부만 겹친다
# (be는 산후 증상·심한 두통 등을 더 보고, 여기는 평가 사례에서 찾은 말을 더 본다)
URGENT = re.compile(
    # 출혈 (be와 같음): 「피가 너무 많이 나요」 「피 나요」 「갈색 피비침」. 커피·나아졌어요는 뺀다
    r"(?<![가-힣])피\s*(?:가\s*)?(?:(?:조금|좀|많이|계속|너무|엄청|자꾸|막)\s*){0,2}"
    r"(?:비쳐|비치|비침|나와|났|나(?!아))|출혈|하혈"
    r"|양수가?\s*(?:터졌|터진|새|샌|흘러)"
    r"|태동이?\s*(?:\S+\s+)?(?:줄었|줄어|없어|없|안\s*느껴|약해)"  # 「태동이 확 줄었어요」 평가 U03
    r"|(?:아기|애기)가?\s*(?:\S+\s+){0,2}(?:안\s*움직|움직이지\s*않)"  # 평가 U08
    r"|피\s*(?:섞인|섞여|묻어)"  # 평가 U06
    # 실신·의식 (식은땀은 LOW_GLUCOSE로 옮김 — 평가 U07)
    r"|정신을?\s*잃|의식을?\s*잃|의식이\s*(?:흐려|희미)|쓰러[져졌지]"
    r"|배가\s*심하게\s*(?:아파|아프|뭉쳐|뭉치)|(?:심한|극심한|참을\s*수\s*없는)\s*통증"
    r"|(?:시야가|눈앞이)\s*(?:흐려|흐릿|침침)|경련"
)
# 마음 위기 (be RiskSignalRules.MIND와 같은 목록)
MIND = re.compile(
    r"죽고\s*싶|사라지고\s*싶|없어지고\s*싶|살기\s*싫|살고\s*싶지\s*않|자해|극단적\s*(?:인\s*)?선택"
    # 돌려 말하기 — 평가 M04
    r"|살아서\s*뭐\s*하|내일이\s*안\s*왔으면"
    # 「다 그만두고 싶어」는 넓게 잡되 퇴사 이야기(회사·직장·알바)는 뺀다
    r"|(?<!회사 )(?<!회사를 )(?<!직장 )(?<!직장을 )(?<!알바 )(?<!알바를 )"
    r"(?:다|모든\s*걸|전부)\s*그만(?:두|하)고\s*싶"
    r"|(?:모든\s*걸|삶을|인생을|이제\s*다)\s*끝내고\s*싶"
    r"|(?:내가|나만?|제가)\s*없으면\s*(?:다들|모두)?\s*편"
    r"|(?:나를|날|스스로를?)\s*해(?:치|칠)|해치고\s*싶"
)
# 폭력·위협 (be RiskSignalRules.SAFETY + 「누구한테 맞았어」)
# 마음 위기(109)가 아니라 112·1366 안내로 가야 한다 (챗봇 평가 C38 「남편이 때려」)
# 「때려치우다」「때려잡다」는 뺀다
# 「맞았」은 「주사·인슐린 맞았」과 겹쳐서 사람(남편·애인·가족·누구) + 한테·에게 바로 뒤만 본다
PERSON = (
    r"(?:남편|신랑|남친|남자\s*친구|애인|가족|시어머니|시아버지|시댁|친정|엄마|아빠"
    r"|오빠|형|언니|누나|누구|누군가|그\s*사람)"
)
SAFETY = re.compile(
    r"때려(?!\s*[치잡])|때렸|때린|때리(?:고|는|면)|폭력|폭행|협박|목을\s*졸"
    + r"|"
    + PERSON
    + r"\s*(?:한테서?|에게서?)\s*(?:(?:또|자꾸|계속|많이|막|매일|맨날|세게|심하게)\s*)?"
    r"맞(?:았|아서|고\s*있)"
)
# 「아기가 배를 발로 때리는 느낌」은 태동이라 뺀다
# 「남편이 발로 때렸어」처럼 사람이 주어면 빼지 않는다 (「엄마 배를」은 주어가 아님)
FETAL_KICK = re.compile(
    r"(?:아기|애기|태아|뱃속|태동)(?:가|이|는)?\s*(?:\S+\s*){0,3}?때(?:리|려|렸)"
)
HAS_PERSON = re.compile(PERSON + r"\s*(?:이|가|는|도|한테|에게)")


def _is_violence(sentence: str) -> bool:
    if not SAFETY.search(sentence):
        return False
    return not (FETAL_KICK.search(sentence) and not HAS_PERSON.search(sentence))


# 저혈당일 수 있는 말 (U-30c) — 인슐린을 쓰는 산모에게 가장 흔한 위급.
# 처치법은 앱이 말하지 않는다 → 종류만 알리고 화면은 「병원에서 배운 대로 + 119」
# 챗봇 be RiskSignalRules.HYPO와 같은 목록 — 한쪽을 고치면 다른 쪽도
LOW_GLUCOSE = re.compile(
    r"저혈당|식은\s*땀|떨림|덜덜|(?:손|몸|다리)이?\s*(?:덜덜\s*)?떨(?:려|리|림|렸)"
    # 「혈당이 너무 낮아요」 「혈당이 떨어져서」. 낮추는 법·낮은 음식·떨어뜨리는 음식은 뺀다
    r"|혈당이?\s*(?:(?:너무|많이|갑자기|확|뚝)\s*)?"
    r"(?:낮(?![추춰출]|은\s*(?:음식|간식|과일|식품|메뉴))"
    r"|떨어(?![뜨트]|지는\s*(?:음식|간식|과일|식품|운동|방법)))"
    # 혈당 이야기 가까이(8자 안) 70 밑 두 자리 (혈당 50 · 혈당이 62 · 재 보니 58)
    # 세 자리(128)·날짜·시간(60분·30초)·나이(50대)·돈(15만원)·약 용량(12단위)은 뺀다
    # (mg/dL은 혈당 단위라 안 뺌)
    r"|(?:혈당|재\s*보니|쟀더니|쟀는데|측정)[^0-9]{0,8}[1-6]\d(?!\d)"
    r"(?!\s*(?:개월|월|일|시|분|초|주|번|개|회|살|대|%|원|만|천|단위|유닛"
    r"|(?i:kg|iu|mg(?!\s*/\s*dl))))"
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


def _is_dose(sentence: str) -> bool:
    has_drug = any(name in sentence for name in DRUG_NAMES)
    return bool(
        STRONG_DOSE.search(sentence)
        or (has_drug and (WEAK_DOSE.search(sentence) or DRUG_NUMBER.search(sentence)))
    )


def classify(sentence: str) -> list[Kind]:
    """한 문장의 종류. 여러 개면 위험한 쪽 하나를 고른다.

    순서: 마음 위기 > 폭력 > 위급 > 저혈당 > 약·용량.

    저혈당만은 약·용량도 같이 본다
    — 「란투스 10단위 맞았는데 혈당 65」에서 용량 안내(U-15)가 가려지지 않게.
    """
    if MIND.search(sentence):
        return ["MIND"]
    if _is_violence(sentence):
        return ["SAFETY"]
    if URGENT.search(sentence):
        return ["URGENT_MEDICAL"]
    if LOW_GLUCOSE.search(sentence):
        return ["LOW_GLUCOSE", "DOSE"] if _is_dose(sentence) else ["LOW_GLUCOSE"]
    return ["DOSE"] if _is_dose(sentence) else []


def find_hits(title: str, content: str) -> list[RuleHit]:
    hits: list[RuleHit] = []
    for sentence in split_sentences(f"{title}\n{content}"):
        for kind in classify(sentence):
            hit = RuleHit(kind=kind, quote=sentence[:MAX_QUOTE])
            if hit not in hits:
                hits.append(hit)
            if len(hits) == MAX_HITS:
                return hits
    return hits
