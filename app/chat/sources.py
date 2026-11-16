"""챗봇이 「일반 정보」(Q-06)에 쓸 수 있는 승인 자료 (ASK-05).

규칙: 학회·공공기관의 환자용 자료만.
**원문을 열어 확인한 문장만** facts에 그대로 옮긴다 (요약·추측 금지).
facts가 빈 자료는 쓰지 않는다 → 하나도 없으면 챗봇은 일반 정보 대신 「병원에 물어보세요」로 답한다.
자료를 넣을 때마다 checked(확인한 날)를 적고, 원문 캡처를 노션 「ASK-05 승인 자료」에 남긴다.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    id: str  # GPT가 고를 번호 (S1, S2 …)
    title: str  # 화면 출처 칩에 보일 이름
    url: str
    checked: str  # 원문을 확인한 날 (YYYY-MM-DD)
    facts: tuple[str, ...]  # 원문에서 그대로 옮긴 문장들


# ⚠️ 비어 있으면 일반 정보 답을 안 한다. 원문 확인 뒤 아래 모양으로 채운다:
# Source(id="S1", title="질병관리청 국가건강정보포털 · 임신성 당뇨병", url="https://health.kdca.go.kr/…",
#        checked="2026-11-15", facts=("원문 문장 1", "원문 문장 2")),
SOURCES: tuple[Source, ...] = ()


def usable_sources() -> list[Source]:
    return [s for s in SOURCES if s.facts]
