"""평가 스크립트 공통 — 사례 읽기 · 내부 주소 부르기 · 결과 저장.

진짜 서버와 똑같은 길(토큰 검사 → 규칙 → GPT → 뒤 검사)을 지나도록 FastAPI TestClient로 부른다.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app

CASES = Path(__file__).parent / "cases"
RESULTS = Path(__file__).parent / "results"
KST = ZoneInfo("Asia/Seoul")


def load_cases(name: str) -> list[dict]:
    lines = (CASES / name).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip() and not line.startswith("//")]


def make_client() -> TestClient:
    if not settings.internal_token:
        sys.exit(".env에 INTERNAL_TOKEN이 없어요 (서버와 같은 값)")
    client = TestClient(app)
    client.headers["X-Internal-Token"] = settings.internal_token
    return client


def percent(count: int, total: int) -> str:
    return f"{count / total:.1%}" if total else "-"


def save(
    name: str, summary: dict, rows: list[dict], model: str | None, prompt_version: str | None
) -> Path:
    """결과를 evals/results/<이름>-<날짜시각>.json 으로 남긴다 (보고서 숫자의 근거).

    model: 실제로 부른 모델 (규칙만이면 None). 기능마다 달라서(메모는 mini) 부르는 쪽이 넘긴다.
    """
    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now(KST).strftime("%Y%m%d-%H%M%S")  # 초까지 — 2~3번 연달아 돌려도 안 덮어쓴다
    path = RESULTS / f"{name}-{stamp}.json"
    body = {"model": model, "promptVersion": prompt_version, "summary": summary, "rows": rows}
    path.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
