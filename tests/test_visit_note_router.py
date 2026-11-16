import pytest
from fastapi.testclient import TestClient
from openai import OpenAIError

from app.core.config import settings
from app.main import app
from app.visit_note.router import get_note_organizer

TOKEN = "test-token"
HEADERS = {"X-Internal-Token": TOKEN, "X-Request-Id": "req-1"}
MEMO = "공복은 95 미만이래요. 인슐린은 4단위로 시작."


class FakeOrganizer:
    def __init__(self, raw=None, error=None):
        self.raw = raw
        self.error = error
        self.asked = []

    def organize(self, sentences, visit_date):
        self.asked.append((sentences, visit_date))
        if self.error:
            raise self.error
        return self.raw


@pytest.fixture
def use(monkeypatch):
    monkeypatch.setattr(settings, "internal_token", TOKEN)

    def _use(organizer):
        app.dependency_overrides[get_note_organizer] = lambda: organizer
        return TestClient(app)

    yield _use
    app.dependency_overrides.clear()


def post(client, memo=MEMO, headers=HEADERS):
    return client.post(
        "/internal/visit-note/organize",
        json={"memo": memo, "visitDate": "2026-11-14"},
        headers=headers,
    )


def test_토큰이_없으면_401(use):
    assert post(use(FakeOrganizer({})), headers={}).status_code == 401


def test_키가_없으면_503(use):
    assert post(use(None)).status_code == 503


def test_GPT_실패나_이상한_답은_502(use):
    assert post(use(FakeOrganizer(error=OpenAIError("timeout")))).status_code == 502
    assert post(use(FakeOrganizer(error=ValueError("bad json")))).status_code == 502


def test_약_이야기뿐이면_GPT를_안_부른다(use):
    organizer = FakeOrganizer({})
    response = post(use(organizer), memo="인슐린은 자기 전 4단위.")
    assert response.status_code == 200
    assert response.json()["insulinSkipped"] is True
    assert organizer.asked == []


def test_찾은_값과_근거를_Spring이_읽는_모양으로(use):
    raw = {"fasting_target": {"value": 95, "evidence": "공복은 95 미만이래요.", "uncertain": False}}
    organizer = FakeOrganizer(raw)
    response = post(use(organizer))

    assert response.status_code == 200
    assert response.json() == {
        "measureSlots": None,
        "fastingTarget": {"value": 95, "evidence": "공복은 95 미만이래요.", "status": "FOUND"},
        "postMealTarget": None,
        "postMealMinutes": None,
        "nextVisitDate": None,
        "insulinSkipped": True,
        "promptVersion": "visit-note-v1",
    }
    sentences, _ = organizer.asked[0]
    assert sentences == ["공복은 95 미만이래요."]  # 인슐린 문장은 안 보냄
