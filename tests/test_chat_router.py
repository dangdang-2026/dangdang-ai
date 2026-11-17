from datetime import date

import pytest
from fastapi.testclient import TestClient
from openai import OpenAIError

from app.chat.brain import ChatReply, Records
from app.chat.router import get_chat_brain
from app.core.config import settings
from app.main import app

TOKEN = "test-token"
HEADERS = {"X-Internal-Token": TOKEN, "X-Request-Id": "req-1"}


class FakeBrain:
    """진짜 GPT 없이: 정해 둔 답을 돌려주고, 받은 것을 기억한다."""

    def __init__(self, reply=None, error=None):
        self.reply = reply
        self.error = error
        self.asked = []

    def decide(self, message, history, records_allowed, today):
        self.asked.append((message, history, records_allowed, today))
        if self.error:
            raise self.error
        return self.reply


@pytest.fixture
def use(monkeypatch):
    monkeypatch.setattr(settings, "internal_token", TOKEN)

    def _use(brain):
        app.dependency_overrides[get_chat_brain] = lambda: brain
        return TestClient(app)

    yield _use
    app.dependency_overrides.clear()


def test_토큰이_없으면_401(use):
    client = use(FakeBrain(ChatReply(kind="UNKNOWN")))
    assert client.post("/internal/chat", json={"message": "안녕"}).status_code == 401


def test_키가_없으면_503(use):
    client = use(None)
    response = client.post("/internal/chat", json={"message": "안녕"}, headers=HEADERS)
    assert response.status_code == 503


def test_GPT가_실패하면_502(use):
    client = use(FakeBrain(error=OpenAIError("Request timed out.")))
    response = client.post("/internal/chat", json={"message": "안녕"}, headers=HEADERS)
    assert response.status_code == 502


def test_빈_말이나_긴_말은_422(use):
    client = use(FakeBrain(ChatReply(kind="UNKNOWN")))
    assert client.post("/internal/chat", json={"message": ""}, headers=HEADERS).status_code == 422
    too_long = {"message": "가" * 501}
    assert client.post("/internal/chat", json=too_long, headers=HEADERS).status_code == 422


def test_기록_답은_Spring이_읽는_모양으로(use):
    reply = ChatReply(
        kind="RECORDS", records=Records(from_=date(2026, 11, 7), to=date(2026, 11, 13), slot=None)
    )
    brain = FakeBrain(reply)
    client = use(brain)
    body = {
        "message": "지난주 기록",
        "recordsAllowed": True,
        "history": [{"role": "user", "text": "앞 질문"}],
    }
    response = client.post("/internal/chat", json=body, headers=HEADERS)

    assert response.status_code == 200
    assert response.json() == {
        "kind": "RECORDS",
        "text": None,
        "records": {"from": "2026-11-07", "to": "2026-11-13", "slot": None},
        "questionDraft": None,
        "sources": [],
        "promptVersion": "chat-v1",
    }
    message, history, records_allowed, _ = brain.asked[0]
    assert message == "지난주 기록"
    assert records_allowed is True
    assert history[0].text == "앞 질문"
