import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.moderation.router import get_dose_checker

TOKEN = "test-token"
HEADERS = {"X-Internal-Token": TOKEN}


class FakeChecker:
    """진짜 GPT 없이: 정해 둔 답을 돌려주고, 받은 문장을 기억한다."""

    def __init__(self, answer):
        self.answer = answer
        self.asked: list[str] = []

    def confirm_dose(self, sentences):
        self.asked.extend(sentences)
        return self.answer if self.answer is None else self.answer[: len(sentences)]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "internal_token", TOKEN)
    app.dependency_overrides[get_dose_checker] = lambda: None  # 기본: 규칙만
    yield TestClient(app)
    app.dependency_overrides.clear()


def check(client, content, title=""):
    response = client.post(
        "/internal/post-check", json={"title": title, "content": content}, headers=HEADERS
    )
    assert response.status_code == 200
    return response.json()


def test_토큰이_없으면_401(client):
    response = client.post("/internal/post-check", json={"content": "안녕하세요"})
    assert response.status_code == 401


def test_피그마_U15_예시는_약_용량으로_걸린다(client):
    body = check(
        client,
        "지난주부터 인슐린 8단위로 늘렸는데 다들 몇 단위 맞으세요? "
        "회사에서 점심 먹고 1시간 알림이 오면 눈치가 보여요.",
        title="식후 1시간 알림, 밖에서는 어떻게 재세요?",
    )
    assert [i["kind"] for i in body["issues"]] == ["DOSE"]
    assert body["issues"][0]["quote"].startswith("지난주부터 인슐린 8단위로")
    assert body["llmChecked"] is False


def test_혈당_수치와_음식_양은_약_용량이_아니다(client):
    body = check(
        client, "식후 혈당이 140mg/dL 나왔어요. 아침은 계란 2알이랑 현미밥 반 공기 먹었어요."
    )
    assert body["issues"] == []


def test_피그마_U30_예시는_위급으로_걸린다(client):
    body = check(client, "배가 자꾸 뭉치고 피가 조금 비쳐요. 다들 이럴 때 어떻게 하셨어요?")
    assert body["issues"] == [
        {"kind": "URGENT_MEDICAL", "quote": "배가 자꾸 뭉치고 피가 조금 비쳐요."}
    ]


def test_마음_위기_표현은_MIND로_걸린다(client):
    body = check(client, "요즘 너무 지쳐서 그냥 사라지고 싶어요")
    assert [i["kind"] for i in body["issues"]] == ["MIND"]


def test_GPT가_아니라고_하면_약_용량에서_뺀다(client):
    fake = FakeChecker(answer=[False])
    app.dependency_overrides[get_dose_checker] = lambda: fake
    body = check(client, "아침 혈당 95 나왔고 메트포르민 500mg 먹는 날이에요")
    assert body["issues"] == []
    assert body["llmChecked"] is True
    assert fake.asked == ["아침 혈당 95 나왔고 메트포르민 500mg 먹는 날이에요"]  # 걸린 문장만 보냄


def test_GPT가_실패하면_규칙_결과를_그대로_쓴다(client):
    app.dependency_overrides[get_dose_checker] = lambda: FakeChecker(answer=None)
    body = check(client, "인슐린 8단위 맞고 있어요")
    assert [i["kind"] for i in body["issues"]] == ["DOSE"]
    assert body["llmChecked"] is False


def test_위급_문장은_GPT에_묻지_않는다(client):
    fake = FakeChecker(answer=[False])
    app.dependency_overrides[get_dose_checker] = lambda: fake
    body = check(client, "어제부터 태동이 줄었어요")
    assert [i["kind"] for i in body["issues"]] == ["URGENT_MEDICAL"]
    assert fake.asked == []


def test_동의가_없으면_GPT에_안_묻고_규칙만(client):
    fake = FakeChecker([False])
    app.dependency_overrides[get_dose_checker] = lambda: fake
    response = client.post("/internal/post-check",
                           json={"content": "인슐린 4단위 맞아요.", "allowLlm": False}, headers=HEADERS)
    body = response.json()
    assert fake.asked == []
    assert body["llmChecked"] is False
    assert [i["kind"] for i in body["issues"]] == ["DOSE"]


def test_본문이_비면_422(client):
    response = client.post("/internal/post-check", json={"content": ""}, headers=HEADERS)
    assert response.status_code == 422
