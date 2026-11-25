import logging
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from openai import APITimeoutError, OpenAIError

from app.core.config import settings
from app.embedding.embedder import TIMEOUT_SECONDS, OpenAIEmbedder
from app.embedding.router import get_embedder
from app.main import app

TOKEN = "test-token"
HEADERS = {"X-Internal-Token": TOKEN, "X-Request-Id": "req-1"}
DIM = 1536


class FakeEmbeddings:
    """OpenAI embeddings 흉내: 순서를 뒤집어서 준다 (index로 다시 세우는지 보려고)."""

    def __init__(self, dim=DIM, drop=0, error=None):
        self.dim = dim
        self.drop = drop
        self.error = error
        self.calls = []

    def create(self, model, input, dimensions, timeout):
        self.calls.append(
            {"model": model, "input": input, "dimensions": dimensions, "timeout": timeout}
        )
        if self.error:
            raise self.error
        data = [
            SimpleNamespace(index=i, embedding=[float(i)] * self.dim)
            for i in range(len(input) - self.drop)
        ]
        return SimpleNamespace(data=list(reversed(data)), usage=SimpleNamespace(total_tokens=7))


class FakeClient:
    def __init__(self, **kwargs):
        self.embeddings = FakeEmbeddings(**kwargs)


@pytest.fixture
def use(monkeypatch):
    monkeypatch.setattr(settings, "internal_token", TOKEN)

    def _use(embedder):
        app.dependency_overrides[get_embedder] = lambda: embedder
        return TestClient(app)

    yield _use
    app.dependency_overrides.clear()


def openai_embedder(**kwargs):
    client = FakeClient(**kwargs)
    return OpenAIEmbedder(client, "text-embedding-3-small"), client


def post(client, texts, headers=HEADERS):
    return client.post("/internal/embeddings", json={"texts": texts}, headers=headers)


def test_보낸_순서대로_벡터를_돌려준다(use):
    embedder, client = openai_embedder()
    response = post(use(embedder), ["김치찌개", "된장국", "달걀말이"])

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "text-embedding-3-small"
    assert body["dim"] == DIM
    assert [v[0] for v in body["vectors"]] == [0.0, 1.0, 2.0]
    assert all(len(v) == DIM for v in body["vectors"])


def test_여러_이름을_한_번에_보낸다(use):
    embedder, client = openai_embedder()
    post(use(embedder), ["김치찌개", "된장국"])

    assert len(client.embeddings.calls) == 1
    call = client.embeddings.calls[0]
    assert call["input"] == ["김치찌개", "된장국"]
    assert call["dimensions"] == DIM
    assert call["timeout"] == TIMEOUT_SECONDS


def test_앞뒤_빈칸은_지우고_너무_긴_이름은_자른다(use):
    embedder, client = openai_embedder()
    response = post(use(embedder), ["  김치찌개 ", "가" * 300])

    assert response.status_code == 200
    sent = client.embeddings.calls[0]["input"]
    assert sent[0] == "김치찌개"
    assert len(sent[1]) == 200


def test_폭_없는_글자만_있으면_422_이고_섞여_있으면_지운다(use):
    embedder, client = openai_embedder()
    c = use(embedder)
    assert post(c, ["\u200b\ufeff"]).status_code == 422
    assert post(c, [" \u200d\u2060 "]).status_code == 422
    assert client.embeddings.calls == []

    assert post(c, ["\ufeff김치\u200b찌개"]).status_code == 200
    assert client.embeddings.calls[0]["input"] == ["김치찌개"]


def test_토큰이_없거나_틀리면_401(use):
    client = use(openai_embedder()[0])
    assert post(client, ["김치찌개"], headers={}).status_code == 401
    assert post(client, ["김치찌개"], headers={"X-Internal-Token": "wrong"}).status_code == 401


def test_키가_없으면_503(use):
    assert post(use(None), ["김치찌개"]).status_code == 503


@pytest.mark.parametrize(
    "texts",
    [[], [" "], ["김치찌개", ""], ["가"] * 101, [1]],
    ids=["0개", "빈칸", "빈_이름_섞임", "101개", "글자가_아님"],
)
def test_이상한_입력은_422_이고_OpenAI를_안_부른다(use, texts):
    embedder, client = openai_embedder()
    assert post(use(embedder), texts).status_code == 422
    assert client.embeddings.calls == []


def test_100개까지는_된다(use):
    embedder, _ = openai_embedder()
    response = post(use(embedder), ["김치찌개"] * 100)
    assert response.status_code == 200
    assert len(response.json()["vectors"]) == 100


@pytest.mark.parametrize(
    "kwargs",
    [
        {"error": APITimeoutError(request=None)},
        {"error": OpenAIError("rate limit")},
        {"dim": 3},  # 차원이 DB 칸과 다름
        {"drop": 1},  # 보낸 것보다 적게 옴
    ],
    ids=["시간_초과", "OpenAI_오류", "차원_다름", "개수_다름"],
)
def test_OpenAI_실패나_이상한_답은_502(use, kwargs):
    embedder, _ = openai_embedder(**kwargs)
    response = post(use(embedder), ["김치찌개", "된장국"])
    assert response.status_code == 502
    assert response.json() == {"detail": "임베딩 호출 실패"}


def test_로그에_음식_이름은_남기지_않는다(use, caplog):
    embedder, _ = openai_embedder()
    with caplog.at_level(logging.INFO, logger="app.embedding.router"):
        post(use(embedder), ["김치찌개", "된장국"])

    assert "개수=2" in caplog.text
    assert "김치찌개" not in caplog.text and "된장국" not in caplog.text
