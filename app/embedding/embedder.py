"""OpenAI 임베딩으로 음식 이름 여러 개를 한 번에 벡터로 바꾼다.

- 이름 100개도 요청 1번 (비용·시간 절약)
- 받은 벡터는 index로 다시 줄 세운다 → 보낸 순서 그대로
- 오류(OpenAIError)·이상한 답(ValueError)은 그대로 올린다 → router가 502
"""

from openai import OpenAI

TIMEOUT_SECONDS = 15.0  # 100개 묶음이라 기본 4초보다 넉넉히
EMBEDDING_DIM = 1536  # Spring DB 칸 vector(1536)과 같아야 한다 (바꾸면 DB도 같이)


class OpenAIEmbedder:
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def embed(self, texts: list[str]) -> tuple[list[list[float]], int | None]:
        """벡터들(보낸 순서)과 쓴 토큰 수를 돌려준다."""
        resp = self.client.embeddings.create(
            model=self.model,
            input=texts,
            dimensions=EMBEDDING_DIM,
            timeout=TIMEOUT_SECONDS,
        )
        rows = sorted(resp.data, key=lambda d: d.index)
        if len(rows) != len(texts):
            raise ValueError(f"벡터 개수가 다름 {len(rows)}/{len(texts)}")
        vectors = [list(d.embedding) for d in rows]
        if any(
            len(v) != EMBEDDING_DIM for v in vectors
        ):  # DB 칸이 vector(1536)이라 다르면 못 넣는다
            raise ValueError("벡터 차원이 다름")
        usage = getattr(resp, "usage", None)
        return vectors, getattr(usage, "total_tokens", None)
