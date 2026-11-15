"""OpenAI 클라이언트 한 곳 (위반 감지 · 챗봇 · 진료 메모 정리가 같이 쓴다).

처음 부를 때 만든다(@lru_cache). 그래서 키가 없어도 서버·CI는 뜬다
"""

from functools import lru_cache

from openai import OpenAI

from app.core.config import settings


@lru_cache
def get_openai_client() -> OpenAI | None:
    if not settings.llm_api_key:  # 키가 없으면 LLM 없이 동작하는 기능만 쓴다
        return None
    # 재시도는 하지 않는다: 늦으면 규칙 결과로 넘어가는 편이 사용자에게 낫다
    return OpenAI(api_key=settings.llm_api_key, timeout=4.0, max_retries=0)
