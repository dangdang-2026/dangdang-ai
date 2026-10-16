# dangdang-ai

당당 AI 서버 (FastAPI). Spring 백엔드가 내부 토큰으로 부르는 **판독실** 역할이다. 브라우저는 직접 부르지 않는다.
AI는 **기록 보조**만 한다 — 음식 이름 후보·진료 메모 정리. 혈당 판단·권고는 하지 않는다.

## 실행

```bash
brew install uv          # 처음 한 번
uv sync                  # 의존성 설치 (Python 3.11)
cp .env.example .env     # 값 채우기 (커밋 금지)
uv run uvicorn app.main:app --reload
# http://localhost:8000/health · http://localhost:8000/docs
```

## 검사

```bash
uv run ruff check . && uv run ruff format --check .
uv run pytest
```

## 구조

```
app/
├── main.py          # FastAPI 앱, /health
└── core/config.py   # 환경변수 설정 (pydantic-settings)
tests/               # pytest
docker/Dockerfile    # 비root · KST · HEALTHCHECK
```
