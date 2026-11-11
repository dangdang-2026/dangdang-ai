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
## 배포

- `main`에 머지되면 `.github/workflows/cd.yml`이 이미지를 만들어 Docker Hub(`DOCKER_REPO`)에 올린다 (`latest` · 커밋 sha)
- 서버에서 켜는 건 dangdang-be `docker/docker-compose.yml`의 `ai` 서비스. 밖으로 포트를 열지 않고, Spring만 `http://ai:8000`으로 부른다
- 모델 파일은 깃 · 이미지에 넣지 않는다 → 서버 `~/dangdang/ai-models/`에 두고 컨테이너 `/app/models`에 읽기 전용으로 붙인다
- 환경변수 `INTERNAL_TOKEN` — Spring과 같은 값 (dangdang-be Secret `ENV_PROD` → 서버 `.env`). 값은 깃 · PR · 노션에 쓰지 않는다

```bash
# 모델을 다시 학습했을 때 (맥에서)
scp models/food.onnx models/labels.json dangdang:~/dangdang/ai-models/
ssh dangdang 'cd ~/dangdang/docker && docker compose restart ai'

# 새 이미지로 바꿀 때 (main 머지 → Actions CD 초록 뒤)
ssh dangdang 'cd ~/dangdang/docker && docker compose pull ai && docker compose up -d ai'
```

## 구조

```
app/
├── main.py          # FastAPI 앱, /health
└── core/config.py   # 환경변수 설정 (pydantic-settings)
tests/               # pytest
docker/Dockerfile    # 비root · KST · HEALTHCHECK
```
