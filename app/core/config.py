from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """환경변수(.env)에서 읽는 설정. 값은 코드에 쓰지 않는다."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "dangdang-ai"
    internal_token: str = ""  # Spring → AI 서버 호출 확인용
    llm_api_key: str = ""  # 생성형 AI API 키
    llm_model: str = (
        "gpt-5.4-nano"  # 위반 감지·챗봇처럼 가벼운 일 (모델 이름은 .env LLM_MODEL로 바꾼다)
    )
    llm_model_note: str = (
        "gpt-5.4-mini"  # 진료 메모 정리 — 근거 문장을 정확히 옮겨야 해서 nano보다 한 단계 위
    )
    # 음식 이름 매칭용 (foods.name_embedding). dimensions를 지원하는 text-embedding-3 계열만
    # (차원 1536은 DB 칸과 맞춰 코드에 고정: app/embedding/embedder.py)
    embedding_model: str = "text-embedding-3-small"
    food_model_path: str = "models/food.onnx"  # 4단계에서 만든 모델
    food_labels_path: str = "models/labels.json"  # 모델 번호 → 음식 이름


settings = Settings()
