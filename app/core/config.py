from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """환경변수(.env)에서 읽는 설정. 값은 코드에 쓰지 않는다."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "dangdang-ai"
    internal_token: str = ""  # Spring → AI 서버 호출 확인용
    llm_api_key: str = ""  # 생성형 AI API 키
    food_model_path: str = "models/food.onnx"  # 4단계에서 만든 모델
    food_labels_path: str = "models/labels.json"  # 모델 번호 → 음식 이름


settings = Settings()
