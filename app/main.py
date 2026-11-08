from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app.core.config import settings
from app.food.predictor import FoodPredictor
from app.food.router import router as food_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = Path(settings.food_model_path)
    labels_path = Path(settings.food_labels_path)
    if model_path.exists() and labels_path.exists():  # 서버 켤 때 1번만 불러 둔다
        app.state.food_predictor = FoodPredictor(model_path, labels_path)
    else:  # 모델이 없어도 서버는 켜지고, 사진 주소만 503
        app.state.food_predictor = None
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.include_router(food_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
