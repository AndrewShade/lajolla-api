from contextlib import asynccontextmanager

from fastapi import FastAPI

from .predictor import load_models, run_prediction, validate_models
from .schemas import PredictionRequest, PredictionResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    validate_models()
    yield


app = FastAPI(
    title="La Jolla Cove Visibility API",
    description="Predicts underwater visibility at La Jolla Cove from ocean and weather conditions.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    return run_prediction(request)
