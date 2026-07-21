from fastapi import FastAPI
from src.api.schema import StrokeRequest
from src.api.predict import PredictStroke


app = FastAPI(title="Stroke Risk Prediction API")
@app.get("/")
def health():
    return {"message":"API Running"}

@app.post("/predict")
def predict(request: StrokeRequest):
    result = PredictStroke(request.model_dump())
    return result   