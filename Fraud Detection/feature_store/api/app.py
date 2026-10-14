from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from api.predict import predict_fraud

app = FastAPI(title="Fraud Detection API")


class PredictionRequest(BaseModel):
    customer_id: int
    force_model: Optional[str] = None


@app.get("/")
def home():
    return {"message": "Real-Time Fraud Detection API"}


@app.post("/predict")
def predict(request: PredictionRequest):
    return predict_fraud(request.customer_id, force_model=request.force_model)
