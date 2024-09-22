from fastapi import FastAPI, Body
from contextlib import asynccontextmanager
from ml.pipeline import model_pipeline
import pandas as pd
from pydantic import BaseModel
from typing import List
import yaml
import json
from pathlib import Path

ml_models = {}

class ImportancePrediction(BaseModel):
    prediction: int

class InputData(BaseModel):
    price: float
    qty: float
    isBuyerMaker: int
    isBestMatch: int
    percent_to_1000: int
    aggregated_trades: int
    price_seen_before: int

@asynccontextmanager
async def lifespan(app: FastAPI):
    # before app start: Load the ML model
    config_path = Path(__file__).parent.parent / "configs/config.yaml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        model_name = config.get("model")

    ml_models[model_name] = model_pipeline
    yield
    # after app closed: Clean up the ML models and release the resources
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

@app.post("/predict", response_model=ImportancePrediction)
async def predict(data: InputData = Body(...), model_name: str = Body(...)):
    data_df = pd.DataFrame([data.dict()])
    data_df.columns=['price','qty','isBuyerMaker','isBestMatch','percent_to_1000','aggregated_trades','price_seen_before']
    result = ml_models[model_name](data_df, model_name)
    return ImportancePrediction(prediction=result)

# TODO: not with Body?