from fastapi import FastAPI
from ml.model import load_model

model=None
app = FastAPI()

@app.on_event("startup")
def startup_event():
    global model
    model=load_model()

@app.get("/predict")
async def root(data_test):
    prediction=model(data_test)

    return prediction
