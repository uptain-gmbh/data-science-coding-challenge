from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import json
from inference import preprocess_data, predict

# Load the model and LabelEncoder
model_filename = 'best_lightgbm_classification_model.pkl'
model = joblib.load(model_filename)
le = joblib.load('label_encoder.pkl')

# Initialize FastAPI
app = FastAPI()

class EmailRequest(BaseModel):
    email: str

class BatchRequest(BaseModel):
    data_path: str


@app.post("/predict/")
async def single_prediction(request: EmailRequest):
    email = request.email
    data = {'email': [email]}
    df = pd.DataFrame(data)
    df = preprocess_data(df)
    result = predict(df)[0]
    return result

@app.post("/predict_batch/")
async def batch_prediction(request: BatchRequest):
    data_path = request.data_path
    df = pd.read_csv(data_path)
    df = preprocess_data(df)
    results = predict(df)
    return results