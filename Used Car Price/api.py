from fastapi import FastAPI
import pickle
import pandas as pd
from pathlib import Path

app = FastAPI(title="Used Car Price Prediction API")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "car_price_model.pkl"

print("🔍 Looking for model at:", MODEL_PATH)

@app.on_event("startup")
def load_model():
    global model
    try:
        model = pickle.load(open(MODEL_PATH, "rb"))
        print("✅ Model loaded successfully")
    except Exception as e:
        print("❌ Failed to load model:", e)
        raise e

@app.post("/predict")
def predict_price(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return {"predicted_price": round(prediction, 2)}

