from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from contextlib import asynccontextmanager
import uvicorn
import json

# Global variables for data and model
df = None
model = None
label_encoders = {}

# Lifespan handler (replaces @app.on_event)
@asynccontextmanager
async def lifespan(app: FastAPI):
    global df, model, label_encoders
    try:
        # Load the dataset
        df = pd.read_csv("bengaluru_house_prices_cleaned.csv")
        print(f"Loaded dataset with {len(df)} records")

        # Features and target
        feature_columns = ["area_type", "location_grouped", "bhk", "total_sqft", "bath", "balcony"]
        target_column = "price"

        model_df = df[feature_columns + [target_column]].copy().dropna()

        # Encode categorical columns
        for col in ["area_type", "location_grouped"]:
            le = LabelEncoder()
            model_df[col] = le.fit_transform(model_df[col].astype(str))
            label_encoders[col] = le

        # Train model
        X = model_df[feature_columns]
        y = model_df[target_column]
        model = LinearRegression()
        model.fit(X, y)

        print("Model trained successfully!")
    except Exception as e:
        print(f"Error loading data or training model: {e}")

    yield  # app runs here

    # Cleanup (if needed)
    print("Shutting down...")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Bengaluru House Price API",
    description="API for Bengaluru house price data analysis and prediction",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class HouseData(BaseModel):
    area_type: str
    location: str
    size: str
    total_sqft: float
    bath: float
    balcony: float

class PredictionRequest(BaseModel):
    area_type: str
    location: str
    bhk: int
    total_sqft: float
    bath: int
    balcony: int

class HouseResponse(BaseModel):
    area_type: str
    availability: str
    location: str
    size: str
    bhk: int
    society: str
    total_sqft: float
    bath: float
    balcony: float
    price: float
    price_per_sqft: float

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to Bengaluru House Price API",
        "total_records": len(df) if df is not None else 0,
        "endpoints": {
            "/docs": "API documentation",
            "/houses": "Get all houses with filters",
            "/locations": "Get all available locations",
            "/predict": "Predict house price",
            "/stats": "Get dataset statistics"
        }
    }

# Get all houses with optional filters
@app.get("/houses", response_model=List[dict])
async def get_houses(
    location: Optional[str] = Query(None, description="Filter by location"),
    bhk: Optional[int] = Query(None, description="Filter by number of BHK"),
    min_price: Optional[float] = Query(None, description="Minimum price in lakhs"),
    max_price: Optional[float] = Query(None, description="Maximum price in lakhs"),
    area_type: Optional[str] = Query(None, description="Filter by area type"),
    limit: int = Query(100, description="Number of records to return")
):
    if df is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded")

    filtered_df = df.copy()

    if location:
        filtered_df = filtered_df[filtered_df["location_grouped"] == location]
    if bhk:
        filtered_df = filtered_df[filtered_df["bhk"] == bhk]
    if min_price:
        filtered_df = filtered_df[filtered_df["price"] >= min_price]
    if max_price:
        filtered_df = filtered_df[filtered_df["price"] <= max_price]
    if area_type:
        filtered_df = filtered_df[filtered_df["area_type"] == area_type]

    return filtered_df.head(limit).to_dict(orient="records")
