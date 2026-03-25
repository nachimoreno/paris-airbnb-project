"""
FastAPI service for Paris Airbnb Price Prediction.

Model is baked into the Docker image at build time via:
    models/model/   (saved by train.py)
    run_id.txt      (written by train.py)
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# Global state
RUN_ID: Optional[str] = None
model: Optional[Any] = None


# --- Pydantic models ---------------------------------------------------------

class ListingRequest(BaseModel):
    room_type: str = Field(
        ..., description="'Private room', 'Entire home/apt', or 'Shared room'"
    )
    day: str = Field(
        ..., description="'Weekday' or 'Weekend'"
    )
    person_capacity: int = Field(
        ..., ge=1, description="Maximum number of guests"
    )
    bedrooms: int = Field(
        ..., ge=0, description="Number of bedrooms"
    )
    superhost: bool = Field(
        ..., description="Host has superhost status"
    )
    shared_room: bool = Field(
        ..., description="Listing is a shared room"
    )
    private_room: bool = Field(
        ..., description="Listing is a private room"
    )
    multiple_rooms: int = Field(
        ..., ge=0, description="Host has multiple listings (0 or 1)"
    )
    business: int = Field(
        ..., ge=0, description="Business listing flag (0 or 1)"
    )
    cleanliness_rating: float = Field(
        ..., ge=0, le=10, description="Cleanliness score out of 10"
    )
    guest_satisfaction: float = Field(
        ..., ge=0, le=100, description="Overall guest satisfaction score"
    )
    city_center_km: float = Field(
        ..., ge=0, description="Distance to city centre in km"
    )
    metro_distance_km: float = Field(
        ..., ge=0, description="Distance to nearest metro in km"
    )
    normalised_attraction_index: float = Field(
        ..., description="Normalised attraction index"
    )
    normalised_restaurant_index: float = Field(
        ..., description="Normalised restaurant index"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "room_type": "Private room",
                "day": "Weekday",
                "person_capacity": 2,
                "bedrooms": 1,
                "superhost": True,
                "shared_room": False,
                "private_room": True,
                "multiple_rooms": 0,
                "business": 0,
                "cleanliness_rating": 9.0,
                "guest_satisfaction": 92.0,
                "city_center_km": 1.5,
                "metro_distance_km": 0.3,
                "normalised_attraction_index": 25.0,
                "normalised_restaurant_index": 65.0,
            }
        }


class PredictionResponse(BaseModel):
    price: float
    model_version: str


# --- Lifespan: load model at startup -----------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global RUN_ID, model

    run_id_path = Path("run_id.txt")
    if run_id_path.exists():
        RUN_ID = run_id_path.read_text().strip()
        print(f"[startup] Found run_id: {RUN_ID}")
    else:
        print("[startup] run_id.txt not found – health will report 'unknown'.")
        RUN_ID = None

    model_dir = Path("models/model")
    if model_dir.exists():
        try:
            model = mlflow.sklearn.load_model(str(model_dir))
            print(f"[startup] Model loaded from {model_dir}")
        except Exception as e:
            print(f"[startup] Failed to load model: {e}")
            model = None
    else:
        print(f"[startup] Model directory not found: {model_dir}")
        model = None

    yield


# --- App ---------------------------------------------------------------------

app = FastAPI(
    title="Paris Airbnb Price Predictor",
    description="Predict a nightly listing price for a Paris Airbnb property.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {"message": "Welcome to the Paris Airbnb Price Prediction API"}


@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "degraded",
        "run_id": RUN_ID or "unknown",
        "model_loaded": model is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(listing: ListingRequest):
    if model is None:
        raise HTTPException(
            status_code=500, detail="Model not loaded. Check /health."
        )

    feature_dict = {
        "Day": listing.day,
        "Room Type": listing.room_type,
        "Person Capacity": listing.person_capacity,
        "Superhost": int(listing.superhost),
        "Shared Room": int(listing.shared_room),
        "Private Room": int(listing.private_room),
        "Multiple Rooms": listing.multiple_rooms,
        "Business": listing.business,
        "Cleanliness Rating": listing.cleanliness_rating,
        "Guest Satisfaction": listing.guest_satisfaction,
        "Bedrooms": listing.bedrooms,
        "City Center (km)": listing.city_center_km,
        "Metro Distance (km)": listing.metro_distance_km,
        "Normalised Attraction Index": listing.normalised_attraction_index,
        "Normalised Restraunt Index": listing.normalised_restaurant_index,
    }

    price = float(model.predict([feature_dict])[0])
    return PredictionResponse(
        price=round(price, 2),
        model_version=RUN_ID or "unknown"
    )


# Local dev
if __name__ == "__main__":
    import uvicorn
    try:
        uvicorn.run("app:app", host="0.0.0.0", port=9696, reload=True)
    except Exception as e:
        print(f"Failed to start app: {e}")
        raise e
