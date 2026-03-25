"""
Simulate real-time requests to the running FastAPI model and log predictions.

Usage:
    python simulate.py

This will:
- Load Paris Airbnb data
- Send random samples to the /predict endpoint
- Collect predictions + ground truth prices
- Append them to data/predictions.csv
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent
API_URL = "http://localhost:9696/predict"
LOG_PATH = PROJECT_ROOT / "data" / "predictions.csv"
DATA_PATH = PROJECT_ROOT / "data" / "paris_airbnb_clean.csv"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_data(n_rows: int = 100) -> pd.DataFrame:
    """Load a sample of Paris Airbnb data and apply training-time filters."""
    print(f"Loading data from {DATA_PATH}")
    dataframe = pd.read_csv(DATA_PATH)

    dataframe = dataframe[
        (dataframe["Price"] >= 10) & (dataframe["Price"] <= 2000)
    ].copy()

    dataframe = dataframe.sample(
        n=min(n_rows, len(dataframe)),
        random_state=42,
    ).reset_index(drop=True)

    print(f"Loaded {len(dataframe)} rows for simulation")
    return dataframe


def _to_bool(value) -> bool:
    """Convert CSV boolean-ish values to Python bool."""
    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return False

    if isinstance(value, str):
        normalised_value = value.strip().lower()
        if normalised_value in {"true", "1", "yes"}:
            return True
        if normalised_value in {"false", "0", "no"}:
            return False

    if isinstance(value, (int, float)):
        return bool(value)

    return bool(value)


def _to_int(value) -> int:
    """Convert numeric-ish values to int."""
    return int(value)


def _to_float(value) -> float:
    """Convert numeric-ish values to float."""
    return float(value)


def build_payload(row: pd.Series) -> dict:
    """Map raw CSV columns to the FastAPI request schema."""
    return {
        "room_type": str(row["Room Type"]),
        "day": str(row["Day"]),
        "person_capacity": _to_int(row["Person Capacity"]),
        "bedrooms": _to_int(row["Bedrooms"]),
        "superhost": _to_bool(row["Superhost"]),
        "shared_room": _to_bool(row["Shared Room"]),
        "private_room": _to_bool(row["Private Room"]),
        "multiple_rooms": _to_int(row["Multiple Rooms"]),
        "business": _to_int(row["Business"]),
        "cleanliness_rating": _to_float(row["Cleanliness Rating"]),
        "guest_satisfaction": _to_float(row["Guest Satisfaction"]),
        "city_center_km": _to_float(row["City Center (km)"]),
        "metro_distance_km": _to_float(row["Metro Distance (km)"]),
        "normalised_attraction_index": _to_float(
            row["Normalised Attraction Index"]
        ),
        "normalised_restaurant_index": _to_float(
            row["Normalised Restraunt Index"]
        ),
    }


def simulate_requests(
    dataframe: pd.DataFrame,
    sleep_seconds: float = 0.05,
) -> pd.DataFrame:
    """Send each row to the prediction API and log the results."""
    logged_rows = []

    for row_index, row in dataframe.iterrows():
        payload = build_payload(row)

        try:
            response = requests.post(API_URL, json=payload, timeout=5)

            if response.status_code != 200:
                print(
                    f"Request failed for row {row_index} "
                    f"with status {response.status_code}"
                )
                print(response.text)
                continue

            response_json = response.json()
            predicted_price = float(response_json["price"])
            model_version = response_json.get("model_version", "unknown")
            actual_price = float(row["Price"])

            logged_rows.append(
                {
                    "ts": pd.Timestamp.utcnow().isoformat(),
                    "prediction": predicted_price,
                    "actual_price": actual_price,
                    "abs_error": abs(predicted_price - actual_price),
                    "model_version": model_version,
                    **payload,
                }
            )

        except Exception as error:
            print(f"Request failed for row {row_index}: {error}")

        if (row_index + 1) % 20 == 0:
            print(f"Progress: {row_index + 1}/{len(dataframe)}")

        time.sleep(sleep_seconds)

    return pd.DataFrame(logged_rows)


def main() -> None:
    print("\nStarting simulation...\n")
    dataframe = load_data(n_rows=100)
    output_dataframe = simulate_requests(dataframe)

    if output_dataframe.empty:
        print("No predictions recorded. Make sure app.py is running.")
        return

    if LOG_PATH.exists():
        previous_dataframe = pd.read_csv(LOG_PATH)
        output_dataframe = pd.concat(
            [previous_dataframe, output_dataframe],
            ignore_index=True,
        )

    output_dataframe.to_csv(LOG_PATH, index=False)
    print(f"Wrote {len(output_dataframe)} total rows to {LOG_PATH}")


if __name__ == "__main__":
    main()
