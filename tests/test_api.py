"""External API tests for the running FastAPI Paris Airbnb Price service.

Requires the server already running (e.g. `python app.py` showing Uvicorn
on port 9696).

These are deployment-level tests that issue real HTTP requests instead of
using FastAPI's in-process TestClient.
"""
import requests

BASE_URL = "http://127.0.0.1:9696"

SAMPLE_LISTING = {
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


def test_health_endpoint():
    resp = requests.get(f"{BASE_URL}/health")
    assert resp.status_code == 200, (
        f"Unexpected status: {resp.status_code} body={resp.text}"
    )
    data = resp.json()
    assert data.get("status") == "ok"
    assert isinstance(data.get("run_id"), str) and len(data["run_id"]) > 5


def test_predict_endpoint():
    resp = requests.post(f"{BASE_URL}/predict", json=SAMPLE_LISTING)
    assert resp.status_code == 200, (
        f"Unexpected status: {resp.status_code} body={resp.text}"
    )
    data = resp.json()

    # Validate response structure
    assert "price" in data and "model_version" in data
    # Sanity checks: price should be a positive number in a plausible range
    assert isinstance(data["price"], float)
    assert 10 < data["price"] < 2000
    assert isinstance(
        data["model_version"], str
    ) and len(
        data["model_version"]
    ) > 5


def test_predict_weekend():
    payload = {
        **SAMPLE_LISTING,
        "day": "Weekend",
        "room_type": "Entire home/apt"
    }
    resp = requests.post(f"{BASE_URL}/predict", json=payload)
    assert resp.status_code == 200, (
        f"Unexpected status: {resp.status_code} body={resp.text}"
    )
    data = resp.json()
    assert data["price"] > 0
