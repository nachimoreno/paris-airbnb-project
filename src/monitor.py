"""
Generate an Evidently HTML report comparing past vs recent predictions.

Usage:
    python monitor.py

This will:
- Load the logged predictions from data/predictions.csv
- Split them into reference (older) vs current (newer) data
- Generate a data drift + regression performance report
"""

from pathlib import Path

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, RegressionPreset
from evidently.report import Report


LOG_PATH = Path("data/predictions.csv")
REPORT_PATH = Path("monitoring_report.html")

NUMERICAL_FEATURES = [
    "person_capacity",
    "bedrooms",
    "multiple_rooms",
    "business",
    "cleanliness_rating",
    "guest_satisfaction",
    "city_center_km",
    "metro_distance_km",
    "normalised_attraction_index",
    "normalised_restaurant_index",
]

CATEGORICAL_FEATURES = [
    "room_type",
    "day",
    "superhost",
    "shared_room",
    "private_room",
    "model_version",
]


def main() -> None:
    print("\nStarting monitoring report...\n")

    if not LOG_PATH.exists():
        raise FileNotFoundError(
            "No logged predictions found. Run simulate.py first!"
        )

    dataframe = pd.read_csv(LOG_PATH, parse_dates=["ts"])
    dataframe = dataframe.dropna(subset=["prediction", "actual_price"])
    print(f"Loaded {len(dataframe)} logged predictions")

    if len(dataframe) < 2:
        raise ValueError(
            "Need at least 2 logged predictions to build a report."
        )

    dataframe = dataframe.sort_values("ts")
    midpoint = len(dataframe) // 2
    reference_data = dataframe.iloc[:midpoint].copy()
    current_data = dataframe.iloc[midpoint:].copy()

    if reference_data.empty or current_data.empty:
        raise ValueError(
            "Reference or current dataset is empty. "
            "Generate more predictions first."
        )

    print(
        f"Reference: {len(reference_data)} | Current: {len(current_data)}"
    )

    column_mapping = ColumnMapping(
        target="actual_price",
        prediction="prediction",
        numerical_features=NUMERICAL_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
    )

    print("\nGenerating Evidently drift report...")
    report = Report(metrics=[DataDriftPreset(), RegressionPreset()])
    report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping,
    )

    report.save_html(str(REPORT_PATH))
    print(f"Report saved: {REPORT_PATH.resolve()}")
    print("Open it in your browser to explore drift metrics.\n")


if __name__ == "__main__":
    main()
