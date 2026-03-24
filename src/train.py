"""
Train Paris Airbnb Price model and package for deployment.
"""

from __future__ import annotations

from pathlib import Path
import shutil

import mlflow
import mlflow.sklearn
import pandas as pd
import xgboost as xgb
import tomli as tomllib
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "paris_airbnb_clean.csv"
DEPLOYMENT_MODEL_PATH = PROJECT_ROOT / "models" / "model"

# Columns used as features (drops City and raw/unnormalised index columns)
FEATURE_COLS = [
    "Day",
    "Room Type",
    "Person Capacity",
    "Superhost",
    "Shared Room",
    "Private Room",
    "Multiple Rooms",
    "Business",
    "Cleanliness Rating",
    "Guest Satisfaction",
    "Bedrooms",
    "City Center (km)",
    "Metro Distance (km)",
    "Normalised Attraction Index",
    "Normalised Restraunt Index",
]

# Boolean columns stored as "True"/"False" strings that need to become 0/1
BOOL_COLS = ["Shared Room", "Private Room", "Superhost"]


def load_data() -> pd.DataFrame:
    """Load Paris Airbnb CSV and apply basic filters."""
    try:
        print("Loading data...")
        df = pd.read_csv(DATA_PATH)

        # Remove implausible prices (outliers below €10 or above €2 000)
        df = df[(df["Price"] >= 10) & (df["Price"] <= 2000)].copy()

        print(f"Loaded {len(df):,} rows after filtering")
    except Exception as e:
        print(f"Failed to load data: {e}")
        raise e
    return df


def prepare_features(df: pd.DataFrame):
    """Build feature dicts and extract target."""
    try:
        df = df.copy()

        # Convert "True"/"False" strings to integers so DictVectorizer treats
        # them as numeric rather than creating one-hot columns, fail if
        # unexpected values are found
        for col in BOOL_COLS:
            try:
                df[col] = df[col].map({True: 1, False: 0}).astype(int)
            except Exception as e:
                print(
                    f"Failed to map booleans to integers in column {col}, "
                    "inspecting and raising exception."
                )
                print(df[col].unique())
                print(df[col].value_counts(dropna=False))
                raise e

        features = df[FEATURE_COLS].to_dict(orient="records")
        target = df["Price"].values
    except Exception as e:
        print(f"Failed to prepare features for model training: {e}")
        raise e
    return features, target


def train_and_log(X_train, y_train, X_val, y_val) -> str:
    """Train model, log to MLflow, and save artifact for deployment."""
    print("Training model...")
    try:
        with open("utils/config.toml", "rb") as f:
            config = tomllib.load(f)

        mlflow.set_tracking_uri(config["mlflow"]["url"])
    except Exception as e:
        print(f"Failed to load config for MLflow: {e}")
        raise e

    mlflow.set_experiment("paris-airbnb-price")

    pipeline = Pipeline(
        [
            ("vectorizer", DictVectorizer(sparse=True)),
            (
                "regressor",
                xgb.XGBRegressor(
                    objective="reg:squarederror",
                    max_depth=6,
                    learning_rate=0.1,
                    n_estimators=300,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                ),
            ),
        ]
    )

    try:
        with mlflow.start_run() as run:
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_val)

            rmse = ((y_val - y_pred) ** 2).mean() ** 0.5
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            mlflow.sklearn.log_model(pipeline, "model")

            run_id = run.info.run_id
            print(f"Run ID:  {run_id}")
            print(f"RMSE:    {rmse:.2f}")
            print(f"MAE:     {mae:.2f}")
            print(f"R²:      {r2:.4f}")
    except Exception as e:
        print(f"Failed to train model and log to MLflow: {e}")
        raise e

    # Save the trained pipeline to the standard deployment path
    print("Creating deployment-ready model...")
    try:
        if DEPLOYMENT_MODEL_PATH.exists():
            shutil.rmtree(DEPLOYMENT_MODEL_PATH)

        mlflow.sklearn.save_model(pipeline, str(DEPLOYMENT_MODEL_PATH))
    except Exception as e:
        print(f"Failed to save model: {e}")
        raise e
    print(f"Model saved to: {DEPLOYMENT_MODEL_PATH}")

    try:
        with open("src/run_id.txt", "w") as f:
            f.write(run_id)
    except Exception as e:
        print(f"Failed to save run_id: {e}")
        raise e

    return run_id


def main() -> None:
    """Main training pipeline."""
    df = load_data()
    X, y = prepare_features(df)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    train_and_log(X_train, y_train, X_val, y_val)
    print("Training complete.")


if __name__ == "__main__":
    main()
