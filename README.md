# Paris Airbnb Price Predictor – CI/CD Pipeline

This module uses a CI/CD pipeline that automates the training, packaging, and testing of the Paris Airbnb Price Prediction service. The main workflow (`ci-cd.yml`) calls a reusable training job, builds a self-contained Docker image with the model baked in, and pushes it to the **GitHub Container Registry (GHCR)**.

Render is then used to pull this pre-built, validated image and run it as a live web service.

---

## 🔁 Workflow Concept

```
Git Push
    │
    ▼
┌──────────────────────────────────────────┐
│ CI/CD Pipeline (ci-cd.yml)               │
│                                          │
│  1. Calls train.yml → creates artifact   │
│  2. Lints & tests the code               │
│  3. Builds & tests Docker image          │
│  4. Pushes image to GHCR                 │
│                                          │
└──────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────┐
│ GitHub Container Registry (GHCR)         │
│                                          │
│  Stores the final, versioned image:      │
│  ghcr.io/<user>/<repo>:latest            │
│                                          │
└──────────────────────────────────────────┘
    │
    ▼ (Manual Deploy on Render)
┌──────────────────────────────────────────┐
│ Render.com                               │
│                                          │
│  Pulls the image from GHCR and runs it   │
│  as a live web service.                  │
│                                          │
└──────────────────────────────────────────┘
```

---

## 🗂️ Key Files

| File | Role |
| :--- | :--- |
| `data/paris_airbnb_clean.csv` | Source dataset (6,688 Paris Airbnb listings, 19 features). |
| `src/train.py` | Trains the XGBoost model, logs metrics to MLflow, and saves a production-ready copy to `models/model/`. |
| `src/app.py` | FastAPI service that loads `models/model/` at startup and serves price predictions. |
| `tests/test_api.py` | Automated tests run against the live Docker container to verify the API. |
| `.github/workflows/ci-cd.yml` | **The main orchestrator**: calls training, lints, builds, tests, and pushes the image. |
| `.github/workflows/train.yml` | A **reusable component** dedicated solely to running `train.py`. |
| `src/Dockerfile` | Packages the application code and trained model into a single container image. |
| `requirements.txt` | Production dependencies for the Docker container. |
| `.flake8` | Linting rules for the project. |
| `.gitignore` | Git ignore rules for the project. |

**Note on data:** Place `paris_airbnb_clean.csv` inside a `data/` folder at the root of the project before running `train.py`. The CSV is used only during training and is not required inside the Docker image.

---

## 🚀 First-Time Deployment Guide

Follow these steps in order to deploy the project to your own Render account.

### 1️⃣ Validate Your Code Locally (Pre-Flight Check)

Before pushing, run the linter locally to catch style errors early and prevent the CI/CD pipeline from failing.

```bash
flake8 .
```

If this command shows no output, you are good to go!

### 2️⃣ Commit and Push Your Code

```bash
git add .
git commit -m "feat: Finalize CI/CD pipeline for Paris Airbnb predictor"
git push origin main
```

This push will automatically trigger the CI/CD pipeline.

### 3️⃣ Wait for the Pipeline to Succeed

Go to your repository's **Actions** tab and wait for the **CI/CD Pipeline** to complete. This first run trains the model, builds the Docker image, and pushes it to GHCR — it will be **private** by default.

### 4️⃣ Make the Docker Image Public (One-Time Action)

You must make the image package public so Render can pull it.

1. On your repository's main page, go to the **Packages** section on the right sidebar.
2. Click on your image name (e.g., `paris-airbnb-predictor`).
3. Go to **Package settings**.
4. Scroll to the "Danger Zone" and change the visibility to **Public**.

### 5️⃣ Create the Render Service

1. On the Render Dashboard, click **New → Web Service**.
2. Choose **"Deploy an existing image from a registry"**.
3. For the Image URL, enter `ghcr.io/<your-github-user>/<your-repo-name>` (all lowercase).
4. Give the service a name, select the **Free** instance type, and click **Create Web Service**.

### 6️⃣ Verify Your Deployment

Once Render is live (≈1–2 min), check the health endpoint. You can find your service's URL on the Render dashboard.

```bash
curl https://<your-service-name>.onrender.com/health
```

The response should show `"status": "ok"` and `"model_loaded": true`.

---

## 🔮 Making a Prediction

Send a POST request to `/predict` with a listing's features:

```bash
curl -X POST https://<your-service-name>.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "room_type": "Private room",
    "day": "Weekday",
    "person_capacity": 2,
    "bedrooms": 1,
    "superhost": true,
    "shared_room": false,
    "private_room": true,
    "multiple_rooms": 0,
    "business": 0,
    "cleanliness_rating": 9.0,
    "guest_satisfaction": 92.0,
    "city_center_km": 1.5,
    "metro_distance_km": 0.3,
    "normalised_attraction_index": 25.0,
    "normalised_restaurant_index": 65.0
  }'
```

Example response:

```json
{
  "price": 143.27,
  "model_version": "a1b2c3d4e5f6..."
}
```

---

## 🔄 How to Redeploy with Changes

1. Make your code changes (`train.py`, `app.py`, etc.).
2. Run `flake8 .` locally to check for issues.
3. `git commit` and `git push` to `main`.
4. Wait for the **CI/CD Pipeline** to complete on GitHub Actions.
5. Go to your Render dashboard, find your service, and click **Manual Deploy → Deploy latest commit**.
