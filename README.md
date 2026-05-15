# La Jolla Cove Visibility API

REST API for predicting underwater visibility conditions at La Jolla Cove. Trained on historical CDIP buoy, NOAA tide/wind, and OpenWeatherMap rain data, the model returns a continuous visibility estimate alongside a Go/No-Go decision and a four-class condition rating.

**Live endpoint:** `https://fpg7dmau5ry3d7s63gtj4t5jlu0gqanc.lambda-url.us-west-2.on.aws/docs`

---

## Architecture

```
FastAPI app  →  Docker image  →  AWS ECR  →  AWS Lambda (container)
```

- **FastAPI** serves predictions via a single `/predict` endpoint and exposes interactive docs at `/docs`
- **Docker** packages the app and three XGBoost model artifacts into a portable image
- **ECR** stores the image at `935171706543.dkr.ecr.us-west-2.amazonaws.com/lajolla-api`
- **Lambda** runs the container via Mangum with a public Function URL

> For production workloads with sustained traffic, ECS Express Mode would replace Lambda as the serving layer.

---

## Local development

**Requirements:** Python 3.12, pip

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The API starts at `http://localhost:8000`. Interactive docs are at `http://localhost:8000/docs`.

On startup, all three models load and a sanity check runs a dummy prediction through each one. If any model output is non-finite or out of range the process exits immediately rather than serving bad predictions.

---

## Running with Docker

**Requirements:** Docker Desktop

```bash
# Build
docker build -t lajolla-api .

# Run
docker run -p 8000:8000 lajolla-api
```

The container runs as a non-root user. The `HEALTHCHECK` in the Dockerfile pings `/health` every 30 seconds so Docker can detect and restart unhealthy containers automatically.

---

## API

### `GET /health`
Returns `{"status": "ok"}`. Used by Docker and Lambda for health probing.

### `POST /predict`
Returns visibility predictions from all three models given ocean and weather conditions.

**Required fields:**

| Field | Type | Description |
|-------|------|-------------|
| `wave_height_max_m` | float | Maximum significant wave height (m) |
| `wave_height_mean_m` | float | Mean significant wave height (m) |
| `wave_period_mean_s` | float | Mean peak wave period (s) |
| `wave_direction_deg` | float | Mean wave direction (0–360°) |
| `wind_speed_ms` | float | Mean wind speed (m/s) |
| `wind_gust_ms` | float | Peak wind gust (m/s) |
| `wind_dir_mean_deg` | float | Mean wind direction (0–360°) |
| `tide_max_m` | float | Maximum tidal height, MLLW datum (m) |
| `tide_mean_m` | float | Mean tidal height, MLLW datum (m) |

**Optional fields:** `rain_today_mm`, `rain_yesterday_mm`, `rain_two_days_ago_mm`, `forecast_date` (YYYY-MM-DD), plus hindcast features (SST, PSD, lag values) that default to 0 when omitted.

**Response:**

```json
{
  "visibility_feet": 18.4,
  "go_no_go": false,
  "go_probability": 0.3812,
  "condition": "Fair",
  "condition_probabilities": {
    "Poor": 0.1823,
    "Fair": 0.5341,
    "Good": 0.2187,
    "Excellent": 0.0649
  },
  "hindcast_features_used": false
}
```

`hindcast_features_used` indicates whether SST, PSD, or lag features were provided. When `false`, those 21 features are zeroed and prediction quality is reduced.

Full schema and a live request builder are available at `/docs`.

---

## Deploying an update

After changing app code or model files:

```bash
# Authenticate Docker to ECR (token valid 12 hours)
aws ecr get-login-password --region us-west-2 | \
  docker login --username AWS --password-stdin 935171706543.dkr.ecr.us-west-2.amazonaws.com

# Build for Lambda's architecture, push directly to ECR
docker buildx build --platform linux/amd64 --provenance=false \
  -t 935171706543.dkr.ecr.us-west-2.amazonaws.com/lajolla-api:latest \
  --push .

# Tell Lambda to pull the new image
aws lambda update-function-code \
  --function-name lajolla-api \
  --image-uri 935171706543.dkr.ecr.us-west-2.amazonaws.com/lajolla-api:latest \
  --region us-west-2
```

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_DIR` | `models/` | Path to the directory containing the three model JSON files |
| `GO_THRESHOLD` | `0.5` | Probability threshold for the binary Go/No-Go decision |
