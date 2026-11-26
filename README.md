# Geospatial Weather Intelligence & Predictive Analytics Platform

A full-stack weather application where users select a point on a Leaflet map to view current conditions, an LSTM temperature forecast, nearby saved forecasts, and a seven-day multi-variable outlook.

## Features

- Interactive Angular and Leaflet map for selecting coordinates around the world.
- Current temperature, humidity, wind, rain, precipitation, and weather description from Open-Meteo.
- A PyTorch LSTM that uses the previous 168 hours of weather and location-aware features to predict temperature for the next 24 hours.
- A seven-day Open-Meteo outlook with daily high/low temperature, average humidity, maximum wind speed, rain total, precipitation total, and weather description.
- Django REST API backed by MongoDB, including GeoJSON 2dsphere indexes for nearby forecast queries.
- Historical data ingestion and model training/evaluation scripts for 53 locations.

> **Forecast source:** The LSTM currently predicts temperature only for the next 24 hours. The seven-day forecasts for humidity, wind, rain, precipitation, and daily conditions are provided by Open-Meteo, not the LSTM.

## Technology stack

Python, Django REST Framework, PyTorch, LSTM, Angular, TypeScript, Leaflet, MongoDB, Open-Meteo API, NumPy, Pandas, scikit-learn

## Architecture

```text
Angular + Leaflet
      │ map coordinates / forecast display
      ▼
Django REST API ───── Open-Meteo Forecast API
      │                     │ current + historical + forecast data
      ├── PyTorch LSTM      │
      │   168h input → 24h temperature prediction
      │                     │
      └── MongoDB ◄─────────┘
          current conditions, latest forecasts,
          forecast request history, geospatial queries
```

## Requirements

- Windows with Python 3.11 and Node.js/npm
- MongoDB Community Server running locally (default URI: `mongodb://localhost:27017`)
- Internet access for Open-Meteo requests and OpenStreetMap map tiles

## Setup on Windows

Run commands from the project root (`E:\WEATHER PREDICTION BY LSTM`).

### 1. Install Python dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks virtual environment activation, allow it for the current terminal or use the virtual environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Confirm MongoDB is running

In PowerShell:

```powershell
Get-Service MongoDB
```

By default, the backend connects to `localhost:27017` and uses the `weather_forecast` database. You can override these with environment variables:

```powershell
$env:MONGODB_URI = "mongodb://localhost:27017"
$env:MONGODB_DATABASE = "weather_forecast"
```

### 3. Fetch training data and train the model

From the project root, with the virtual environment active:

```powershell
cd backend
python -m ml_pipeline.fetch_data
python -m ml_pipeline.train
```

The data fetcher downloads historical hourly weather for the configured 53 locations and caches files under `backend/ml_pipeline/data/`. Training writes the model, scaler, and metrics under `backend/ml_pipeline/artifacts/`. The trained model is required for the prediction endpoint; Django loads it when the app starts.

### 4. Start Django

From `backend/`:

```powershell
python manage.py check
python manage.py runserver 8000
```

The health endpoint is available at `http://127.0.0.1:8000/api/health/`.

### 5. Install and start Angular

Open a second PowerShell window:

```powershell
cd "E:\WEATHER PREDICTION BY LSTM\frontend"
npm install
npm start
```

Open `http://localhost:4200`. The Angular development proxy forwards `/api` requests to Django at `http://127.0.0.1:8000`.

## API endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/api/health/` | Check API and MongoDB availability |
| `GET` | `/api/weather/current/?lat={lat}&lon={lon}` | Current weather for coordinates |
| `GET` | `/api/weather/history/?lat={lat}&lon={lon}` | Recent 168-hour weather history and provider outlook data |
| `POST` | `/api/forecast/predict/` | Current weather, observed history, 24-hour LSTM/provider temperature comparison, and seven-day Open-Meteo outlook |
| `GET` | `/api/forecast/nearby/?lat={lat}&lon={lon}&radius_m={meters}` | Nearby saved forecasts (default radius: 50,000 m) |

### Example prediction request (PowerShell)

```powershell
$body = @{ lat = 9.9252; lon = 78.1198 } | ConvertTo-Json
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/forecast/predict/" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

Clicking the map in the Angular app sends the same latitude/longitude request automatically.

## Model and evaluation

- Input sequence: 168 hourly observations (7 days)
- Prediction horizon: 24 hourly temperature values
- Training/evaluation data: 53 locations; chronological splits are used for the standard training pipeline
- Loss: mean squared error (MSE)
- Optimizer: Adam, with gradient clipping and validation-based early stopping
- Metrics: MAE and RMSE in degrees Celsius, compared against simple persistence baselines

The source-matched operational spatial backtest trained on 42 locations and held out 11 unseen locations. Its experimental model reported **1.751°C MAE** and **2.366°C RMSE**; the same-hour-previous-day baseline reported **1.846°C MAE** and **2.613°C RMSE**. This is an experiment result, not a guarantee of accuracy for every location or weather condition. The live endpoint loads `weather_lstm.pt` and `weather_scaler.pkl`; the operational backtest saves its separate experiment artifacts.

To rerun the baseline evaluation for the saved standard model:

```powershell
cd backend
python -m ml_pipeline.train --evaluate-only
```

Other experiments include:

```powershell
python -m ml_pipeline.spatial_eval
python -m ml_pipeline.fetch_forecast_archive
python -m ml_pipeline.operational_backtest
```

## MongoDB collections

The backend uses the `weather_forecast` database:

- `current_weather`: latest current conditions per rounded coordinate.
- `forecasts`: latest forecast per coordinate, used for nearby geospatial searches.
- `forecast_runs`: append-only records of prediction requests, including issue time and both model/provider forecasts.

MongoDB creates collections when the first corresponding API request is stored. Weather history fetched by the history endpoint is returned to the caller; it is not stored as a permanent hourly collection.

## Project structure

```text
backend/
  config/                 Django project settings and root URLs
  weather_api/            REST endpoints, Open-Meteo integration, MongoDB, LSTM
  ml_pipeline/            Data fetching, feature engineering, training, evaluation
frontend/
  src/app/                Angular map and weather dashboard
  proxy.conf.json         Development proxy to Django
requirements.txt          Python dependencies
```

Training data, generated model artifacts, virtual environments, and the local SQLite database are excluded from Git by `.gitignore`. For a fresh checkout, fetch data and train the model before starting Django.
