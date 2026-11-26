from datetime import datetime, timedelta, timezone
from collections import Counter

import numpy as np
import requests
import torch

from ml_pipeline.features import build_feature_matrix

from .apps import WeatherApiConfig
from .db import ensure_current_weather_indexes, ensure_forecast_indexes, get_database

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_CODE_DESCRIPTIONS = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


class WeatherProviderError(Exception):
    """Raised when Open-Meteo cannot provide usable weather data."""


class ModelUnavailableError(Exception):
    """Raised when trained model artifacts have not been loaded by Django."""


def fetch_last_seven_days(lat: float, lon: float) -> dict:
    hourly_fields = (
        "temperature_2m,relative_humidity_2m,precipitation,rain,"
        "wind_speed_10m,weather_code"
    )
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": hourly_fields,
        "past_days": 7,
        "forecast_days": 8,
        "wind_speed_unit": "kmh",
        "timezone": "UTC",
    }
    try:
        response = requests.get(OPEN_METEO_URL, params=params, timeout=20)
        response.raise_for_status()
    except requests.Timeout as exc:
        raise WeatherProviderError("Open-Meteo did not respond before the 20-second timeout.") from exc
    except requests.ConnectionError as exc:
        raise WeatherProviderError(
            "Django could not connect to Open-Meteo. Check this computer's internet, DNS, or firewall."
        ) from exc
    except requests.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else "unknown"
        raise WeatherProviderError(f"Open-Meteo returned HTTP {status_code}.") from exc
    except requests.RequestException as exc:
        raise WeatherProviderError("The request to Open-Meteo failed.") from exc

    try:
        payload = response.json()
        hourly = payload["hourly"]
        units = payload["hourly_units"]
        times = [datetime.fromisoformat(value) for value in hourly["time"]]
    except (ValueError, KeyError, TypeError) as exc:
        raise WeatherProviderError("Open-Meteo returned an unexpected hourly response format.") from exc

    times = [
        value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
        for value in times
    ]
    now = datetime.now(timezone.utc)
    field_names = [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "rain",
        "wind_speed_10m",
        "weather_code",
    ]
    records = []
    provider_forecast = []
    for index, timestamp in enumerate(times):
        values = {name: hourly[name][index] for name in field_names}
        if any(value is None for value in values.values()):
            continue
        record = {"time": timestamp.isoformat(), **values}
        if timestamp <= now:
            records.append(record)
        else:
            provider_forecast.append(record)

    records = records[-168:]
    if len(records) != 168:
        raise WeatherProviderError(
            f"Open-Meteo returned {len(records)} complete past hourly records; 168 are required."
        )
    if len(provider_forecast) < 24:
        raise WeatherProviderError(
            f"Open-Meteo returned {len(provider_forecast)} complete future hourly records; at least 24 are required."
        )

    # Build seven complete UTC calendar-day summaries after today. The daily
    # outlook is Open-Meteo data; the local LSTM currently predicts temperature only.
    future_by_date = {}
    for record in provider_forecast:
        day = datetime.fromisoformat(record["time"]).date()
        if day <= now.date():
            continue
        future_by_date.setdefault(day.isoformat(), []).append(record)

    provider_outlook_next_7_days = []
    for day, day_records in sorted(future_by_date.items())[:7]:
        codes = [int(row["weather_code"]) for row in day_records]
        most_common_code = Counter(codes).most_common(1)[0][0]
        provider_outlook_next_7_days.append({
            "date": day,
            "temperature_min": round(min(float(row["temperature_2m"]) for row in day_records), 1),
            "temperature_max": round(max(float(row["temperature_2m"]) for row in day_records), 1),
            "humidity_mean": round(sum(float(row["relative_humidity_2m"]) for row in day_records) / len(day_records)),
            "precipitation_sum": round(sum(float(row["precipitation"]) for row in day_records), 1),
            "rain_sum": round(sum(float(row["rain"]) for row in day_records), 1),
            "wind_speed_max": round(max(float(row["wind_speed_10m"]) for row in day_records), 1),
            "weather_code": most_common_code,
            "weather_description": WEATHER_CODE_DESCRIPTIONS.get(most_common_code, "Unknown"),
        })
    if len(provider_outlook_next_7_days) < 7:
        raise WeatherProviderError(
            f"Open-Meteo returned only {len(provider_outlook_next_7_days)} complete future days; 7 are required."
        )

    return {
        "location": {"type": "Point", "coordinates": [lon, lat]},
        "lat": lat,
        "lon": lon,
        "hours": len(records),
        "hourly": records,
        "provider_forecast_next_24h": provider_forecast[:24],
        "provider_outlook_next_7_days": provider_outlook_next_7_days,
        "units": {
            "temperature_2m": units["temperature_2m"],
            "relative_humidity_2m": units["relative_humidity_2m"],
            "precipitation": units["precipitation"],
            "rain": units["rain"],
            "wind_speed_10m": units["wind_speed_10m"],
        },
        "source": "open-meteo",
    }


def fetch_current_weather(lat: float, lon: float) -> dict:
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": (
            "temperature_2m,relative_humidity_2m,precipitation,rain,"
            "wind_speed_10m,weather_code"
        ),
        "wind_speed_unit": "kmh",
        "timezone": "UTC",
    }

    try:
        response = requests.get(OPEN_METEO_URL, params=params, timeout=15)
        response.raise_for_status()
    except requests.Timeout as exc:
        raise WeatherProviderError("Open-Meteo did not respond before the 15-second timeout.") from exc
    except requests.ConnectionError as exc:
        raise WeatherProviderError(
            "Django could not connect to Open-Meteo. Check this computer's internet, DNS, or firewall."
        ) from exc
    except requests.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else "unknown"
        raise WeatherProviderError(f"Open-Meteo returned HTTP {status_code}.") from exc
    except requests.RequestException as exc:
        raise WeatherProviderError("The request to Open-Meteo failed.") from exc

    try:
        payload = response.json()
        current = payload["current"]
        units = payload["current_units"]
        code = int(current["weather_code"])
    except (ValueError, KeyError, TypeError) as exc:
        raise WeatherProviderError("Open-Meteo returned an unexpected response format.") from exc

    doc = {
        "location": {"type": "Point", "coordinates": [lon, lat]},
        "lat": round(lat, 4),
        "lon": round(lon, 4),
        "current": {
            "time": current["time"],
            "temperature": current["temperature_2m"],
            "relative_humidity": current["relative_humidity_2m"],
            "precipitation": current["precipitation"],
            "rain": current["rain"],
            "wind_speed": current["wind_speed_10m"],
            "weather_code": code,
            "weather_description": WEATHER_CODE_DESCRIPTIONS.get(code, "Unknown"),
        },
        "units": {
            "temperature": units["temperature_2m"],
            "relative_humidity": units["relative_humidity_2m"],
            "precipitation": units["precipitation"],
            "rain": units["rain"],
            "wind_speed": units["wind_speed_10m"],
        },
        "source": "open-meteo",
        "updated_at": datetime.now(timezone.utc),
    }

    ensure_current_weather_indexes()
    get_database().current_weather.update_one(
        {"lat": doc["lat"], "lon": doc["lon"]},
        {"$set": doc},
        upsert=True,
    )
    return doc


def predict_for_location(lat: float, lon: float) -> dict:
    if WeatherApiConfig.model is None or WeatherApiConfig.scaler is None:
        raise ModelUnavailableError("The trained weather model is not loaded. Restart Django after training.")

    current = fetch_current_weather(lat, lon)
    history = fetch_last_seven_days(lat, lon)
    records = history["hourly"]
    feature_matrix = build_feature_matrix(
        [row["temperature_2m"] for row in records],
        [row["relative_humidity_2m"] for row in records],
        [row["precipitation"] for row in records],
        [row["wind_speed_10m"] for row in records],
        [row["time"] for row in records],
        lat,
        lon,
    )
    scaled = WeatherApiConfig.scaler.transform(feature_matrix)
    tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(WeatherApiConfig.device)
    with torch.no_grad():
        prediction_scaled = WeatherApiConfig.model(tensor).cpu().numpy().reshape(-1)

    scaler = WeatherApiConfig.scaler
    predicted_temperatures = (prediction_scaled * scaler.scale_[0] + scaler.mean_[0]).round(2)
    last_input_time = datetime.fromisoformat(records[-1]["time"])
    predicted_next_24h = [
        {
            "time": (last_input_time + timedelta(hours=step)).isoformat(),
            "temperature": float(temperature),
        }
        for step, temperature in enumerate(predicted_temperatures, start=1)
    ]
    observed_last_24h = [
        {"time": row["time"], "temperature": row["temperature_2m"]}
        for row in records[-24:]
    ]
    doc = {
        "location": {"type": "Point", "coordinates": [lon, lat]},
        "lat": round(lat, 4),
        "lon": round(lon, 4),
        "current": current["current"],
        "units": current["units"],
        "observed_last_24h": observed_last_24h,
        "predicted_next_24h": predicted_next_24h,
        "provider_forecast_next_24h": [
            {
                "time": row["time"],
                "temperature": round(float(row["temperature_2m"]), 2),
            }
            for row in history["provider_forecast_next_24h"]
        ],
        "provider_outlook_next_7_days": history["provider_outlook_next_7_days"],
        "model_status": "lstm_inference",
        "model_version": "weather_lstm_global_v1",
        "source": "open-meteo",
        "updated_at": datetime.now(timezone.utc),
    }
    ensure_forecast_indexes()
    database = get_database()
    database.forecasts.update_one(
        {"lat": doc["lat"], "lon": doc["lon"]},
        {"$set": doc},
        upsert=True,
    )
    database.forecast_runs.insert_one({**doc, "issued_at": datetime.now(timezone.utc)})
    return doc


def find_nearby_forecasts(lat: float, lon: float, max_distance_m: int = 50000) -> list:
    cursor = get_database().forecasts.find(
        {
            "location": {
                "$near": {
                    "$geometry": {"type": "Point", "coordinates": [lon, lat]},
                    "$maxDistance": max_distance_m,
                }
            }
        },
        {"_id": 0},
    ).limit(20)
    return list(cursor)
