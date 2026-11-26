"""Fetch archived forecast-model inputs used by the operational backtest."""

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

from ml_pipeline.fetch_data import TRAINING_LOCATIONS

FORECAST_ARCHIVE_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
DATA_DIR = Path(__file__).resolve().parent / "data"
CACHE_DIR = DATA_DIR / "forecast_history"
DATA_PATH = DATA_DIR / "historical_forecast_inputs.csv"
START_DATE = "2023-01-01"
END_DATE = "2025-12-31"


def fetch_location(name: str, lat: float, lon: float, refresh: bool = False) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{name}.csv"
    if cache_path.exists() and not refresh:
        print(f"Using cached forecast archive for {name}.")
        return pd.read_csv(cache_path, parse_dates=["time"])

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
        "wind_speed_unit": "kmh",
        "timezone": "UTC",
    }
    for attempt in range(4):
        response = requests.get(FORECAST_ARCHIVE_URL, params=params, timeout=90)
        if response.status_code == 429:
            if attempt == 3:
                detail = response.text[:400].replace("\n", " ")
                raise RuntimeError(
                    f"Open-Meteo rate limit persisted for {name}: {detail}. "
                    "Cached locations remain available; rerun later to resume."
                )
            retry_after = response.headers.get("Retry-After")
            try:
                delay = int(retry_after) if retry_after else 5 * (2 ** attempt)
            except ValueError:
                delay = 5 * (2 ** attempt)
            delay = min(max(delay, 1), 120)
            print(f"Rate limited for {name}; retrying in {delay} seconds...")
            time.sleep(delay)
            continue
        response.raise_for_status()
        time.sleep(1)
        break

    hourly = response.json()["hourly"]
    frame = pd.DataFrame(hourly)
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    frame["location"] = name
    frame["latitude"] = lat
    frame["longitude"] = lon
    frame = frame.dropna(
        subset=["temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m"]
    ).reset_index(drop=True)
    frame.to_csv(cache_path, index=False)
    return frame


def fetch_forecast_archive(refresh: bool = False) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for name, (lat, lon) in TRAINING_LOCATIONS.items():
        print(f"Fetching forecast-model history for {name} ({lat}, {lon})...")
        frames.append(fetch_location(name, lat, lon, refresh=refresh))
    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["location", "time"]).reset_index(drop=True)
    result.to_csv(DATA_PATH, index=False)
    print(f"Saved {len(result):,} rows from {result['location'].nunique()} locations to {DATA_PATH}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch the Historical Forecast API input series.")
    parser.add_argument("--refresh", action="store_true", help="Redownload cached location files.")
    args = parser.parse_args()
    fetch_forecast_archive(refresh=args.refresh)
