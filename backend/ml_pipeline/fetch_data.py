from datetime import datetime, timedelta, timezone
from pathlib import Path
import time

import pandas as pd
import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DATA_PATH = Path(__file__).resolve().parent / "data" / "historical_weather.csv"

# Starter coverage across tropical, arid, temperate, and southern-hemisphere climates.
# This is a prototype sample, not enough to claim reliable coverage at every world location.
TRAINING_LOCATIONS = {
    "madurai": (9.9252, 78.1198),
    "chennai": (13.0827, 80.2707),
    "delhi": (28.6139, 77.2090),
    "mumbai": (19.0760, 72.8777),
    "kolkata": (22.5726, 88.3639),
    "dhaka": (23.8103, 90.4125),
    "kathmandu": (27.7172, 85.3240),
    "bangkok": (13.7563, 100.5018),
    "singapore": (1.3521, 103.8198),
    "jakarta": (-6.2088, 106.8456),
    "manila": (14.5995, 120.9842),
    "beijing": (39.9042, 116.4074),
    "shanghai": (31.2304, 121.4737),
    "seoul": (37.5665, 126.9780),
    "riyadh": (24.7136, 46.6753),
    "dubai": (25.2048, 55.2708),
    "tehran": (35.6892, 51.3890),
    "istanbul": (41.0082, 28.9784),
    "cairo": (30.0444, 31.2357),
    "lagos": (6.5244, 3.3792),
    "cape_town": (-33.9249, 18.4241),
    "addis_ababa": (8.9806, 38.7578),
    "dakar": (14.7167, -17.4677),
    "casablanca": (33.5731, -7.5898),
    "johannesburg": (-26.2041, 28.0473),
    "nairobi": (-1.2921, 36.8219),
    "london": (51.5072, -0.1276),
    "paris": (48.8566, 2.3522),
    "oslo": (59.9139, 10.7522),
    "moscow": (55.7558, 37.6173),
    "rome": (41.9028, 12.4964),
    "madrid": (40.4168, -3.7038),
    "berlin": (52.5200, 13.4050),
    "reykjavik": (64.1466, -21.9426),
    "new_york": (40.7128, -74.0060),
    "los_angeles": (34.0522, -118.2437),
    "chicago": (41.8781, -87.6298),
    "miami": (25.7617, -80.1918),
    "mexico_city": (19.4326, -99.1332),
    "anchorage": (61.2181, -149.9003),
    "toronto": (43.6532, -79.3832),
    "vancouver": (49.2827, -123.1207),
    "sao_paulo": (-23.5505, -46.6333),
    "buenos_aires": (-34.6037, -58.3816),
    "lima": (-12.0464, -77.0428),
    "bogota": (4.7110, -74.0721),
    "santiago": (-33.4489, -70.6693),
    "manaus": (-3.1190, -60.0217),
    "sydney": (-33.8688, 151.2093),
    "melbourne": (-37.8136, 144.9631),
    "perth": (-31.9505, 115.8605),
    "auckland": (-36.8509, 174.7645),
    "tokyo": (35.6762, 139.6503),
}


def fetch_location_history(
    name: str,
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    refresh: bool = False,
) -> pd.DataFrame:
    cache_dir = DATA_PATH.parent / "locations"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{name}.csv"
    if cache_path.exists() and not refresh:
        print(f"Using cached history for {name}; pass --refresh to download it again.")
        return pd.read_csv(cache_path, parse_dates=["time"])

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
        "wind_speed_unit": "kmh",
        "timezone": "UTC",
    }
    for attempt in range(4):
        response = requests.get(ARCHIVE_URL, params=params, timeout=60)
        if response.status_code == 429:
            if attempt == 3:
                details = response.text[:500].replace("\n", " ")
                raise RuntimeError(
                    f"Open-Meteo rate limit persisted for {name}. Details: {details}. "
                    "Successful locations are cached; rerun later to resume."
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
    frame = frame.dropna(subset=[
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "wind_speed_10m",
    ]).reset_index(drop=True)
    frame.to_csv(cache_path, index=False)
    return frame


def fetch_training_data(
    start_date: str = "2023-01-01",
    end_date: str | None = None,
    locations: dict[str, tuple[float, float]] | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    if end_date is None:
        end_date = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
    if locations is None:
        locations = TRAINING_LOCATIONS

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for name, (lat, lon) in locations.items():
        print(f"Fetching {name} ({lat}, {lon}) from {start_date} to {end_date}...")
        frames.append(fetch_location_history(name, lat, lon, start_date, end_date, refresh=refresh))

    if not frames:
        raise ValueError("At least one training location is required.")

    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values(["location", "time"]).reset_index(drop=True)
    data.to_csv(DATA_PATH, index=False)
    print(f"Saved {len(data):,} hourly rows from {len(frames)} locations to {DATA_PATH}")
    return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download multi-location hourly weather training data.")
    parser.add_argument("--refresh", action="store_true", help="Redownload all location data instead of using cached CSVs.")
    args = parser.parse_args()
    fetch_training_data(refresh=args.refresh)
