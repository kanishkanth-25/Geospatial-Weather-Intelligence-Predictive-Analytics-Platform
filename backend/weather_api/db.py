import os
from functools import lru_cache

from pymongo import MongoClient


@lru_cache(maxsize=1)
def get_database():
    client = MongoClient(
        os.environ.get("MONGODB_URI", "mongodb://localhost:27017"),
        serverSelectionTimeoutMS=5000,
    )
    client.admin.command("ping")
    return client[os.environ.get("MONGODB_DATABASE", "weather_forecast")]


@lru_cache(maxsize=1)
def ensure_current_weather_indexes():
    get_database().current_weather.create_index([("location", "2dsphere")])


@lru_cache(maxsize=1)
def ensure_forecast_indexes():
    database = get_database()
    database.forecasts.create_index([("location", "2dsphere")])
    database.forecasts.create_index([("lat", 1), ("lon", 1)], unique=True)
    database.forecast_runs.create_index([("location", "2dsphere")])
    database.forecast_runs.create_index([("issued_at", 1)])
