import numpy as np
import pandas as pd


FEATURE_NAMES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "hour_sin",
    "hour_cos",
    "latitude_normalized",
    "longitude_sin",
    "longitude_cos",
]


def build_feature_matrix(
    temperature,
    relative_humidity,
    precipitation,
    wind_speed,
    timestamps,
    latitude: float,
    longitude: float,
) -> np.ndarray:
    """Build the shared, ordered feature matrix used for training and inference."""
    hours = np.asarray([pd.Timestamp(value).hour for value in timestamps], dtype=np.float32)
    longitude_radians = np.deg2rad(longitude)
    return np.column_stack(
        [
            np.asarray(temperature, dtype=np.float32),
            np.asarray(relative_humidity, dtype=np.float32),
            np.asarray(precipitation, dtype=np.float32),
            np.asarray(wind_speed, dtype=np.float32),
            np.sin(2 * np.pi * hours / 24.0),
            np.cos(2 * np.pi * hours / 24.0),
            np.full(len(hours), latitude / 90.0, dtype=np.float32),
            np.full(len(hours), np.sin(longitude_radians), dtype=np.float32),
            np.full(len(hours), np.cos(longitude_radians), dtype=np.float32),
        ]
    )
