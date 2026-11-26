import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from ml_pipeline.features import build_feature_matrix


class WeatherWindowDataset(Dataset):
    def __init__(self, segments, seq_len: int, horizon: int, stride: int):
        self.x_segments = []
        self.y_segments = []
        self.windows = []
        for features, temperatures in segments:
            x_tensor = torch.from_numpy(np.asarray(features, dtype=np.float32))
            y_tensor = torch.from_numpy(np.asarray(temperatures, dtype=np.float32))
            segment_index = len(self.x_segments)
            self.x_segments.append(x_tensor)
            self.y_segments.append(y_tensor)
            max_start = len(x_tensor) - seq_len - horizon
            for start in range(0, max_start + 1, stride):
                self.windows.append((segment_index, start))

        self.seq_len = seq_len
        self.horizon = horizon

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        segment_index, start = self.windows[index]
        x = self.x_segments[segment_index]
        y = self.y_segments[segment_index]
        target_start = start + self.seq_len
        return x[start:target_start], y[target_start : target_start + self.horizon]


def prepare_splits(
    frame: pd.DataFrame,
    seq_len: int = 168,
    horizon: int = 24,
    stride: int = 12,
):
    """Create chronological splits per location; fit normalization on training rows only."""
    required = {
        "time", "location", "latitude", "longitude", "temperature_2m",
        "relative_humidity_2m", "precipitation", "wind_speed_10m",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Training data is missing columns: {', '.join(sorted(missing))}")

    train_raw, val_raw, test_raw = [], [], []
    for _, group in frame.sort_values(["location", "time"]).groupby("location", sort=False):
        group = group.reset_index(drop=True)
        if len(group) < seq_len + horizon + 3:
            continue

        features = build_feature_matrix(
            group["temperature_2m"].to_numpy(),
            group["relative_humidity_2m"].to_numpy(),
            group["precipitation"].to_numpy(),
            group["wind_speed_10m"].to_numpy(),
            group["time"].to_numpy(),
            float(group["latitude"].iloc[0]),
            float(group["longitude"].iloc[0]),
        )
        temperatures = group["temperature_2m"].to_numpy(dtype=np.float32)
        n = len(group)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)
        train_raw.append((features[:train_end], temperatures[:train_end]))
        val_raw.append((features[train_end:val_end], temperatures[train_end:val_end]))
        test_raw.append((features[val_end:], temperatures[val_end:]))

    if not train_raw or not val_raw or not test_raw:
        raise ValueError("Not enough data to make train, validation, and test windows.")

    scaler = StandardScaler()
    scaler.fit(np.concatenate([features for features, _ in train_raw], axis=0))
    temperature_mean = scaler.mean_[0]
    temperature_scale = scaler.scale_[0]

    def scale_segments(raw_segments):
        return [
            (
                scaler.transform(features).astype(np.float32),
                ((temperatures - temperature_mean) / temperature_scale).astype(np.float32),
            )
            for features, temperatures in raw_segments
        ]

    train_ds = WeatherWindowDataset(scale_segments(train_raw), seq_len, horizon, stride)
    val_ds = WeatherWindowDataset(scale_segments(val_raw), seq_len, horizon, stride)
    test_ds = WeatherWindowDataset(scale_segments(test_raw), seq_len, horizon, stride)
    if not len(train_ds) or not len(val_ds) or not len(test_ds):
        raise ValueError("One or more splits contain no valid sequence windows.")
    return train_ds, val_ds, test_ds, scaler
