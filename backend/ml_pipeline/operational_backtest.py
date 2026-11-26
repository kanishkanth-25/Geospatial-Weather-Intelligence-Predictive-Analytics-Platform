"""Train/evaluate with forecast-archive inputs and historical-weather targets."""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ml_pipeline.fetch_data import DATA_PATH as ACTUAL_DATA_PATH
from ml_pipeline.fetch_forecast_archive import DATA_PATH as FORECAST_DATA_PATH
from ml_pipeline.features import build_feature_matrix, FEATURE_NAMES
from ml_pipeline.train import HORIZON, SEQ_LEN, WINDOW_STRIDE
from weather_api.ml_model import WeatherLSTM

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
SPATIAL_TEST_LOCATIONS = {
    "anchorage", "cairo", "lagos", "london", "mexico_city", "nairobi",
    "new_york", "reykjavik", "sao_paulo", "sydney", "tokyo",
}


class OperationalWindowDataset(Dataset):
    """Sequence inputs from forecast archives paired with observed target temperatures."""

    def __init__(self, entries):
        self.features = []
        self.targets = []
        self.windows = []
        for x_values, y_values, starts in entries:
            segment_index = len(self.features)
            self.features.append(torch.from_numpy(np.asarray(x_values, dtype=np.float32)))
            self.targets.append(torch.from_numpy(np.asarray(y_values, dtype=np.float32)))
            self.windows.extend((segment_index, int(start)) for start in starts)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        segment_index, start = self.windows[index]
        target_start = start + SEQ_LEN
        return (
            self.features[segment_index][start:target_start],
            self.targets[segment_index][target_start : target_start + HORIZON],
        )


def _contiguous_ranges(times):
    values = pd.DatetimeIndex(times).to_numpy()
    if len(values) == 0:
        return []
    break_points = np.flatnonzero(np.diff(values) != np.timedelta64(1, "h")) + 1
    boundaries = np.concatenate(([0], break_points, [len(values)]))
    return [(int(boundaries[i]), int(boundaries[i + 1])) for i in range(len(boundaries) - 1)]


def make_operational_splits(forecast_frame, actual_frame):
    forecast_groups = {name: group for name, group in forecast_frame.groupby("location", sort=False)}
    actual_groups = {name: group for name, group in actual_frame.groupby("location", sort=False)}
    locations = sorted(set(forecast_groups) & set(actual_groups))
    missing_test = SPATIAL_TEST_LOCATIONS - set(locations)
    if missing_test:
        raise ValueError(f"Missing held-out locations: {', '.join(sorted(missing_test))}")

    train_raw, val_raw, test_raw, train_feature_rows, train_target_rows = [], [], [], [], []
    training_locations = []
    for location in locations:
        forecast = forecast_groups[location].copy()
        actual = actual_groups[location][["time", "temperature_2m"]].rename(
            columns={"temperature_2m": "target_temperature"}
        )
        joined = forecast.merge(actual, on="time", how="inner", validate="one_to_one")
        joined = joined.sort_values("time").dropna().reset_index(drop=True)
        if len(joined) < SEQ_LEN + HORIZON + 3:
            continue

        latitude = float(joined["latitude"].iloc[0])
        longitude = float(joined["longitude"].iloc[0])
        features = build_feature_matrix(
            joined["temperature_2m"].to_numpy(),
            joined["relative_humidity_2m"].to_numpy(),
            joined["precipitation"].to_numpy(),
            joined["wind_speed_10m"].to_numpy(),
            joined["time"].to_numpy(),
            latitude,
            longitude,
        )
        targets = joined["target_temperature"].to_numpy(dtype=np.float32)
        n = len(joined)

        if location in SPATIAL_TEST_LOCATIONS:
            training_split_end = n
            validation_split_end = n
        else:
            training_locations.append(location)
            training_split_end = int(n * 0.70)
            validation_split_end = int(n * 0.85)
            train_feature_rows.append(features[:training_split_end])
            train_target_rows.append(targets[:training_split_end])

        for range_start, range_end in _contiguous_ranges(joined["time"]):
            max_start = range_end - range_start - SEQ_LEN - HORIZON
            if max_start < 0:
                continue
            selected_train, selected_val, selected_test = [], [], []
            for local_start in range(0, max_start + 1, WINDOW_STRIDE):
                global_start = range_start + local_start
                target_start = global_start + SEQ_LEN
                target_end = target_start + HORIZON
                if location in SPATIAL_TEST_LOCATIONS:
                    selected_test.append(local_start)
                elif target_end <= training_split_end:
                    selected_train.append(local_start)
                elif target_start >= training_split_end and target_end <= validation_split_end:
                    selected_val.append(local_start)

            segment_features = features[range_start:range_end]
            segment_targets = targets[range_start:range_end]
            if selected_train:
                train_raw.append((segment_features, segment_targets, selected_train))
            if selected_val:
                val_raw.append((segment_features, segment_targets, selected_val))
            if selected_test:
                test_raw.append((segment_features, segment_targets, selected_test))

    if not train_raw or not val_raw or not test_raw:
        raise ValueError("Could not construct non-empty operational train/validation/test windows.")

    feature_scaler = StandardScaler().fit(np.concatenate(train_feature_rows, axis=0))
    target_scaler = StandardScaler().fit(np.concatenate(train_target_rows).reshape(-1, 1))

    def scale_entries(entries):
        return [
            (
                feature_scaler.transform(x_values).astype(np.float32),
                target_scaler.transform(y_values.reshape(-1, 1)).reshape(-1).astype(np.float32),
                starts,
            )
            for x_values, y_values, starts in entries
        ]

    return (
        OperationalWindowDataset(scale_entries(train_raw)),
        OperationalWindowDataset(scale_entries(val_raw)),
        OperationalWindowDataset(scale_entries(test_raw)),
        feature_scaler,
        target_scaler,
        training_locations,
    )


def evaluate_operational(model, loader, device, feature_scaler, target_scaler):
    model.eval()
    predictions, targets, previous_day, persistence = [], [], [], []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            predictions.append(model(x_batch.to(device)).cpu().numpy())
            targets.append(y_batch.numpy())
            raw_last_day = x_batch[:, -HORIZON:, 0].numpy() * feature_scaler.scale_[0] + feature_scaler.mean_[0]
            raw_last_hour = x_batch[:, -1:, 0].numpy() * feature_scaler.scale_[0] + feature_scaler.mean_[0]
            previous_day.append(raw_last_day)
            persistence.append(np.repeat(raw_last_hour, HORIZON, axis=1))

    target_c = target_scaler.inverse_transform(np.concatenate(targets).reshape(-1, 1)).reshape(-1)
    lstm_c = target_scaler.inverse_transform(np.concatenate(predictions).reshape(-1, 1)).reshape(-1)

    def score(predicted_c):
        error = target_c - predicted_c
        return {
            "rmse_c": round(float(np.sqrt(np.mean(error ** 2))), 3),
            "mae_c": round(float(np.mean(np.abs(error))), 3),
        }

    return {
        "lstm": score(lstm_c),
        "same_hour_previous_day": score(np.concatenate(previous_day).reshape(-1)),
        "persistence_last_hour": score(np.concatenate(persistence).reshape(-1)),
    }


def run_backtest(epochs=20, batch_size=128, learning_rate=1e-3, patience=4):
    if not FORECAST_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Forecast-model input data not found at {FORECAST_DATA_PATH}. "
            "Run python -m ml_pipeline.fetch_forecast_archive first."
        )
    if not ACTUAL_DATA_PATH.exists():
        raise FileNotFoundError(f"Historical target data not found at {ACTUAL_DATA_PATH}.")

    forecast_frame = pd.read_csv(FORECAST_DATA_PATH, parse_dates=["time"])
    actual_frame = pd.read_csv(ACTUAL_DATA_PATH, parse_dates=["time"])
    train_ds, val_ds, test_ds, feature_scaler, target_scaler, training_locations = make_operational_splits(
        forecast_frame, actual_frame
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WeatherLSTM(input_size=len(FEATURE_NAMES)).to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACTS_DIR / "weather_lstm_operational.pt"
    best_val_loss, best_epoch, no_improvement = float("inf"), 0, 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_total = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(x_batch), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_total += loss.item() * len(x_batch)
        train_loss = train_loss_total / len(train_ds)

        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                val_loss_total += loss_function(model(x_batch), y_batch).item() * len(x_batch)
        val_loss = val_loss_total / len(val_ds)
        print(f"Epoch {epoch:02d}/{epochs} | Train MSE: {train_loss:.5f} | Val MSE: {val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss, best_epoch, no_improvement = val_loss, epoch, 0
            torch.save(model.state_dict(), model_path)
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print(f"Early stopping at epoch {epoch}; best validation epoch was {best_epoch}.")
                break

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    metrics = {
        "train_locations": training_locations,
        "train_location_count": len(training_locations),
        "spatial_test_locations": sorted(SPATIAL_TEST_LOCATIONS),
        "best_epoch": best_epoch,
        "train_windows": len(train_ds),
        "validation_windows": len(val_ds),
        "test_windows": len(test_ds),
        "sequence_hours": SEQ_LEN,
        "forecast_hours": HORIZON,
        "metrics": evaluate_operational(model, test_loader, device, feature_scaler, target_scaler),
        "device": str(device),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(feature_scaler, ARTIFACTS_DIR / "feature_scaler_operational.pkl")
    joblib.dump(target_scaler, ARTIFACTS_DIR / "target_scaler_operational.pkl")
    metrics_path = ARTIFACTS_DIR / "operational_backtest_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Source-matched spatial backtest:", json.dumps(metrics, indent=2))
    print(f"Saved experiment model to {model_path}")
    print(f"Saved metrics to {metrics_path}")
    return metrics


if __name__ == "__main__":
    run_backtest()
