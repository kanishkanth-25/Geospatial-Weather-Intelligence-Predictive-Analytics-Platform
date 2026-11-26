import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader

from ml_pipeline.dataset import WeatherWindowDataset
from ml_pipeline.fetch_data import DATA_PATH
from ml_pipeline.features import build_feature_matrix, FEATURE_NAMES
from ml_pipeline.train import HORIZON, SEQ_LEN, WINDOW_STRIDE, evaluate
from weather_api.ml_model import WeatherLSTM

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
SPATIAL_TEST_LOCATIONS = {
    "cairo",
    "lagos",
    "london",
    "reykjavik",
    "new_york",
    "anchorage",
    "mexico_city",
    "sao_paulo",
    "nairobi",
    "sydney",
    "tokyo",
}


def make_spatial_splits(frame):
    train_raw, val_raw, test_raw = [], [], []
    present_locations = set(frame["location"].unique())
    missing = SPATIAL_TEST_LOCATIONS - present_locations
    if missing:
        raise ValueError(f"Spatial test locations missing from data: {', '.join(sorted(missing))}")

    for location, group in frame.sort_values(["location", "time"]).groupby("location", sort=False):
        group = group.reset_index(drop=True)
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
        if location in SPATIAL_TEST_LOCATIONS:
            test_raw.append((features, temperatures))
        else:
            split = int(len(group) * 0.85)
            train_raw.append((features[:split], temperatures[:split]))
            val_raw.append((features[split:], temperatures[split:]))

    scaler = StandardScaler()
    scaler.fit(np.concatenate([features for features, _ in train_raw], axis=0))
    temperature_mean, temperature_scale = scaler.mean_[0], scaler.scale_[0]

    def scale(segments):
        return [
            (
                scaler.transform(features).astype(np.float32),
                ((temperatures - temperature_mean) / temperature_scale).astype(np.float32),
            )
            for features, temperatures in segments
        ]

    return (
        WeatherWindowDataset(scale(train_raw), SEQ_LEN, HORIZON, WINDOW_STRIDE),
        WeatherWindowDataset(scale(val_raw), SEQ_LEN, HORIZON, WINDOW_STRIDE),
        WeatherWindowDataset(scale(test_raw), SEQ_LEN, HORIZON, WINDOW_STRIDE),
        scaler,
        sorted(present_locations - SPATIAL_TEST_LOCATIONS),
    )


def run_spatial_evaluation(epochs=20, batch_size=128, learning_rate=1e-3, patience=4):
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found at {DATA_PATH}. Run python -m ml_pipeline.fetch_data first.")
    frame = pd.read_csv(DATA_PATH, parse_dates=["time"])
    train_ds, val_ds, test_ds, scaler, training_locations = make_spatial_splits(frame)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WeatherLSTM(input_size=len(FEATURE_NAMES)).to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACTS_DIR / "weather_lstm_spatial.pt"
    best_val_loss = float("inf")
    best_epoch = 0
    no_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(x_batch), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item() * len(x_batch)
        train_loss = total_train_loss / len(train_ds)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                total_val_loss += loss_function(model(x_batch), y_batch).item() * len(x_batch)
        val_loss = total_val_loss / len(val_ds)
        print(f"Epoch {epoch:02d}/{epochs} | Train MSE: {train_loss:.5f} | Val MSE: {val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improvement = 0
            torch.save(model.state_dict(), model_path)
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print(f"Early stopping at epoch {epoch}; best validation epoch was {best_epoch}.")
                break

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    metrics = {
        "training_locations": training_locations,
        "training_location_count": len(training_locations),
        "spatial_test_locations": sorted(SPATIAL_TEST_LOCATIONS),
        "spatial_test_location_count": len(SPATIAL_TEST_LOCATIONS),
        "best_epoch": best_epoch,
        "train_windows": len(train_ds),
        "validation_windows": len(val_ds),
        "test_windows": len(test_ds),
        "sequence_hours": SEQ_LEN,
        "forecast_hours": HORIZON,
        "metrics": evaluate(model, test_loader, device, scaler),
        "device": str(device),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(scaler, ARTIFACTS_DIR / "weather_scaler_spatial.pkl")
    result_path = ARTIFACTS_DIR / "spatial_holdout_metrics.json"
    result_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Spatial holdout metrics:", json.dumps(metrics, indent=2))
    print(f"Saved spatial model to {model_path}")
    print(f"Saved spatial evaluation to {result_path}")
    return metrics


if __name__ == "__main__":
    run_spatial_evaluation()
