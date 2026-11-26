import json
import argparse
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from ml_pipeline.dataset import prepare_splits
from ml_pipeline.fetch_data import DATA_PATH
from ml_pipeline.features import FEATURE_NAMES
from weather_api.ml_model import WeatherLSTM

SEQ_LEN = 168
HORIZON = 24
WINDOW_STRIDE = 12
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


def evaluate(model, loader, device, scaler):
    model.eval()
    predictions, previous_day, persistence, targets = [], [], [], []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            prediction = model(x_batch.to(device)).cpu().numpy()
            predictions.append(prediction)
            previous_day.append(x_batch[:, -HORIZON:, 0].numpy())
            persistence.append(
                np.repeat(x_batch[:, -1:, 0].numpy(), HORIZON, axis=1)
            )
            targets.append(y_batch.numpy())

    y_true_scaled = np.concatenate(targets).reshape(-1)

    def score(prediction_batches):
        y_pred_scaled = np.concatenate(prediction_batches).reshape(-1)
        temperature_mean = scaler.mean_[0]
        temperature_scale = scaler.scale_[0]
        y_pred = y_pred_scaled * temperature_scale + temperature_mean
        y_true = y_true_scaled * temperature_scale + temperature_mean
        rmse = float(np.sqrt(np.mean(np.square(y_true - y_pred))))
        mae = float(np.mean(np.abs(y_true - y_pred)))
        return {"rmse_c": round(rmse, 3), "mae_c": round(mae, 3)}

    return {
        "lstm": score(predictions),
        "same_hour_previous_day": score(previous_day),
        "persistence_last_hour": score(persistence),
    }


def evaluate_saved_model(batch_size: int = 128):
    model_path = ARTIFACTS_DIR / "weather_lstm.pt"
    scaler_path = ARTIFACTS_DIR / "weather_scaler.pkl"
    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError("Trained model artifacts not found. Run python -m ml_pipeline.train first.")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found at {DATA_PATH}.")

    frame = pd.read_csv(DATA_PATH, parse_dates=["time"])
    _, _, test_ds, scaler = prepare_splits(frame, SEQ_LEN, HORIZON, WINDOW_STRIDE)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WeatherLSTM(input_size=len(FEATURE_NAMES)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    comparison = evaluate(model, test_loader, device, scaler)
    result = {
        "test_windows": len(test_ds),
        "locations": int(frame["location"].nunique()),
        "sequence_hours": SEQ_LEN,
        "forecast_hours": HORIZON,
        "metrics": comparison,
        "device": str(device),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    result_path = ARTIFACTS_DIR / "baseline_comparison.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Held-out baseline comparison:", json.dumps(result, indent=2))
    print(f"Saved comparison to {result_path}")
    return result


def train(epochs: int = 20, batch_size: int = 128, learning_rate: float = 1e-3, patience: int = 4):
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found at {DATA_PATH}. Run python -m ml_pipeline.fetch_data first.")

    frame = pd.read_csv(DATA_PATH, parse_dates=["time"])
    train_ds, val_ds, test_ds, scaler = prepare_splits(frame, SEQ_LEN, HORIZON, WINDOW_STRIDE)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WeatherLSTM(input_size=len(FEATURE_NAMES)).to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACTS_DIR / "weather_lstm.pt"
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
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
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                total_val_loss += loss_function(model(x_batch), y_batch).item() * len(x_batch)
        val_loss = total_val_loss / len(val_ds)
        print(f"Epoch {epoch:02d}/{epochs} | Train MSE: {train_loss:.5f} | Val MSE: {val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch}; best validation epoch was {best_epoch}.")
                break

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    metrics = evaluate(model, test_loader, device, scaler)
    metrics.update({
        "best_epoch": best_epoch,
        "sequence_hours": SEQ_LEN,
        "forecast_hours": HORIZON,
        "window_stride_hours": WINDOW_STRIDE,
        "training_locations": int(frame["location"].nunique()),
        "train_windows": len(train_ds),
        "validation_windows": len(val_ds),
        "test_windows": len(test_ds),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
    })
    joblib.dump(scaler, ARTIFACTS_DIR / "weather_scaler.pkl")
    (ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Held-out test metrics:", metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate the global weather LSTM.")
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Compare the saved model with simple baselines without retraining.",
    )
    args = parser.parse_args()
    if args.evaluate_only:
        evaluate_saved_model()
    else:
        train()
