import torch
from torch import nn


class WeatherLSTM(nn.Module):
    """Forecast 24 scaled temperatures from 168 hourly, location-aware features."""

    def __init__(self, input_size=9, hidden_size=64, num_layers=2, output_horizon=24):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, output_horizon)

    def forward(self, values):
        sequence, _ = self.lstm(values)
        return self.output(sequence[:, -1, :])
