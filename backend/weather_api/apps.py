import joblib
import torch
from django.apps import AppConfig
from django.conf import settings


class WeatherApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'weather_api'
    model = None
    scaler = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def ready(self):
        model_path = settings.ML_ARTIFACTS_DIR / "weather_lstm.pt"
        scaler_path = settings.ML_ARTIFACTS_DIR / "weather_scaler.pkl"
        if not model_path.exists() or not scaler_path.exists():
            return

        from .ml_model import WeatherLSTM

        model = WeatherLSTM().to(self.device)
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        WeatherApiConfig.model = model
        WeatherApiConfig.scaler = joblib.load(scaler_path)
