from django.urls import path

from .views import current_weather, health, nearby_forecasts, predict_weather, weather_history

urlpatterns = [
    path("health/", health, name="health"),
    path("weather/current/", current_weather, name="current-weather"),
    path("weather/history/", weather_history, name="weather-history"),
    path("forecast/predict/", predict_weather, name="predict-weather"),
    path("forecast/nearby/", nearby_forecasts, name="nearby-forecasts"),
]
