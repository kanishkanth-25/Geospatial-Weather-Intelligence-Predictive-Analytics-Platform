import math

from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .db import get_database
from .services import (
    ModelUnavailableError,
    WeatherProviderError,
    fetch_current_weather,
    fetch_last_seven_days,
    find_nearby_forecasts,
    predict_for_location,
)


@api_view(["GET"])
def health(request):
    try:
        get_database().command("ping")
    except Exception:
        return Response(
            {"api": "ok", "mongodb": "unavailable"},
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    return Response({"api": "ok", "mongodb": "ok"}, status=status.HTTP_200_OK)


@api_view(["GET"])
def current_weather(request):
    try:
        lat = float(request.query_params.get("lat"))
        lon = float(request.query_params.get("lon"))
        if not math.isfinite(lat) or not math.isfinite(lon):
            raise ValueError
        if not -90 <= lat <= 90 or not -180 <= lon <= 180:
            raise ValueError
    except (TypeError, ValueError):
        return Response(
            {"error": "Provide valid latitude and longitude coordinates."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    try:
        result = fetch_current_weather(lat, lon)
    except WeatherProviderError as exc:
        return Response({"error": str(exc)}, status=status.HTTP_502_BAD_GATEWAY)
    except Exception:
        return Response(
            {"error": "Weather data was retrieved, but could not be saved."},
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    return Response(result, status=status.HTTP_200_OK)


@api_view(["GET"])
def weather_history(request):
    try:
        lat = float(request.query_params.get("lat"))
        lon = float(request.query_params.get("lon"))
        if not math.isfinite(lat) or not math.isfinite(lon):
            raise ValueError
        if not -90 <= lat <= 90 or not -180 <= lon <= 180:
            raise ValueError
    except (TypeError, ValueError):
        return Response(
            {"error": "Provide valid latitude and longitude coordinates."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    try:
        result = fetch_last_seven_days(lat, lon)
    except WeatherProviderError as exc:
        return Response({"error": str(exc)}, status=status.HTTP_502_BAD_GATEWAY)
    return Response(result, status=status.HTTP_200_OK)


@api_view(["POST"])
def predict_weather(request):
    try:
        lat = float(request.data.get("lat"))
        lon = float(request.data.get("lon"))
        if not math.isfinite(lat) or not math.isfinite(lon):
            raise ValueError
        if not -90 <= lat <= 90 or not -180 <= lon <= 180:
            raise ValueError
    except (TypeError, ValueError):
        return Response(
            {"error": "Provide valid latitude and longitude coordinates."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    try:
        result = predict_for_location(lat, lon)
    except ModelUnavailableError as exc:
        return Response({"error": str(exc)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
    except WeatherProviderError as exc:
        return Response({"error": str(exc)}, status=status.HTTP_502_BAD_GATEWAY)
    except Exception:
        return Response(
            {"error": "Forecast inference or storage failed."},
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    return Response(result, status=status.HTTP_200_OK)


@api_view(["GET"])
def nearby_forecasts(request):
    try:
        lat = float(request.query_params.get("lat"))
        lon = float(request.query_params.get("lon"))
        radius_m = int(request.query_params.get("radius_m", 50000))
        if not math.isfinite(lat) or not math.isfinite(lon):
            raise ValueError
        if not -90 <= lat <= 90 or not -180 <= lon <= 180 or radius_m <= 0:
            raise ValueError
    except (TypeError, ValueError):
        return Response(
            {"error": "Provide valid coordinates and a positive radius_m."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    try:
        results = find_nearby_forecasts(lat, lon, radius_m)
    except Exception:
        return Response(
            {"error": "Nearby forecast lookup failed."},
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    return Response({"count": len(results), "results": results}, status=status.HTTP_200_OK)
