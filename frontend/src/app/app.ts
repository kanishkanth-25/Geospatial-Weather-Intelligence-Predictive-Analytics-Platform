import { AfterViewInit, Component, OnDestroy, inject, signal } from '@angular/core';
import { DecimalPipe } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import * as L from 'leaflet';

interface HourlyTemperature { time: string; temperature: number }
interface DailyOutlook {
  date: string;
  temperature_min: number;
  temperature_max: number;
  humidity_mean: number;
  precipitation_sum: number;
  rain_sum: number;
  wind_speed_max: number;
  weather_description: string;
}
interface ForecastResponse {
  lat: number; lon: number;
  current: { time: string; temperature: number; relative_humidity: number; precipitation: number; rain: number; wind_speed: number; weather_description: string };
  observed_last_24h: HourlyTemperature[];
  predicted_next_24h: HourlyTemperature[];
  provider_forecast_next_24h: HourlyTemperature[];
  provider_outlook_next_7_days: DailyOutlook[];
  model_status: string; model_version: string; source: string;
}

@Component({ selector: 'app-root', standalone: true, imports: [DecimalPipe], templateUrl: './app.html', styleUrls: ['./app.css', './weather-outlook.css'] })
export class App implements AfterViewInit, OnDestroy {
  private readonly http = inject(HttpClient);
  private map?: L.Map;
  private marker?: L.Marker;
  loading = signal(false);
  error = signal('');
  forecast = signal<ForecastResponse | undefined>(undefined);
  selectedPoint = signal<{ lat: number; lon: number } | undefined>(undefined);

  ngAfterViewInit(): void {
    this.map = L.map('weather-map', { zoomControl: false }).setView([20, 0], 2);
    L.control.zoom({ position: 'bottomright' }).addTo(this.map);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19, attribution: '&copy; OpenStreetMap contributors' }).addTo(this.map);
    this.map.on('click', (event: L.LeafletMouseEvent) => {
      const point = { lat: event.latlng.lat, lon: event.latlng.lng };
      this.selectedPoint.set(point);
      this.error.set('');
      this.marker?.remove();
      const icon = L.divIcon({
        className: 'weather-pin-wrap',
        html: '<span class="weather-pin"></span>',
        iconSize: [22, 22],
        iconAnchor: [11, 11],
      });
      this.marker = L.marker([point.lat, point.lon], { icon }).addTo(this.map!);
      this.loadForecast(point.lat, point.lon);
    });
    setTimeout(() => this.map?.invalidateSize(), 0);
  }

  ngOnDestroy(): void { this.map?.remove(); }

  private loadForecast(lat: number, lon: number): void {
    this.loading.set(true);
    this.forecast.set(undefined);
    this.http.post<ForecastResponse>('/api/forecast/predict/', { lat, lon }).subscribe({
      next: (forecast) => { this.forecast.set(forecast); this.loading.set(false); },
      error: (response) => {
        this.error.set(response.error?.error ?? 'Could not load the forecast. Check that Django and MongoDB are running.');
        this.loading.set(false);
      },
    });
  }

  get localTimeLabel(): string {
    const forecast = this.forecast();
    return forecast?.current.time
      ? new Date(`${forecast.current.time}Z`).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' })
      : '';
  }
  formatHour(value: string): string { return new Date(value).toLocaleTimeString([], { hour: 'numeric' }); }
  formatDate(value: string): string { return new Date(`${value}T12:00:00Z`).toLocaleDateString([], { weekday: 'short', month: 'short', day: 'numeric' }); }
  formatLatitude(value: number): string { return `${Math.abs(value).toFixed(4)}° ${value < 0 ? 'S' : 'N'}`; }
  formatLongitude(value: number): string { return `${Math.abs(value).toFixed(4)}° ${value < 0 ? 'W' : 'E'}`; }
}
