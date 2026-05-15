from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    wave_height_max_m: float = Field(..., gt=0, description="Maximum significant wave height (m)")
    wave_height_mean_m: float = Field(..., gt=0, description="Mean significant wave height (m)")
    wave_period_mean_s: float = Field(..., gt=0, description="Mean peak wave period (s)")
    wave_direction_deg: float = Field(..., ge=0, lt=360, description="Mean wave direction in degrees (0=N, 90=E)")

    wind_speed_ms: float = Field(..., ge=0, description="Mean wind speed (m/s)")
    wind_gust_ms: float = Field(..., ge=0, description="Peak wind gust (m/s)")
    wind_dir_mean_deg: float = Field(..., ge=0, lt=360, description="Mean wind direction in degrees")

    rain_today_mm: float = Field(0.0, ge=0, description="Rainfall today (mm)")
    rain_yesterday_mm: float = Field(0.0, ge=0, description="Rainfall yesterday (mm)")
    rain_two_days_ago_mm: float = Field(0.0, ge=0, description="Rainfall two days ago (mm)")

    tide_max_m: float = Field(..., description="Maximum tidal height, MLLW datum (m)")
    tide_mean_m: float = Field(..., description="Mean tidal height, MLLW datum (m)")

    forecast_date: Optional[date] = Field(
        None,
        description="Date for season encoding as YYYY-MM-DD. Defaults to today if omitted.",
    )

    sea_surface_temp_c: float = Field(0.0, description="Sea surface temperature (°C) — hindcast, defaults 0")
    sst_change_24h: float = Field(0.0, description="SST change vs 24h prior (°C) — hindcast, defaults 0")
    sst_change_48h: float = Field(0.0, description="SST change vs 48h prior (°C) — hindcast, defaults 0")
    wave_peak_psd_max: float = Field(0.0, ge=0, description="Peak wave power spectral density — hindcast, defaults 0")

    wave_height_mean_lag1: float = Field(0.0, ge=0, description="Mean wave height 1 day prior (m)")
    wave_height_mean_lag2: float = Field(0.0, ge=0, description="Mean wave height 2 days prior (m)")
    wave_height_mean_lag3: float = Field(0.0, ge=0, description="Mean wave height 3 days prior (m)")
    wave_height_max_lag1: float = Field(0.0, ge=0, description="Max wave height 1 day prior (m)")
    wave_height_max_lag2: float = Field(0.0, ge=0, description="Max wave height 2 days prior (m)")
    wave_height_max_lag3: float = Field(0.0, ge=0, description="Max wave height 3 days prior (m)")
    wind_speed_lag1: float = Field(0.0, ge=0, description="Wind speed 1 day prior (m/s)")
    wind_speed_lag2: float = Field(0.0, ge=0, description="Wind speed 2 days prior (m/s)")
    wind_speed_lag3: float = Field(0.0, ge=0, description="Wind speed 3 days prior (m/s)")
    wave_peak_psd_lag1: float = Field(0.0, ge=0, description="Peak PSD 1 day prior")
    wave_peak_psd_lag2: float = Field(0.0, ge=0, description="Peak PSD 2 days prior")
    wave_peak_psd_lag3: float = Field(0.0, ge=0, description="Peak PSD 3 days prior")
    rain_72h_weighted_lag1: float = Field(0.0, ge=0, description="Weighted 72h rain 1 day prior (mm)")
    rain_72h_weighted_lag2: float = Field(0.0, ge=0, description="Weighted 72h rain 2 days prior (mm)")
    rain_72h_weighted_lag3: float = Field(0.0, ge=0, description="Weighted 72h rain 3 days prior (mm)")
    swell_trend_3d: float = Field(0.0, description="3-day wave height trend (waveHs diff over 3 days)")


class PredictionResponse(BaseModel):
    visibility_feet: float
    go_no_go: bool
    go_probability: float
    condition: Literal["Poor", "Fair", "Good", "Excellent"]
    condition_probabilities: dict[str, float]
    hindcast_features_used: bool
