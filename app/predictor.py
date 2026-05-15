import math
from datetime import date
from typing import Optional

import numpy as np
import xgboost as xgb

from .config import settings
from .schemas import PredictionRequest, PredictionResponse

FEATURE_ORDER = [
    "waveHs_max",
    "waveHs_mean",
    "waveTp_mean",
    "wave_steepness_mean",
    "swell_energy_mean",
    "waveDp_sine_mean",
    "waveDp_cos_mean",
    "wavePeakPSD_max",
    "sstSeaSurfaceTemperature",
    "sst_diff_24h",
    "sst_diff_48h",
    "wind_speed",
    "wind_gust",
    "wind_dir_mean",
    "rain_mm",
    "rain_72h_weighted_mm",
    "tide_max",
    "tide_mean",
    "season_sine",
    "season_cos",
    "waveHs_mean_lag1",
    "waveHs_mean_lag2",
    "waveHs_mean_lag3",
    "waveHs_max_lag1",
    "waveHs_max_lag2",
    "waveHs_max_lag3",
    "wind_speed_lag1",
    "wind_speed_lag2",
    "wind_speed_lag3",
    "wavePeakPSD_max_lag1",
    "wavePeakPSD_max_lag2",
    "wavePeakPSD_max_lag3",
    "rain_72h_weighted_mm_lag1",
    "rain_72h_weighted_mm_lag2",
    "rain_72h_weighted_mm_lag3",
    "swell_trend_3d",
]

CLASS_LABELS = {0: "Poor", 1: "Fair", 2: "Good", 3: "Excellent"}

_reg_model: Optional[xgb.Booster] = None
_bin_model: Optional[xgb.Booster] = None
_cls_model: Optional[xgb.Booster] = None


def load_models() -> None:
    global _reg_model, _bin_model, _cls_model
    _reg_model = xgb.Booster()
    _reg_model.load_model(str(settings.model_dir / "regression_cove_model.json"))
    _bin_model = xgb.Booster()
    _bin_model.load_model(str(settings.model_dir / "binary_cove_model.json"))
    _cls_model = xgb.Booster()
    _cls_model.load_model(str(settings.model_dir / "fourClass_cove_model.json"))


def _build_features(req: PredictionRequest) -> tuple[dict[str, float], bool]:
    d = req.forecast_date or date.today()

    day_of_year = d.timetuple().tm_yday
    season_sine = math.sin(2 * math.pi * day_of_year / 365)
    season_cos = math.cos(2 * math.pi * day_of_year / 365)

    hs = req.wave_height_mean_m
    tp = req.wave_period_mean_s
    wave_steepness = hs / tp if tp > 0 else 0.0
    swell_energy = (hs**2) * tp

    wave_dir_rad = math.radians(req.wave_direction_deg)
    waveDp_sine = math.sin(wave_dir_rad)
    waveDp_cos = math.cos(wave_dir_rad)

    rain_72h = (
        req.rain_today_mm
        + req.rain_yesterday_mm * 0.6
        + req.rain_two_days_ago_mm * 0.3
    )

    hindcast_used = any([
        req.sea_surface_temp_c != 0.0,
        req.sst_change_24h != 0.0,
        req.sst_change_48h != 0.0,
        req.wave_peak_psd_max != 0.0,
        req.wave_height_mean_lag1 != 0.0,
        req.wave_height_mean_lag2 != 0.0,
        req.wave_height_mean_lag3 != 0.0,
        req.wave_height_max_lag1 != 0.0,
        req.wave_height_max_lag2 != 0.0,
        req.wave_height_max_lag3 != 0.0,
        req.wind_speed_lag1 != 0.0,
        req.wind_speed_lag2 != 0.0,
        req.wind_speed_lag3 != 0.0,
        req.wave_peak_psd_lag1 != 0.0,
        req.wave_peak_psd_lag2 != 0.0,
        req.wave_peak_psd_lag3 != 0.0,
        req.rain_72h_weighted_lag1 != 0.0,
        req.rain_72h_weighted_lag2 != 0.0,
        req.rain_72h_weighted_lag3 != 0.0,
    ])

    features = {
        "waveHs_max": req.wave_height_max_m,
        "waveHs_mean": hs,
        "waveTp_mean": tp,
        "wave_steepness_mean": wave_steepness,
        "swell_energy_mean": swell_energy,
        "waveDp_sine_mean": waveDp_sine,
        "waveDp_cos_mean": waveDp_cos,
        "wavePeakPSD_max": req.wave_peak_psd_max,
        "sstSeaSurfaceTemperature": req.sea_surface_temp_c,
        "sst_diff_24h": req.sst_change_24h,
        "sst_diff_48h": req.sst_change_48h,
        "wind_speed": req.wind_speed_ms,
        "wind_gust": req.wind_gust_ms,
        "wind_dir_mean": req.wind_dir_mean_deg,
        "rain_mm": req.rain_today_mm,
        "rain_72h_weighted_mm": rain_72h,
        "tide_max": req.tide_max_m,
        "tide_mean": req.tide_mean_m,
        "season_sine": season_sine,
        "season_cos": season_cos,
        "waveHs_mean_lag1": req.wave_height_mean_lag1,
        "waveHs_mean_lag2": req.wave_height_mean_lag2,
        "waveHs_mean_lag3": req.wave_height_mean_lag3,
        "waveHs_max_lag1": req.wave_height_max_lag1,
        "waveHs_max_lag2": req.wave_height_max_lag2,
        "waveHs_max_lag3": req.wave_height_max_lag3,
        "wind_speed_lag1": req.wind_speed_lag1,
        "wind_speed_lag2": req.wind_speed_lag2,
        "wind_speed_lag3": req.wind_speed_lag3,
        "wavePeakPSD_max_lag1": req.wave_peak_psd_lag1,
        "wavePeakPSD_max_lag2": req.wave_peak_psd_lag2,
        "wavePeakPSD_max_lag3": req.wave_peak_psd_lag3,
        "rain_72h_weighted_mm_lag1": req.rain_72h_weighted_lag1,
        "rain_72h_weighted_mm_lag2": req.rain_72h_weighted_lag2,
        "rain_72h_weighted_mm_lag3": req.rain_72h_weighted_lag3,
        "swell_trend_3d": req.swell_trend_3d,
    }

    return features, hindcast_used


def validate_models() -> None:
    dummy = PredictionRequest(
        wave_height_max_m=0.5,
        wave_height_mean_m=0.3,
        wave_period_mean_s=10.0,
        wave_direction_deg=270.0,
        wind_speed_ms=3.0,
        wind_gust_ms=5.0,
        wind_dir_mean_deg=270.0,
        tide_max_m=1.0,
        tide_mean_m=0.5,
    )
    result = run_prediction(dummy)

    if not math.isfinite(result.visibility_feet):
        raise RuntimeError(f"Regression model returned non-finite value: {result.visibility_feet}")

    if not (0.0 <= result.go_probability <= 1.0):
        raise RuntimeError(f"Binary model returned out-of-range probability: {result.go_probability}")

    prob_sum = sum(result.condition_probabilities.values())
    if not (0.99 <= prob_sum <= 1.01):
        raise RuntimeError(f"4-class model probabilities do not sum to 1.0: {prob_sum:.4f}")


def run_prediction(req: PredictionRequest) -> PredictionResponse:
    features, hindcast_used = _build_features(req)

    row = np.array([[features[f] for f in FEATURE_ORDER]], dtype=np.float32)
    dmatrix = xgb.DMatrix(row, feature_names=FEATURE_ORDER)

    reg_pred = float(_reg_model.predict(dmatrix)[0])
    bin_prob = float(_bin_model.predict(dmatrix)[0])
    cls_probs = _cls_model.predict(dmatrix)[0]
    cls_class = int(np.argmax(cls_probs))

    return PredictionResponse(
        visibility_feet=round(max(0.0, reg_pred), 1),
        go_no_go=bin_prob >= settings.go_threshold,
        go_probability=round(bin_prob, 4),
        condition=CLASS_LABELS[cls_class],
        condition_probabilities={
            CLASS_LABELS[i]: round(float(p), 4) for i, p in enumerate(cls_probs)
        },
        hindcast_features_used=hindcast_used,
    )
