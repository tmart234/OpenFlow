import argparse
import logging
from datetime import datetime

import pandas as pd

from data.utils import data_utils

"""
14-day daily TMIN/TMAX + precipitation forecast for a point.

The model design (see the wiki + the Phase 2 plan) needs forecast TMIN/TMAX
*and* daily precipitation as inputs to the model's forecast window. The
official NWS api.weather.gov forecast caps at ~7 days, so we fetch from
Open-Meteo's free public API, which serves GFS-based daily forecasts out to
16 days -- this is the "GFS / outlooks to reach 14 days" path the user picked.

Training uses *actual* (shifted) historical TMIN/TMAX/precipitation in the
forecast window; this module is what an inference run will call. Importantly,
this is point data only -- soil moisture / SWE are current-conditions
features, not things this fetcher produces.

Units: TMIN/TMAX default to Fahrenheit for backwards compatibility (NOTE: raw
GHCND / OpenFlow training data is in tenths of °C, NOT Fahrenheit — callers
feeding the ML model must pass `temperature_unit='celsius'` and multiply
TMIN/TMAX by 10, as `ml_inference.forecast_weather` does). Precipitation is in
millimetres (matches GHCND PRCP after combine_data's tenths-of-mm conversion).
"""

if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
DEFAULT_HORIZON_DAYS = 14
# Open-Meteo's free public API tops out at 16 days.
MAX_HORIZON_DAYS = 16


def _empty_forecast():
    return pd.DataFrame(columns=['Date', 'TMIN', 'TMAX', 'precipitation'])


def get_open_meteo_forecast(lat, lon, days=DEFAULT_HORIZON_DAYS, temperature_unit='fahrenheit'):
    """
    Fetch a daily TMIN/TMAX + precipitation forecast from Open-Meteo for the
    next `days` days starting today. Default temperature unit is Fahrenheit for
    backwards compatibility; note that GHCND / ML training temperatures are in
    tenths of °C (use `temperature_unit='celsius'` and multiply by 10 for model
    inputs). Precipitation is in mm (Open-Meteo's `precipitation_sum` default),
    matching the GHCND PRCP unit after combine_data's tenths-of-mm conversion.
    """
    if days < 1 or days > MAX_HORIZON_DAYS:
        raise ValueError(f"days must be in 1..{MAX_HORIZON_DAYS}")
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "forecast_days": days,
        "temperature_unit": temperature_unit,
        "timezone": "America/Denver",
    }
    response = data_utils.request_with_retry(OPEN_METEO_URL, params=params)
    if response is None:
        return _empty_forecast()
    try:
        payload = response.json()
    except ValueError:
        logger.error("Open-Meteo returned a non-JSON response")
        return _empty_forecast()

    daily = payload.get('daily') or {}
    dates = daily.get('time') or []
    tmax = daily.get('temperature_2m_max') or []
    tmin = daily.get('temperature_2m_min') or []
    precip = daily.get('precipitation_sum') or []
    if not dates or len(dates) != len(tmax) or len(dates) != len(tmin):
        logger.error("Unexpected Open-Meteo payload shape")
        return _empty_forecast()
    # precipitation_sum is the newest of the three fields requested; tolerate
    # an older Open-Meteo deployment that doesn't return it by zero-filling.
    if len(precip) != len(dates):
        logger.warning("Open-Meteo returned no precipitation_sum; defaulting to 0")
        precip = [0.0] * len(dates)

    return pd.DataFrame({
        'Date': dates,
        'TMIN': tmin,
        'TMAX': tmax,
        'precipitation': precip,
    })


def get_forecast(lat, lon, days=DEFAULT_HORIZON_DAYS):
    """
    Public API: a 14-day daily TMIN/TMAX + precipitation forecast for
    (lat, lon).

    Currently always Open-Meteo (GFS-backed). Kept as a thin wrapper so a
    future api.weather.gov-primary / Open-Meteo-fallback blend can drop in
    without touching callers.
    """
    return get_open_meteo_forecast(lat, lon, days=days)


OPEN_METEO_ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"


def _hourly_precip_intensity_daily(hourly: dict) -> pd.DataFrame:
    """Collapse hourly precipitation (mm/h) and VPD (kPa) into daily stats."""
    times = hourly.get('time') or []
    if not times:
        return pd.DataFrame(columns=['Date', 'vpd', 'precip_max1h', 'precip_max3h', 'precip_wet_hours'])
    h = pd.DataFrame({'time': pd.to_datetime(times)})
    h['Date'] = h['time'].dt.strftime('%Y-%m-%d')
    precip = pd.to_numeric(pd.Series(hourly.get('precipitation') or [0.0] * len(times)),
                           errors='coerce').fillna(0.0)
    h['precip'] = precip.clip(lower=0.0)
    h['p3h'] = h['precip'].rolling(3, min_periods=1).sum()
    h['wet'] = (h['precip'] >= 0.5).astype(float)
    vpd_raw = hourly.get('vapour_pressure_deficit')
    h['vpd'] = (pd.to_numeric(pd.Series(vpd_raw), errors='coerce')
                if vpd_raw is not None and len(vpd_raw) == len(times) else float('nan'))
    g = h.groupby('Date', as_index=False).agg(
        vpd=('vpd', 'mean'),
        precip_max1h=('precip', 'max'),
        precip_max3h=('p3h', 'max'),
        precip_wet_hours=('wet', 'sum'),
    )
    return g


def get_open_meteo_extended_forecast(lat, lon, days=DEFAULT_HORIZON_DAYS,
                                     temperature_unit='celsius',
                                     timezone='America/Denver') -> pd.DataFrame:
    """
    Single-call 14-day Open-Meteo forecast returning weather, energy balance
    (`srad` MJ/m^2/day, `pet` mm/day, `vpd` kPa), and sub-daily precipitation
    intensity (`precip_max1h`, `precip_max3h`, `precip_wet_hours`).
    Temperatures are in `temperature_unit` (caller multiplies celsius by 10
    for training tenths-of-deg-C units).
    """
    if days < 1 or days > MAX_HORIZON_DAYS:
        raise ValueError(f"days must be in 1..{MAX_HORIZON_DAYS}")
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ("temperature_2m_max,temperature_2m_min,precipitation_sum,"
                  "shortwave_radiation_sum,et0_fao_evapotranspiration"),
        "hourly": "precipitation,vapour_pressure_deficit",
        "forecast_days": days,
        "temperature_unit": temperature_unit,
        "timezone": timezone,
    }
    response = data_utils.request_with_retry(OPEN_METEO_URL, params=params)
    if response is None:
        return pd.DataFrame()
    try:
        payload = response.json()
    except ValueError:
        logger.error("Open-Meteo extended forecast returned non-JSON")
        return pd.DataFrame()
    daily = payload.get('daily') or {}
    dates = daily.get('time') or []
    if not dates:
        return pd.DataFrame()
    n = len(dates)
    out = pd.DataFrame({
        'Date': dates,
        'TMIN': daily.get('temperature_2m_min') or [float('nan')] * n,
        'TMAX': daily.get('temperature_2m_max') or [float('nan')] * n,
        'precipitation': daily.get('precipitation_sum') or [0.0] * n,
        'srad': daily.get('shortwave_radiation_sum') or [float('nan')] * n,
        'pet': daily.get('et0_fao_evapotranspiration') or [float('nan')] * n,
    })
    hourly_df = _hourly_precip_intensity_daily(payload.get('hourly') or {})
    if not hourly_df.empty:
        out = out.merge(hourly_df, on='Date', how='left')
    else:
        for col in ('vpd', 'precip_max1h', 'precip_max3h', 'precip_wet_hours'):
            out[col] = float('nan')
    return out


def get_open_meteo_ensemble(lat, lon, days=DEFAULT_HORIZON_DAYS,
                            models='gfs_seamless',
                            timezone='America/Denver') -> pd.DataFrame:
    """
    Fetch multi-member GFS ensemble forecasts from Open-Meteo's public
    Ensemble API (31 members, zero auth) and summarize daily spread:
      Date, precip_ens_mean (mm), precip_ens_std (mm), precip_ens_p90 (mm),
      temp_ens_std (tenths of deg C).
    """
    import numpy as np
    if days < 1 or days > MAX_HORIZON_DAYS:
        raise ValueError(f"days must be in 1..{MAX_HORIZON_DAYS}")
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "precipitation_sum,temperature_2m_max",
        "models": models,
        "forecast_days": days,
        "timezone": timezone,
    }
    response = data_utils.request_with_retry(OPEN_METEO_ENSEMBLE_URL, params=params)
    if response is None:
        return pd.DataFrame(columns=['Date', 'precip_ens_mean', 'precip_ens_std',
                                     'precip_ens_p90', 'temp_ens_std'])
    try:
        payload = response.json()
    except ValueError:
        logger.error("Open-Meteo ensemble returned non-JSON")
        return pd.DataFrame(columns=['Date', 'precip_ens_mean', 'precip_ens_std',
                                     'precip_ens_p90', 'temp_ens_std'])
    daily = payload.get('daily') or {}
    dates = daily.get('time') or []
    if not dates:
        return pd.DataFrame(columns=['Date', 'precip_ens_mean', 'precip_ens_std',
                                     'precip_ens_p90', 'temp_ens_std'])
    p_cols = [v for k, v in daily.items() if k.startswith('precipitation_sum') and isinstance(v, list) and len(v) == len(dates)]
    t_cols = [v for k, v in daily.items() if k.startswith('temperature_2m_max') and isinstance(v, list) and len(v) == len(dates)]
    if not p_cols:
        return pd.DataFrame(columns=['Date', 'precip_ens_mean', 'precip_ens_std',
                                     'precip_ens_p90', 'temp_ens_std'])
    p_arr = np.asarray([[float(x) if x is not None else np.nan for x in row] for row in p_cols], dtype=float)
    t_arr = (np.asarray([[float(x) if x is not None else np.nan for x in row] for row in t_cols], dtype=float)
             if t_cols else None)
    return pd.DataFrame({
        'Date': dates,
        'precip_ens_mean': np.nanmean(p_arr, axis=0),
        'precip_ens_std': np.nanstd(p_arr, axis=0),
        'precip_ens_p90': np.nanpercentile(p_arr, 90, axis=0),
        # Match tenths-of-deg-C training scale for temperature spread.
        'temp_ens_std': (np.nanstd(t_arr, axis=0) * 10.0 if t_arr is not None else np.zeros(len(dates))),
    })


def main(lat, lon, days=DEFAULT_HORIZON_DAYS):
    df = get_forecast(lat, lon, days=days)
    data_utils.preview_data(df)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetch a daily temperature forecast for a point.')
    parser.add_argument('--lat', type=float, required=True)
    parser.add_argument('--lon', type=float, required=True)
    parser.add_argument('--days', type=int, default=DEFAULT_HORIZON_DAYS,
                        help=f'Number of forecast days (1..{MAX_HORIZON_DAYS})')
    args = parser.parse_args()
    main(args.lat, args.lon, args.days)
