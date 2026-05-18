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

Units: TMIN/TMAX in Fahrenheit (matches GHCND TMIN/TMAX), precipitation in
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
    next `days` days starting today. Default temperature unit is Fahrenheit to
    match the GHCND TMIN/TMAX feature scale used during training. Precipitation
    is in mm (Open-Meteo's `precipitation_sum` default), matching the GHCND
    PRCP unit after combine_data's tenths-of-mm conversion.
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
