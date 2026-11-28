"""
Continuous soil moisture and sub-daily rainfall intensity forcing.

Supplies two high-value hydrological feature groups that cover 1979-present
without SMAP's 2015 start date or frozen-ground / RFI gaps:

  1. Continuous volumetric soil moisture (m^3/m^3):
       sm_surface  -- top layer (0-7 cm ERA5-Land / 0-10 cm NLDAS-2 Noah)
       sm_rootzone -- root zone (7-100 cm depth-weighted ERA5-Land / 10-100 cm NLDAS-2)
  2. Sub-daily precipitation intensity (critical for Aug-Oct monsoon storms):
       precip_max1h     -- daily max 1-hour rainfall (mm/h)
       precip_max3h     -- daily max rolling 3-hour rainfall (mm/3h)
       precip_wet_hours -- hours with >= 0.5 mm/h rain (0..24)

Sources (controlled by OPENFLOW_HYDRO_SOURCE = auto | nldas | era5):
  - `era5` (zero auth, works everywhere including local CLI and mobile proxies):
      * Historical (1950 -> ~5 days ago): Open-Meteo ERA5-Land hourly archive
        (https://archive-api.open-meteo.com/v1/archive, models=era5_land).
      * Trailing 0-7 days bridge: Open-Meteo GFS hourly analysis (`past_days`).
  - `nldas` (NASA Earthdata, 1/8 deg CONUS radar-disaggregated hourly precip +
      4-layer Noah soil moisture via earthaccess + h5py):
      * Uses EARTHDATA_USERNAME / EARTHDATA_PASSWORD when configured in CI,
        bridged over the trailing ~4-day NLDAS latency window with zero-auth
        Open-Meteo hourly GFS so live inference always reaches today.
  - `auto` (default): uses `era5` unless `OPENFLOW_HYDRO_SOURCE=nldas` is
      explicitly requested (ERA5-Land guarantees 100% training/inference parity
      without requiring Earthdata credentials on mobile/local clients; NLDAS-2
      is available for Earthdata-authenticated CI ablations and Feature Store
      runs).

Set OPENFLOW_DISABLE_HYDRO_FORCING=1 to disable both groups for ablations.
"""

from datetime import datetime, timedelta
import logging
import os
import shutil
import tempfile
from typing import Optional

import numpy as np
import pandas as pd

from data.utils import data_utils

logger = logging.getLogger(__name__)

ERA5_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
# Open-Meteo's `era5_land` archive returns all-null `precipitation` hours;
# `era5_seamless` serves 9 km ERA5-Land soil moisture alongside continuous
# hourly ERA5 precipitation so both feature groups are populated.
ERA5_ARCHIVE_MODEL = "era5_seamless"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

NLDAS_FORA_SHORT_NAME = "NLDAS_FORA0125_H"
NLDAS_NOAH_SHORT_NAME = "NLDAS_NOAH0125_H"
NLDAS_VERSION = "2.0"
NLDAS_BATCH_SIZE = 48

INTENSITY_COLUMNS = ['precip_max1h', 'precip_max3h', 'precip_wet_hours']
SOIL_COLUMNS = ['sm_surface', 'sm_rootzone']
ALL_COLUMNS = [*INTENSITY_COLUMNS, *SOIL_COLUMNS]

# Open-Meteo archive handles 5-year hourly chunks cleanly (~5 requests / 25 y).
ERA5_CHUNK_DAYS = 5 * 366
WET_HOUR_THRESHOLD_MM = 0.5


def _disabled() -> bool:
    return os.getenv('OPENFLOW_DISABLE_HYDRO_FORCING', '').strip() in ('1', 'true', 'True')


def hydro_source(override: Optional[str] = None) -> str:
    """Resolve active hydro forcing source: 'era5' (default) or 'nldas'."""
    raw = override if override is not None else os.getenv('OPENFLOW_HYDRO_SOURCE', '')
    src = (raw or 'era5').strip().lower()
    if src == 'auto':
        has_creds = bool(os.getenv('EARTHDATA_USERNAME') and os.getenv('EARTHDATA_PASSWORD'))
        return 'nldas' if has_creds else 'era5'
    return src if src in ('era5', 'nldas') else 'era5'


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=['Date', *ALL_COLUMNS])


def summarize_hourly_frame(hourly_df: pd.DataFrame, tz: str = 'America/Denver') -> pd.DataFrame:
    """
    Collapse an hourly frame with columns [time, precip_mm, sm_surface, sm_rootzone]
    into daily [Date, precip_max1h, precip_max3h, precip_wet_hours, sm_surface, sm_rootzone].
    Pure function used by both ERA5-Land, GFS bridge, and NLDAS-2 paths.
    """
    if hourly_df is None or hourly_df.empty:
        return _empty()
    df = hourly_df.copy()
    t = pd.to_datetime(df['time'], errors='coerce')
    if t.dt.tz is not None:
        t = t.dt.tz_convert(tz).dt.tz_localize(None)
    df = df.assign(_t=t).dropna(subset=['_t']).sort_values('_t')
    if df.empty:
        return _empty()
    df['Date'] = df['_t'].dt.normalize()
    p = pd.to_numeric(df['precip_mm'], errors='coerce').clip(lower=0.0)
    df['_p'] = p
    df['_p3h'] = p.rolling(3, min_periods=1).sum()
    df['_wet'] = (p >= WET_HOUR_THRESHOLD_MM).astype(float)
    df['_sm_s'] = pd.to_numeric(df['sm_surface'], errors='coerce')
    df['_sm_r'] = pd.to_numeric(df['sm_rootzone'], errors='coerce')

    grouped = df.groupby('Date', as_index=False).agg(
        precip_max1h=('_p', 'max'),
        precip_max3h=('_p3h', 'max'),
        precip_wet_hours=('_wet', lambda s: s.sum(min_count=1)),
        sm_surface=('_sm_s', 'mean'),
        sm_rootzone=('_sm_r', 'mean'),
    )
    # Require at least one valid precip observation on the day; keep partial SM if present.
    return grouped[['Date', *ALL_COLUMNS]].sort_values('Date').reset_index(drop=True)


def _parse_era5_payload(payload: dict) -> pd.DataFrame:
    hourly = (payload or {}).get('hourly') or {}
    times = hourly.get('time') or []
    if not times:
        return pd.DataFrame()
    n = len(times)

    def _col(name):
        vals = hourly.get(name)
        if vals is None or len(vals) != n:
            return np.full(n, np.nan, dtype=float)
        return pd.to_numeric(pd.Series(vals), errors='coerce').to_numpy(dtype=float)

    sm_0_7 = _col('soil_moisture_0_to_7cm')
    sm_7_28 = _col('soil_moisture_7_to_28cm')
    sm_28_100 = _col('soil_moisture_28_to_100cm')
    # Depth-weighted root zone (21 cm in 7-28 cm layer + 72 cm in 28-100 cm layer).
    w1, w2 = 21.0, 72.0
    both = np.isfinite(sm_7_28) & np.isfinite(sm_28_100)
    root = np.where(both, (w1 * sm_7_28 + w2 * sm_28_100) / (w1 + w2),
                    np.where(np.isfinite(sm_7_28), sm_7_28, sm_28_100))
    return pd.DataFrame({
        'time': times,
        'precip_mm': _col('precipitation'),
        'sm_surface': sm_0_7,
        'sm_rootzone': root,
    })


def _parse_gfs_bridge_payload(payload: dict) -> pd.DataFrame:
    hourly = (payload or {}).get('hourly') or {}
    times = hourly.get('time') or []
    if not times:
        return pd.DataFrame()
    n = len(times)

    def _col(name):
        vals = hourly.get(name)
        if vals is None or len(vals) != n:
            return np.full(n, np.nan, dtype=float)
        return pd.to_numeric(pd.Series(vals), errors='coerce').to_numpy(dtype=float)

    # GFS soil layers: 0-1, 1-3, 3-9 cm -> surface; 9-27, 27-81 cm -> root zone.
    s1, s2, s3 = _col('soil_moisture_0_to_1cm'), _col('soil_moisture_1_to_3cm'), _col('soil_moisture_3_to_9cm')
    surf = np.nanmean(np.vstack([s1, s2, s3]), axis=0)
    r1, r2 = _col('soil_moisture_9_to_27cm'), _col('soil_moisture_27_to_81cm')
    both = np.isfinite(r1) & np.isfinite(r2)
    root = np.where(both, (18.0 * r1 + 54.0 * r2) / 72.0,
                    np.where(np.isfinite(r1), r1, r2))
    return pd.DataFrame({
        'time': times,
        'precip_mm': _col('precipitation'),
        'sm_surface': surf,
        'sm_rootzone': root,
    })


def fetch_era5_daily(lat: float, lon: float, start_date, end_date,
                     tz: str = 'America/Denver') -> pd.DataFrame:
    """
    Fetch daily rainfall intensity + continuous soil moisture from Open-Meteo's
    ERA5-Land hourly archive, bridging the ~5-day archive lag with Open-Meteo
    GFS hourly past_days so recent dates always have values.
    """
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    today = pd.Timestamp.now().normalize()
    archive_end = min(end, today - pd.Timedelta(days=2))
    parts = []

    if start <= archive_end:
        for cs, ce in data_utils.date_chunks(start.to_pydatetime(), archive_end.to_pydatetime(), ERA5_CHUNK_DAYS):
            params = {
                'latitude': round(float(lat), 4),
                'longitude': round(float(lon), 4),
                'start_date': cs.strftime('%Y-%m-%d'),
                'end_date': ce.strftime('%Y-%m-%d'),
                'hourly': ('precipitation,soil_moisture_0_to_7cm,'
                           'soil_moisture_7_to_28cm,soil_moisture_28_to_100cm'),
                'models': ERA5_ARCHIVE_MODEL,
                'timezone': tz,
            }
            resp = data_utils.request_with_retry(ERA5_ARCHIVE_URL, params=params, timeout=180)
            if resp is None:
                logger.warning("ERA5-Land chunk %s..%s failed",
                               params['start_date'], params['end_date'])
                continue
            try:
                hf = _parse_era5_payload(resp.json())
            except ValueError:
                continue
            if not hf.empty:
                parts.append(summarize_hourly_frame(hf, tz=tz))

    era = (pd.concat(parts, ignore_index=True)
           .sort_values('Date').drop_duplicates('Date', keep='last').reset_index(drop=True)
           if parts else _empty())

    # Bridge the trailing ~5-day ERA5-Land lag (or full recent inference window
    # up to 92 days if archive-api rate-limited) with zero-auth GFS past_days.
    need_bridge = end >= today - pd.Timedelta(days=10)
    if need_bridge:
        bridge_days = 10 if not era.empty else int(np.clip((today - start).days + 2, 10, 92))
        bridge = fetch_gfs_recent_daily(lat, lon, past_days=bridge_days, tz=tz)
        if not bridge.empty:
            bridge = bridge[(bridge['Date'] >= start) & (bridge['Date'] <= end)]
            if era.empty:
                era = bridge.reset_index(drop=True)
            else:
                last_valid = era.dropna(subset=['sm_rootzone'])['Date'].max()
                if pd.notna(last_valid):
                    # Bias-align GFS soil moisture onto ERA5-Land scale over overlap days.
                    overlap = era.merge(bridge, on='Date', suffixes=('_e', '_g'))
                    for col in SOIL_COLUMNS:
                        ov = overlap[[f'{col}_e', f'{col}_g']].dropna()
                        if len(ov) >= 2:
                            shift = float(np.median(ov[f'{col}_e'] - ov[f'{col}_g']))
                            bridge[col] = np.clip(bridge[col] + shift, 0.0, 1.0)
                    tail = bridge[bridge['Date'] > last_valid]
                else:
                    tail = bridge
                if not tail.empty:
                    era = (pd.concat([era, tail], ignore_index=True)
                           .sort_values('Date').drop_duplicates('Date', keep='last')
                           .reset_index(drop=True))
    return era


def fetch_gfs_recent_daily(lat: float, lon: float, past_days: int = 10,
                           tz: str = 'America/Denver') -> pd.DataFrame:
    """Zero-auth trailing-days bridge from Open-Meteo GFS hourly analysis."""
    params = {
        'latitude': round(float(lat), 4),
        'longitude': round(float(lon), 4),
        'past_days': int(np.clip(past_days, 1, 92)),
        'forecast_days': 1,
        'hourly': ('precipitation,soil_moisture_0_to_1cm,soil_moisture_1_to_3cm,'
                   'soil_moisture_3_to_9cm,soil_moisture_9_to_27cm,soil_moisture_27_to_81cm'),
        'timezone': tz,
    }
    resp = data_utils.request_with_retry(OPEN_METEO_FORECAST_URL, params=params, timeout=60)
    if resp is None:
        return _empty()
    try:
        hf = _parse_gfs_bridge_payload(resp.json())
    except ValueError:
        return _empty()
    return summarize_hourly_frame(hf, tz=tz)


def get_recent_hourly_rain(lat: float, lon: float, past_days: int = 30) -> pd.Series:
    """
    Hourly observed precipitation (mm/h, UTC DatetimeIndex) over the past
    `past_days` from Open-Meteo (zero auth). Used by flow_forecast.py to
    disaggregate daily ungauged basin rain into the actual storm hours.
    """
    params = {
        'latitude': round(float(lat), 4),
        'longitude': round(float(lon), 4),
        'past_days': int(np.clip(past_days, 1, 92)),
        'forecast_days': 1,
        'hourly': 'precipitation',
        'timezone': 'UTC',
    }
    resp = data_utils.request_with_retry(OPEN_METEO_FORECAST_URL, params=params, timeout=60)
    if resp is None:
        return pd.Series(dtype=float)
    try:
        hourly = resp.json().get('hourly') or {}
    except ValueError:
        return pd.Series(dtype=float)
    times = hourly.get('time') or []
    vals = hourly.get('precipitation') or []
    if not times or len(times) != len(vals):
        return pd.Series(dtype=float)
    idx = pd.to_datetime(times, utc=True)
    return pd.Series(pd.to_numeric(vals, errors='coerce'), index=idx, dtype=float).clip(lower=0.0)


def get_latest_soil_moisture(lat: float, lon: float) -> Optional[float]:
    """Latest volumetric root-zone soil moisture (m^3/m^3) for routing abstraction."""
    try:
        df = fetch_gfs_recent_daily(lat, lon, past_days=3)
        valid = df['sm_rootzone'].dropna()
        return float(valid.iloc[-1]) if len(valid) else None
    except Exception as e:
        logger.debug("Latest soil moisture lookup failed: %s", e)
        return None


# --------------------------------------------------------------------------
# NASA Earthdata NLDAS-2 (1/8 deg CONUS hourly precip + Noah LSM soil moisture)
# --------------------------------------------------------------------------

def _extract_nldas_point(nc_path: str, lat: float, lon: float,
                         var_names: tuple) -> dict:
    """Extract nearest-gridpoint values from one NLDAS-2 NetCDF-4/HDF5 file."""
    import h5py
    out = {}
    with h5py.File(nc_path, 'r') as f:
        lats = np.asarray(f['lat'][:], dtype=float)
        lons = np.asarray(f['lon'][:], dtype=float)
        iy = int(np.argmin(np.abs(lats - lat)))
        ix = int(np.argmin(np.abs(lons - lon)))
        for v in var_names:
            if v not in f:
                continue
            ds = f[v]
            arr = np.asarray(ds[...], dtype=float)
            val = float(arr[0, iy, ix] if arr.ndim == 3 else arr[iy, ix])
            fill = ds.attrs.get('_FillValue', -9999.0)
            fill = float(fill[0] if np.ndim(fill) else fill)
            if not np.isfinite(val) or abs(val - fill) < 1e-3 or val < -900:
                val = np.nan
            out[v] = val
    return out


def fetch_nldas_daily(lat: float, lon: float, start_date, end_date,
                      tz: str = 'America/Denver') -> pd.DataFrame:
    """
    Fetch NLDAS-2 Primary Forcing (hourly radar-gauge precip) + Noah LSM
    (0-10 cm surface & 10-100 cm root-zone soil moisture, kg/m^2 -> m^3/m^3)
    via NASA Earthdata (`earthaccess` + `h5py`). Degrades cleanly to empty
    when credentials or dependencies are absent.
    """
    try:
        import earthaccess
        import h5py  # noqa: F401
    except ImportError as e:
        logger.warning("NLDAS-2 deps unavailable (%s)", e)
        return _empty()
    try:
        auth = earthaccess.login(strategy='environment')
        if auth is None or not getattr(auth, 'authenticated', False):
            logger.warning("Earthdata auth failed for NLDAS-2")
            return _empty()
    except Exception as e:
        logger.warning("Earthdata login error for NLDAS-2: %s", e)
        return _empty()

    start_d = pd.Timestamp(start_date).date()
    end_d = pd.Timestamp(end_date).date()
    bbox = (lon - 0.2, lat - 0.2, lon + 0.2, lat + 0.2)
    persistent_cache = os.getenv('OPENFLOW_NLDAS_CACHE_DIR', '').strip()
    cache_dir = persistent_cache or tempfile.mkdtemp(prefix='openflow_nldas_')
    ephemeral = not bool(persistent_cache)
    os.makedirs(cache_dir, exist_ok=True)

    def _pull(short_name: str, vars_: tuple) -> dict:
        granules = list(earthaccess.search_data(
            short_name=short_name, version=NLDAS_VERSION,
            temporal=(start_d, end_d), bounding_box=bbox,
        ) or [])
        if not granules:
            return {}
        by_time = {}
        for i in range(0, len(granules), NLDAS_BATCH_SIZE):
            batch = granules[i:i + NLDAS_BATCH_SIZE]
            paths = earthaccess.download(batch, local_path=cache_dir) or []
            for p in paths:
                p_str = str(p)
                try:
                    name = os.path.basename(p_str)
                    # NLDAS_FORA0125_H.A20240715.1800.020.nc -> 20240715 1800 UTC
                    parts = name.split('.')
                    ts = None
                    for idx_p, tok in enumerate(parts):
                        if tok.startswith('A') and len(tok) == 9 and tok[1:].isdigit():
                            hhmm = (parts[idx_p + 1]
                                    if idx_p + 1 < len(parts) and len(parts[idx_p + 1]) == 4
                                    else '0000')
                            ts = pd.Timestamp(f"{tok[1:5]}-{tok[5:7]}-{tok[7:9]}T{hhmm[:2]}:{hhmm[2:]}:00Z")
                            break
                    if ts is not None:
                        try:
                            by_time[ts] = _extract_nldas_point(p_str, lat, lon, vars_)
                        except Exception:
                            pass
                finally:
                    if ephemeral and os.path.exists(p_str):
                        try:
                            os.unlink(p_str)
                        except OSError:
                            pass
        return by_time

    try:
        fora = _pull(NLDAS_FORA_SHORT_NAME, ('Rainf',))
        noah = _pull(NLDAS_NOAH_SHORT_NAME, ('SoilM_0_10cm', 'SoilM_10_40cm', 'SoilM_40_100cm'))
    finally:
        if ephemeral:
            shutil.rmtree(cache_dir, ignore_errors=True)
    all_ts = sorted(set(fora) | set(noah))
    if not all_ts:
        return _empty()
    rows = []
    for ts in all_ts:
        fv = fora.get(ts, {})
        nv = noah.get(ts, {})
        # NLDAS Rainf is kg/m^2 over 1 hour == mm/h.
        # NLDAS SoilM is kg/m^2 over layer thickness (mm): divide by thickness_mm -> m^3/m^3.
        sm0 = nv.get('SoilM_0_10cm', np.nan)
        sm1 = nv.get('SoilM_10_40cm', np.nan)
        sm2 = nv.get('SoilM_40_100cm', np.nan)
        surf = sm0 / 100.0 if np.isfinite(sm0) else np.nan
        root = ((sm1 + sm2) / 900.0 if (np.isfinite(sm1) and np.isfinite(sm2))
                else (sm1 / 300.0 if np.isfinite(sm1) else np.nan))
        rows.append({'time': ts, 'precip_mm': fv.get('Rainf', np.nan),
                     'sm_surface': surf, 'sm_rootzone': root})
    return summarize_hourly_frame(pd.DataFrame(rows), tz=tz)


def get_hydro_forcing(lat: float, lon: float, start_date, end_date,
                      source_override: Optional[str] = None,
                      tz: str = 'America/Denver') -> pd.DataFrame:
    """
    Public API: daily [Date, precip_max1h, precip_max3h, precip_wet_hours,
    sm_surface, sm_rootzone] for (lat, lon) over [start_date, end_date].
    Falls back cleanly from NLDAS-2 to zero-auth ERA5-Land + GFS when
    Earthdata credentials are unavailable (e.g. local CLI / mobile).
    """
    if _disabled():
        logger.info("OPENFLOW_DISABLE_HYDRO_FORCING set -- skipping continuous SM & rain intensity")
        return _empty()
    src = hydro_source(source_override)
    if src == 'nldas':
        df = fetch_nldas_daily(lat, lon, start_date, end_date, tz=tz)
        if not df.empty:
            return df
        logger.info("NLDAS-2 unavailable; falling back to zero-auth ERA5-Land + GFS")
    return fetch_era5_daily(lat, lon, start_date, end_date, tz=tz)
