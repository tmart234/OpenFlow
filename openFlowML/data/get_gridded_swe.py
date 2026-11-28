"""
Elevation-band gridded Snow Water Equivalent (SWE) for spring snowmelt (Phase 3).

Single-HUC SNOTEL point averages miss retreating snowlines (e.g. Royal Gorge
has 0 SNOTEL sites in its HUC8; Clear Creek and State Line have only 1). This
module partitions each station's contributing-basin sample points
(`basin_zones.json`) into three elevation bands — low, mid, and high tertiles
via Open-Meteo's zero-auth Elevation API — and tracks representative daily SWE
(mm water equivalent) and 1-day snowmelt rate (`delta_swe`) per band:

  swe_low, swe_mid, swe_high    -- daily SWE (mm) in lower / middle / alpine bands
  delta_swe_mid, delta_swe_high -- 1-day SWE change (mm/day; negative = active melt)

Data source (100% zero auth, continuous 1950-present):
  - History: Open-Meteo ERA5-Land archive (`snow_depth_water_equivalent` in m -> mm)
  - Trailing lag bridge (last ~7 days): Open-Meteo GFS `snow_depth` (m * 250 mm/m
    snow-density approximation, bias-aligned to ERA5-Land over the overlap).

Set OPENFLOW_DISABLE_GRIDDED_SWE=1 to disable for cross-validation ablations.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data import basin_zones
from data.utils import data_utils

logger = logging.getLogger(__name__)

ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"
ERA5_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
ERA5_ARCHIVE_MODEL = "era5_seamless"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
# Snow depth (m) -> approximate SWE (mm) using a 250 kg/m^3 snow-density
# factor, matching the GFS tail bridge. Open-Meteo's archive endpoint leaves
# `snow_depth_water_equivalent` undefined (all nulls), whereas `snow_depth`
# is populated continuously back to 1950 across all Western US elevation bands.
SNOW_DEPTH_TO_SWE_MM = 250.0

BANDS = ('low', 'mid', 'high')
SWE_COLUMNS = ['swe_low', 'swe_mid', 'swe_high', 'delta_swe_mid', 'delta_swe_high']
ERA5_CHUNK_DAYS = 5 * 366

# Process-level cache of representative (lat, lon) per elevation band by site_id.
_BAND_POINT_CACHE: Dict[str, Dict[str, Tuple[float, float]]] = {}


def _disabled() -> bool:
    return os.getenv('OPENFLOW_DISABLE_GRIDDED_SWE', '').strip() in ('1', 'true', 'True')


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=['Date', *SWE_COLUMNS])


def select_elevation_band_points(points: List[dict],
                                 elevations: List[float]) -> Dict[str, Tuple[float, float]]:
    """
    Pick one representative (lat, lon) point per elevation tertile (low, mid,
    high) by taking the median-elevation point inside each tertile. Pure.
    """
    valid = [(float(e), float(p['lat']), float(p['lon']))
             for p, e in zip(points, elevations) if e is not None and np.isfinite(e)]
    if not valid:
        return {}
    valid.sort(key=lambda x: x[0])
    n = len(valid)
    if n == 1:
        pt = (valid[0][1], valid[0][2])
        return {b: pt for b in BANDS}
    if n == 2:
        return {'low': (valid[0][1], valid[0][2]),
                'mid': (valid[0][1], valid[0][2]),
                'high': (valid[1][1], valid[1][2])}
    t1, t2 = n // 3, (2 * n) // 3
    groups = {'low': valid[:t1], 'mid': valid[t1:t2], 'high': valid[t2:]}
    out = {}
    for band, grp in groups.items():
        mid_item = grp[len(grp) // 2]
        out[band] = (mid_item[1], mid_item[2])
    return out


def _resolve_band_points(site_id: str, fallback_lat: float, fallback_lon: float,
                         zones: Optional[dict] = None) -> Dict[str, Tuple[float, float]]:
    if site_id in _BAND_POINT_CACHE:
        return _BAND_POINT_CACHE[site_id]
    cfg = basin_zones.station_zones(site_id, zones)
    pts = (cfg or {}).get('points') or []
    if not pts:
        res = {b: (float(fallback_lat), float(fallback_lon)) for b in BANDS}
        _BAND_POINT_CACHE[site_id] = res
        return res
    lats = ','.join(f"{p['lat']:.4f}" for p in pts)
    lons = ','.join(f"{p['lon']:.4f}" for p in pts)
    resp = data_utils.request_with_retry(ELEVATION_URL, params={'latitude': lats, 'longitude': lons}, timeout=30)
    if resp is not None:
        try:
            elevs = resp.json().get('elevation') or []
            if len(elevs) == len(pts):
                selected = select_elevation_band_points(pts, elevs)
                if selected:
                    _BAND_POINT_CACHE[site_id] = selected
                    return selected
        except ValueError:
            pass
    # Fallback without elevation API: pick one sample point from local/mid/far.
    by_zone = {z: [p for p in pts if p.get('zone') == z] for z in basin_zones.ZONES}
    first = (float(fallback_lat), float(fallback_lon))
    res = {
        'low': (by_zone['local'][0]['lat'], by_zone['local'][0]['lon']) if by_zone['local'] else first,
        'mid': (by_zone['mid'][0]['lat'], by_zone['mid'][0]['lon']) if by_zone['mid'] else first,
        'high': (by_zone['far'][0]['lat'], by_zone['far'][0]['lon']) if by_zone['far'] else first,
    }
    _BAND_POINT_CACHE[site_id] = res
    return res


def _parse_hourly_swe_block(payload: Optional[dict], density_mm_per_m: float = SNOW_DEPTH_TO_SWE_MM) -> pd.Series:
    """Parse one Open-Meteo hourly response object into a daily SWE Series (mm)."""
    hourly = (payload or {}).get('hourly') or {}
    times = hourly.get('time') or []
    if not times:
        return pd.Series(dtype=float)
    swe_vals = hourly.get('snow_depth_water_equivalent')
    sd_vals = hourly.get('snow_depth')
    if swe_vals is not None and len(swe_vals) == len(times):
        s_swe = pd.Series(pd.to_numeric(swe_vals, errors='coerce'),
                          index=pd.to_datetime(times).normalize(), dtype=float)
        if s_swe.notna().any():
            return (s_swe.groupby(level=0).mean() * 1000.0).clip(lower=0.0).dropna()
    if sd_vals is not None and len(sd_vals) == len(times):
        s_sd = pd.Series(pd.to_numeric(sd_vals, errors='coerce'),
                         index=pd.to_datetime(times).normalize(), dtype=float)
        if s_sd.notna().any():
            return (s_sd.groupby(level=0).mean() * float(density_mm_per_m)).clip(lower=0.0).dropna()
    return pd.Series(dtype=float)


def _fetch_points_swe(coords: List[Tuple[float, float]], start: pd.Timestamp,
                      end: pd.Timestamp, tz: str = 'America/Denver') -> Dict[Tuple[float, float], pd.Series]:
    """
    Batch-fetch daily SWE (mm) for one or more (lat, lon) points from
    Open-Meteo's ERA5-Seamless archive (`snow_depth`) + GFS tail bridge.
    Passing all elevation-band coordinates in one comma-separated request cuts
    archive HTTP calls by up to 3x per station.
    """
    if not coords:
        return {}
    today = pd.Timestamp.now().normalize()
    archive_end = min(end, today - pd.Timedelta(days=2))
    lats_param = ','.join(f"{lat:.4f}" for lat, _ in coords) if len(coords) > 1 else round(float(coords[0][0]), 4)
    lons_param = ','.join(f"{lon:.4f}" for _, lon in coords) if len(coords) > 1 else round(float(coords[0][1]), 4)

    parts_by_coord: Dict[Tuple[float, float], List[pd.Series]] = {c: [] for c in coords}
    if start <= archive_end:
        for cs, ce in data_utils.date_chunks(start.to_pydatetime(), archive_end.to_pydatetime(), ERA5_CHUNK_DAYS):
            params = {
                'latitude': lats_param,
                'longitude': lons_param,
                'start_date': cs.strftime('%Y-%m-%d'),
                'end_date': ce.strftime('%Y-%m-%d'),
                'hourly': 'snow_depth',
                'models': ERA5_ARCHIVE_MODEL,
                'timezone': tz,
            }
            resp = data_utils.request_with_retry(ERA5_ARCHIVE_URL, params=params, timeout=180)
            if resp is None:
                continue
            try:
                raw = resp.json()
            except ValueError:
                continue
            items = raw if isinstance(raw, list) else [raw]
            if len(items) != len(coords):
                continue
            for coord, item in zip(coords, items):
                daily = _parse_hourly_swe_block(item)
                if not daily.empty:
                    parts_by_coord[coord].append(daily)

    out: Dict[Tuple[float, float], pd.Series] = {}
    for coord in coords:
        plist = parts_by_coord[coord]
        series = pd.concat(plist).sort_index() if plist else pd.Series(dtype=float)
        out[coord] = series[~series.index.duplicated(keep='last')].dropna()

    if end >= today - pd.Timedelta(days=10):
        any_empty = any(s.empty for s in out.values())
        past_d = int(np.clip((today - start).days + 2, 10, 92)) if any_empty else 10
        resp = data_utils.request_with_retry(
            OPEN_METEO_FORECAST_URL,
            params={'latitude': lats_param, 'longitude': lons_param,
                    'past_days': past_d, 'forecast_days': 1, 'hourly': 'snow_depth', 'timezone': tz},
            timeout=60,
        )
        if resp is not None:
            try:
                raw = resp.json()
                items = raw if isinstance(raw, list) else [raw]
                if len(items) == len(coords):
                    for coord, item in zip(coords, items):
                        tail = _parse_hourly_swe_block(item)
                        series = out[coord]
                        if not series.empty and not tail.empty:
                            both = pd.concat([series, tail], axis=1, keys=['e', 'g']).dropna()
                            both = both[both['g'] > 1.0]
                            if len(both) >= 2:
                                ratio = float(np.clip(np.median(both['e'] / both['g']), 0.3, 3.0))
                                tail = tail * ratio
                            tail = tail[tail.index > series.index.max()]
                        if not tail.empty:
                            out[coord] = tail.sort_index() if series.empty else pd.concat([series, tail]).sort_index()
            except ValueError:
                pass
    return out


def _fetch_point_swe(lat: float, lon: float, start: pd.Timestamp,
                     end: pd.Timestamp, tz: str = 'America/Denver') -> pd.Series:
    """Daily SWE (mm) for one (lat, lon) from ERA5-Seamless archive + GFS tail bridge."""
    key = (round(float(lat), 4), round(float(lon), 4))
    return _fetch_points_swe([key], start, end, tz=tz).get(key, pd.Series(dtype=float))


def assemble_band_swe(band_series: Dict[str, pd.Series]) -> pd.DataFrame:
    """Combine low/mid/high SWE Series (mm) into DataFrame[Date, *SWE_COLUMNS]. Pure."""
    valid = {f'swe_{b}': s for b, s in band_series.items() if b in BANDS and s is not None and len(s)}
    if not valid:
        return _empty()
    df = pd.DataFrame(valid).sort_index()
    for b in BANDS:
        col = f'swe_{b}'
        if col not in df.columns:
            df[col] = 0.0
    df = df.ffill(limit=7).fillna(0.0)
    df['delta_swe_mid'] = df['swe_mid'].diff().fillna(0.0)
    df['delta_swe_high'] = df['swe_high'].diff().fillna(0.0)
    df.index = pd.DatetimeIndex(df.index, name='Date')
    return df.reset_index()[['Date', *SWE_COLUMNS]]


def get_gridded_swe(site_id: str, lat: float, lon: float, start_date, end_date,
                    zones: Optional[dict] = None, tz: str = 'America/Denver') -> pd.DataFrame:
    """Public API: daily elevation-band SWE + daily melt rate for `site_id`."""
    if _disabled():
        logger.info("OPENFLOW_DISABLE_GRIDDED_SWE set -- skipping elevation-band SWE for %s", site_id)
        return _empty()
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    pts = _resolve_band_points(site_id, lat, lon, zones=zones)
    # Deduplicate identical coordinates so we never fetch the same point twice.
    band_keys = {band: (round(coord[0], 4), round(coord[1], 4)) for band, coord in pts.items()}
    unique_coords = list(dict.fromkeys(band_keys.values()))
    by_coord = _fetch_points_swe(unique_coords, start, end, tz=tz)
    band_series = {band: by_coord.get(key, pd.Series(dtype=float)) for band, key in band_keys.items()}
    df = assemble_band_swe(band_series)
    if df.empty:
        return df
    return df[(df['Date'] >= start) & (df['Date'] <= end)].reset_index(drop=True)
