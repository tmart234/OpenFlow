"""
Basin precipitation by travel-time zone, split into rain and snow.

Zones (local / mid / far river-distance bands over the NLDI contributing
area, plus the 'ungauged' area) come from basin_zones.json. This module
turns them into precipitation series from three NOAA/academic sources:

  gridMET   daily 4-km grids, 1979 -> ~2 days ago (THREDDS NetCDF Subset
            Service; one bounding-box NetCDF-3 request per variable, read
            with scipy). Training history, and the past for inference.
  Stage IV  NWS/RFC daily multi-sensor precipitation analysis (radar +
            gauges, the QPE the River Forecast Centers use), one CONUS
            NetCDF-4 file per 24 h ending 12Z, out within hours. Fills the
            1-2 day gap until gridMET catches up -- which is exactly when a
            storm that is about to flood a river usually happened. Needs h5py
            (skipped with a warning when it isn't installed).
  NWS QPF   NDFD gridpoint forecasts (api.weather.gov): 6-hourly QPF out to
            ~7.5 days + hourly temperature at each zone's sample points
            (gridpoint ids pre-resolved in basin_zones.json). Forecasts.

Rain vs snow uses one rule everywhere (basin_zones.liquid_fraction on mean
air temperature): daily mean temperature for daily series, hourly for the
hourly routing input. Stage IV has no temperature, so its days reuse each
zone's mean temperature from the latest gridMET day (persistence over a
1-2 day gap). Snow is reported as water equivalent (mm).

Outputs:
  get_zone_precip_daily(site, start, end) -> training columns FEATURE_COLUMNS
  observed_daily(cfg, start, today)       -> gridMET + Stage IV, one row per local day
  inference_daily(site, ...)              -> encoder + decoder daily model inputs
  routing_rain_flux(cfg, grid, ...)       -> hourly rain on the ungauged area as a
                                             volumetric flux (cfs), routing.py's
                                             local-inflow input
"""

import io
import logging
import os
import re
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from data import basin_zones
from data.utils import data_utils

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = ['rain_local', 'rain_mid', 'rain_far', 'snow_local', 'snow_mid', 'snow_far']
OBSERVED_COLUMN = 'basin_precip_observed'
# Decoder days with a basin precipitation forecast (NWS QPF reaches ~7.5 days).
QPF_DAYS = 7
AREAS = basin_zones.ZONES + ('ungauged',)

GRIDMET_URL = "https://thredds.northwestknowledge.net/thredds/ncss/agg_met_{var}_1979_CurrentYear_CONUS.nc"
GRIDMET_VARS = {'pr': 'precipitation_amount',
                'tmmn': 'daily_minimum_temperature',
                'tmmx': 'daily_maximum_temperature'}
GRIDMET_CHUNK_DAYS = 3 * 366
STAGE4_URL = ("https://water.noaa.gov/resources/downloads/precip/stageIV/"
              "{d:%Y}/{d:%m}/{d:%d}/nws_precip_1day_{d:%Y%m%d}_conus.nc")
# Stage IV / HRAP grid: polar stereographic on a sphere, true at 60N, 105W.
HRAP_RADIUS_M = 6371200.0
HRAP_K = HRAP_RADIUS_M * (1 + np.sin(np.radians(60.0)))
HRAP_LON0 = -105.0
NWS_GRIDPOINT_URL = "https://api.weather.gov/gridpoints/{gridpoint}"
NWS_HEADERS = {'User-Agent': 'OpenFlow (github.com/tmart234/OpenFlow)',
               'Accept': 'application/geo+json'}
MAX_STAGE4_GAP_DAYS = 7
# Initial abstraction for the routing rain term: the first few mm of basin-
# mean rain per day wet the soil and canopy and produce no runoff, so light
# showers don't raise the river while storms still do.
ABSTRACTION_MM_PER_DAY = 5.0


def _disabled() -> bool:
    return os.getenv('OPENFLOW_DISABLE_BASIN_PRECIP', '').strip() in ('1', 'true', 'True')


# --------------------------------------------------------------------------
# gridMET (history)
# --------------------------------------------------------------------------

def parse_gridmet_netcdf(content: bytes, var: str):
    """
    Parse one NCSS NetCDF-3 subset. Returns (dates, lat, lon, values[T, Y, X])
    with scale/offset applied and fill values as NaN.
    """
    from scipy.io import netcdf_file
    f = netcdf_file(io.BytesIO(content), 'r', mmap=False, maskandscale=True)
    try:
        v = f.variables[var]
        data = np.ma.filled(np.ma.asarray(v[:], dtype=float), np.nan)
        days = np.asarray(f.variables['day'][:], dtype=float)
        lat = np.asarray(f.variables['lat'][:], dtype=float)
        lon = np.asarray(f.variables['lon'][:], dtype=float)
    finally:
        f.close()
    # NCSS writes a variable attribute named "dimensions" ("lon lat time")
    # that shadows scipy's dimension tuple, so infer the axis order from the
    # axis lengths; NCSS's native order (day, lat, lon) breaks ties.
    sizes = {'day': len(days), 'lat': len(lat), 'lon': len(lon)}
    if data.shape != (sizes['day'], sizes['lat'], sizes['lon']):
        order, used = [], set()
        for name in ('day', 'lat', 'lon'):
            ax = next(i for i, n in enumerate(data.shape) if n == sizes[name] and i not in used)
            order.append(ax)
            used.add(ax)
        data = np.transpose(data, order)
    dates = pd.Timestamp('1900-01-01') + pd.to_timedelta(days, unit='D')
    return pd.DatetimeIndex(dates).normalize(), lat, lon, data


def _gridmet_subset(var_key, south, north, west, east, start, end):
    var = GRIDMET_VARS[var_key]
    parts = []
    cur = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    while cur <= end:
        chunk_end = min(cur + pd.Timedelta(days=GRIDMET_CHUNK_DAYS - 1), end)
        params = {'var': var, 'north': north, 'south': south, 'west': west, 'east': east,
                  'time_start': cur.strftime('%Y-%m-%dT00:00:00Z'),
                  'time_end': chunk_end.strftime('%Y-%m-%dT00:00:00Z'),
                  'accept': 'netcdf'}
        response = data_utils.request_with_retry(GRIDMET_URL.format(var=var_key), params=params,
                                                 timeout=300)
        if response is None or not response.content.startswith(b'CDF'):
            raise RuntimeError(f"gridMET {var_key} subset failed for {cur.date()}..{chunk_end.date()}")
        parts.append(parse_gridmet_netcdf(response.content, var))
        cur = chunk_end + pd.Timedelta(days=1)
    dates = parts[0][0].append([p[0] for p in parts[1:]]) if len(parts) > 1 else parts[0][0]
    values = np.concatenate([p[3] for p in parts], axis=0)
    return dates, parts[0][1], parts[0][2], values


def zone_means(cfg: dict, lon2d: np.ndarray, lat2d: np.ndarray, field: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Area means of field[T, Y, X] (or [Y, X]) over each zone and the ungauged
    area, on a grid whose cell centres are lon2d/lat2d. A zone with area but
    no cell centre inside it (tiny zone) uses the cell nearest its first
    sample point; an empty zone is 0.
    """
    field = np.asarray(field, dtype=float)
    squeeze = field.ndim == 2
    if squeeze:
        field = field[None]
    masks = basin_zones.grid_masks(cfg, lon2d, lat2d)
    out = {}
    for name, mask in masks.items():
        meta = cfg.get('ungauged') if name == 'ungauged' else cfg['zones'].get(name)
        area = (meta or {}).get('area_km2', 0)
        if mask.any():
            vals = np.nanmean(field[:, mask], axis=1)
        elif area and area > 0:
            pts = [p for p in cfg.get('points', [])
                   if (p.get('ungauged') if name == 'ungauged' else p.get('zone') == name)]
            if pts:
                d2 = (lat2d - pts[0]['lat']) ** 2 + (lon2d - pts[0]['lon']) ** 2
                iy, ix = np.unravel_index(int(np.argmin(d2)), d2.shape)
                vals = field[:, iy, ix]
            else:
                vals = np.full(field.shape[0], np.nan)
        else:
            vals = np.zeros(field.shape[0])
        out[name] = vals[0] if squeeze else vals
    return out


def gridmet_zone_daily(cfg: dict, start, end) -> pd.DataFrame:
    """
    Daily zone rain / snow (mm water equivalent) from gridMET, plus
    rain_/precip_/tmean_ columns for the ungauged area and tmean_<zone>
    (deg C). Indexed by Date; days gridMET does not have yet are absent.
    """
    south, north, west, east = basin_zones.bbox(cfg)
    dates, lat, lon, pr = _gridmet_subset('pr', south, north, west, east, start, end)
    d2, _, _, tmn = _gridmet_subset('tmmn', south, north, west, east, start, end)
    d3, _, _, tmx = _gridmet_subset('tmmx', south, north, west, east, start, end)
    n = min(len(dates), len(d2), len(d3))
    pr, tmn, tmx, dates = pr[:n], tmn[:n], tmx[:n], dates[:n]
    tmean = (tmn + tmx) / 2.0 - 273.15
    rain = pr * basin_zones.liquid_fraction(tmean)
    lon2d, lat2d = np.meshgrid(lon, lat)
    rain_m = zone_means(cfg, lon2d, lat2d, rain)
    pr_m = zone_means(cfg, lon2d, lat2d, pr)
    t_m = zone_means(cfg, lon2d, lat2d, tmean)
    out = pd.DataFrame(index=pd.DatetimeIndex(dates, name='Date'))
    for z in basin_zones.ZONES:
        out[f'rain_{z}'] = rain_m[z]
        out[f'snow_{z}'] = np.maximum(pr_m[z] - rain_m[z], 0.0)
    for a in AREAS:
        out[f'tmean_{a}'] = t_m[a]
    out['rain_ungauged'] = rain_m['ungauged']
    out['precip_ungauged'] = pr_m['ungauged']
    return out


def get_zone_precip_daily(site_id: str, start_date, end_date,
                          zones: Optional[dict] = None) -> pd.DataFrame:
    """
    Training feature frame DataFrame[Date, *FEATURE_COLUMNS] for a station.
    Empty when the station has no zones, the source is disabled, or gridMET
    failed (combine_data then sets basin_precip_observed = 0).
    """
    empty = pd.DataFrame(columns=['Date'] + FEATURE_COLUMNS)
    if _disabled():
        logger.info("OPENFLOW_DISABLE_BASIN_PRECIP set -- skipping basin precip for %s", site_id)
        return empty
    cfg = basin_zones.station_zones(site_id, zones)
    if cfg is None:
        logger.info("No basin zones for %s; run data/build_basin_zones.py", site_id)
        return empty
    try:
        daily = gridmet_zone_daily(cfg, start_date, end_date)
    except Exception as e:
        logger.warning("gridMET basin precip failed for %s: %s", site_id, e)
        return empty
    return daily[FEATURE_COLUMNS].reset_index()


# --------------------------------------------------------------------------
# Stage IV (recent days)
# --------------------------------------------------------------------------

def hrap_to_lonlat(x, y):
    """Inverse of the HRAP polar stereographic projection (metres -> degrees)."""
    rho = np.hypot(x, y)
    lat = 90.0 - 2.0 * np.degrees(np.arctan(rho / HRAP_K))
    lon = HRAP_LON0 + np.degrees(np.arctan2(x, -y))
    return lon, lat


def lonlat_to_hrap(lon, lat):
    rho = HRAP_K * np.tan(np.radians(45.0 - np.asarray(lat, dtype=float) / 2.0))
    a = np.radians(np.asarray(lon, dtype=float) - HRAP_LON0)
    return rho * np.sin(a), -rho * np.cos(a)


def stage4_zone_precip(cfg: dict, content: bytes) -> Dict[str, float]:
    """Zone + ungauged mean precipitation (mm) from one Stage IV 1-day file."""
    import h5py
    south, north, west, east = basin_zones.bbox(cfg)
    with h5py.File(io.BytesIO(content), 'r') as f:
        x = f['x'][:]
        y = f['y'][:]
        cx, cy = lonlat_to_hrap(np.array([west, east, west, east, (west + east) / 2]),
                                np.array([south, south, north, north, north]))
        pad = 10000.0
        ix = np.where((x >= cx.min() - pad) & (x <= cx.max() + pad))[0]
        iy = np.where((y >= cy.min() - pad) & (y <= cy.max() + pad))[0]
        if not len(ix) or not len(iy):
            raise ValueError("basin outside the Stage IV grid")
        obs = f['observation'][iy.min():iy.max() + 1, ix.min():ix.max() + 1].astype(float)
    obs[(obs < 0) | (obs > 1e6)] = np.nan
    lon2d, lat2d = hrap_to_lonlat(*np.meshgrid(x[ix.min():ix.max() + 1], y[iy.min():iy.max() + 1]))
    means = zone_means(cfg, lon2d, lat2d, obs * 25.4)   # inches -> mm
    return {k: float(v) for k, v in means.items()}


def fetch_stage4(cfg: dict, end_day) -> Optional[Dict[str, float]]:
    """Zone precip (mm) for the 24 h ending 12Z on end_day, or None if unavailable."""
    url = STAGE4_URL.format(d=pd.Timestamp(end_day))
    response = data_utils.request_with_retry(url, timeout=120, retries=2)
    if response is None or not response.content.startswith(b'\x89HDF'):
        return None
    try:
        return stage4_zone_precip(cfg, response.content)
    except ImportError:
        logger.warning("h5py not installed; Stage IV gap fill skipped")
        return None
    except Exception as e:
        logger.warning("Stage IV %s unusable: %s", url, e)
        return None


def observed_daily(cfg: dict, start, today, gridmet: Optional[pd.DataFrame] = None,
                   stage4=fetch_stage4) -> pd.DataFrame:
    """
    Observed basin precipitation, one row per local day from `start` through
    the latest available day (<= today - 1): gridMET where it has the day,
    then Stage IV for the gap (local day D <- the 24 h ending 12Z on D+1;
    rain/snow split on the zone's latest gridMET mean temperature).

    Columns: FEATURE_COLUMNS, rain_ungauged, precip_ungauged, source
    ('gridmet' | 'stage4') and window_start (UTC start of the accumulation
    window, for spreading the day over hours).
    """
    today = pd.Timestamp(today).normalize()
    if gridmet is None:
        try:
            gridmet = gridmet_zone_daily(cfg, start, today)
        except Exception as e:
            logger.warning("gridMET unavailable (%s); using Stage IV only", e)
            gridmet = pd.DataFrame()
    rows = gridmet.copy()
    if len(rows):
        rows['source'] = 'gridmet'
        rows['window_start'] = pd.DatetimeIndex(rows.index).tz_localize('UTC')
        last = rows.index.max()
        temps = rows.iloc[-1]
    else:
        last = pd.Timestamp(start).normalize() - pd.Timedelta(days=1)
        temps = None
    gap_start = max(last + pd.Timedelta(days=1), today - pd.Timedelta(days=MAX_STAGE4_GAP_DAYS))
    extra = []
    for day in pd.date_range(gap_start, today - pd.Timedelta(days=1)):
        mm = stage4(cfg, day + pd.Timedelta(days=1)) if stage4 else None
        if mm is None:
            continue
        rec = {'source': 'stage4', 'window_start': (day + pd.Timedelta(hours=12)).tz_localize('UTC')}
        for a in AREAS:
            t = temps.get(f'tmean_{a}') if temps is not None else None
            frac = float(basin_zones.liquid_fraction(t)) if t is not None and np.isfinite(t) else 1.0
            rain = mm[a] * frac
            if a == 'ungauged':
                rec['rain_ungauged'] = rain
                rec['precip_ungauged'] = mm[a]
            else:
                rec[f'rain_{a}'] = rain
                rec[f'snow_{a}'] = max(mm[a] - rain, 0.0)
        extra.append(pd.Series(rec, name=day))
    if extra:
        rows = pd.concat([rows, pd.DataFrame(extra)]) if len(rows) else pd.DataFrame(extra)
    rows.index = pd.DatetimeIndex(rows.index, name='Date')
    return rows.sort_index()


# --------------------------------------------------------------------------
# NWS gridpoint forecasts
# --------------------------------------------------------------------------

_DUR = re.compile(r'P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?)?$')


def iso_duration_hours(s: str) -> int:
    m = _DUR.match(s)
    if not m:
        raise ValueError(f"bad ISO-8601 duration {s!r}")
    days, hours, minutes = (int(x) if x else 0 for x in m.groups())
    return max(1, days * 24 + hours + (1 if minutes else 0))


def _intervals(values):
    for v in values or []:
        if v.get('value') is None:
            continue
        start_s, dur_s = v['validTime'].split('/')
        yield pd.Timestamp(start_s).tz_convert('UTC'), iso_duration_hours(dur_s), float(v['value'])


def parse_nws_gridpoint(payload: dict) -> pd.DataFrame:
    """
    Hourly UTC frame [precip_mm, temp_c] from an NWS gridpoint payload. QPF
    intervals (usually 6 h) are spread evenly over their hours; temperature
    intervals hold their value. Hours outside the QPF coverage are absent.
    """
    props = payload.get('properties', payload)
    qpf = list(_intervals(props.get('quantitativePrecipitation', {}).get('values')))
    tmp = list(_intervals(props.get('temperature', {}).get('values')))
    if not qpf:
        return pd.DataFrame(columns=['precip_mm', 'temp_c'])
    t0 = min(t for t, _, _ in qpf)
    t1 = max(t + pd.Timedelta(hours=h) for t, h, _ in qpf)
    idx = pd.date_range(t0, t1 - pd.Timedelta(hours=1), freq='1h')
    precip = pd.Series(np.nan, index=idx)
    for t, h, v in qpf:
        hours = pd.date_range(t, periods=h, freq='1h')
        precip.loc[precip.index.intersection(hours)] = v / h
    temp = pd.Series(np.nan, index=idx)
    for t, h, v in tmp:
        hours = pd.date_range(t, periods=h, freq='1h')
        temp.loc[temp.index.intersection(hours)] = v
    temp = temp.ffill().bfill()
    return pd.DataFrame({'precip_mm': precip, 'temp_c': temp})


def nws_points_hourly(cfg: dict) -> Dict[int, pd.DataFrame]:
    """Hourly forecast per sample point index (points that fail are skipped)."""
    out = {}
    for i, p in enumerate(cfg.get('points', [])):
        if not p.get('nws'):
            continue
        response = data_utils.request_with_retry(
            NWS_GRIDPOINT_URL.format(gridpoint=p['nws']), headers=NWS_HEADERS, timeout=60)
        if response is None:
            logger.warning("NWS gridpoint %s unavailable", p['nws'])
            continue
        try:
            df = parse_nws_gridpoint(response.json())
        except Exception as e:
            logger.warning("NWS gridpoint %s unparseable: %s", p['nws'], e)
            continue
        if not df.empty:
            out[i] = df
    logger.info("NWS forecasts at %d/%d sample points", len(out), len(cfg.get('points', [])))
    return out


def _weighted(cfg, per_point: Dict[int, pd.Series], key: str) -> pd.Series:
    """Weighted mean over the sample points of one zone / the ungauged area."""
    weights = basin_zones.point_weights(cfg)[key]
    cols = {i: s for i, s in per_point.items() if weights[i] > 0}
    if not cols:
        return pd.Series(dtype=float)
    frame = pd.DataFrame(cols)
    w = pd.Series({i: weights[i] for i in cols})
    valid = frame.notna()
    num = (frame.fillna(0) * w).sum(axis=1)
    den = (valid * w).sum(axis=1)
    return (num / den.where(den > 0)).astype(float)


def _zero_empty_zones(cfg: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Zones with no area get 0 (no rain can arrive from them), not NaN."""
    for z in basin_zones.ZONES:
        if not (cfg['zones'].get(z) or {}).get('area_km2'):
            for kind in ('rain', 'snow'):
                df[f'{kind}_{z}'] = 0.0
    return df


def points_hourly_zones(cfg: dict, per_point: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Hourly zone rain/snow and ungauged rain (mm/h), hourly-temperature partition."""
    rain = {i: df['precip_mm'] * basin_zones.liquid_fraction(df['temp_c']) for i, df in per_point.items()}
    total = {i: df['precip_mm'] for i, df in per_point.items()}
    out = {}
    for z in basin_zones.ZONES:
        r, t = _weighted(cfg, rain, z), _weighted(cfg, total, z)
        out[f'rain_{z}'] = r
        out[f'snow_{z}'] = (t - r).clip(lower=0)
    out['rain_ungauged'] = _weighted(cfg, rain, 'ungauged')
    return _zero_empty_zones(cfg, pd.DataFrame(out))


def points_daily_zones(cfg: dict, per_point: Dict[int, pd.DataFrame], tz: str) -> pd.DataFrame:
    """
    Daily (local calendar day) zone rain/snow (mm) from per-point hourly
    forecasts, partitioned on DAILY mean temperature like gridMET history.
    """
    rain, total = {}, {}
    for i, df in per_point.items():
        local = df.tz_convert(tz)
        g = local.groupby(local.index.date)
        p = g['precip_mm'].sum(min_count=1)
        t = g['temp_c'].mean()
        rain[i] = p * basin_zones.liquid_fraction(t)
        total[i] = p
    out = {}
    for z in basin_zones.ZONES:
        r, t = _weighted(cfg, rain, z), _weighted(cfg, total, z)
        out[f'rain_{z}'] = r
        out[f'snow_{z}'] = (t - r).clip(lower=0)
    df = _zero_empty_zones(cfg, pd.DataFrame(out))
    df.index = pd.DatetimeIndex(df.index, name='Date')
    return df


# --------------------------------------------------------------------------
# Inference + routing assembly
# --------------------------------------------------------------------------

def _local_today(today, tz: str) -> pd.Timestamp:
    now = pd.Timestamp.now(tz=tz) if today is None else pd.Timestamp(today)
    if now.tzinfo is not None:
        now = now.tz_convert(tz).tz_localize(None)
    return now.normalize()


def inference_daily(site_id: str, today=None, encoder_days: int = 60, decoder_days: int = 14,
                    tz: str = 'America/Denver', zones: Optional[dict] = None,
                    observed: Optional[pd.DataFrame] = None,
                    forecast_points: Optional[Dict[int, pd.DataFrame]] = None) -> pd.DataFrame:
    """
    Reference implementation of the basin-precip model inputs at inference.

    Rows: the encoder_days ending `today` (local date), then decoder_days.
    Columns: FEATURE_COLUMNS (mm/day), basin_precip_observed (1 = real value)
    and source ('gridmet' | 'stage4' | 'nws' | '').
      encoder days  gridMET, then Stage IV for the gap, then (today only) NWS
                    QPF hours, which start around 12Z today;
      decoder days  NWS QPF through QPF_DAYS; later days stay NaN with
                    observed = 0 (windowing marks them unknown).
    Pass observed / forecast_points to skip the network.
    """
    cfg = basin_zones.station_zones(site_id, zones)
    today = _local_today(today, tz)
    days = pd.date_range(today - pd.Timedelta(days=encoder_days - 1),
                         today + pd.Timedelta(days=decoder_days), name='Date')
    out = pd.DataFrame(np.nan, index=days, columns=FEATURE_COLUMNS)
    out['source'] = ''
    if cfg is None:
        out[OBSERVED_COLUMN] = 0
        return out
    if observed is None:
        observed = observed_daily(cfg, days[0], today)
    if len(observed):
        o = observed.reindex(days)
        have = o[FEATURE_COLUMNS].notna().all(axis=1)
        out.loc[have, FEATURE_COLUMNS] = o.loc[have, FEATURE_COLUMNS].astype(float)
        out.loc[have, 'source'] = o.loc[have, 'source']
    if forecast_points is None:
        forecast_points = nws_points_hourly(cfg)
    if forecast_points:
        daily = points_daily_zones(cfg, forecast_points, tz).reindex(days)
        fill = (out[FEATURE_COLUMNS].isna().all(axis=1) & daily.notna().all(axis=1)
                & (days >= today) & (days <= today + pd.Timedelta(days=QPF_DAYS)))
        out.loc[fill, FEATURE_COLUMNS] = daily.loc[fill, FEATURE_COLUMNS]
        out.loc[fill, 'source'] = 'nws'
    out[OBSERVED_COLUMN] = out[FEATURE_COLUMNS].notna().all(axis=1).astype(int)
    return out


def routing_rain_flux(cfg: dict, grid: pd.DatetimeIndex,
                      observed: Optional[pd.DataFrame],
                      forecast_points: Optional[Dict[int, pd.DataFrame]],
                      abstraction_mm_per_day: float = ABSTRACTION_MM_PER_DAY) -> Tuple[np.ndarray, dict]:
    """
    Hourly EFFECTIVE rain on the ungauged area as a volumetric flux (cfs) on
    `grid` (UTC): each observed day (gridMET / Stage IV) spread evenly over
    its 24-h accumulation window, then the NWS hourly forecast; each UTC
    day's rain is reduced by `abstraction_mm_per_day` (proportionally over
    its hours, floored at 0). Hours with no source are 0 and counted in
    info['missing_hours']. info['rate_mm_per_hour'] is the raw rain rate.
    """
    area = float((cfg.get('ungauged') or {}).get('area_km2', 0.0))
    rate = pd.Series(np.nan, index=grid)
    sources = []
    if observed is not None and len(observed) and 'rain_ungauged' in observed:
        for _, row in observed.dropna(subset=['rain_ungauged']).iterrows():
            hours = pd.date_range(row['window_start'], periods=24, freq='1h')
            hours = hours[(hours >= grid[0]) & (hours <= grid[-1])]
            rate.loc[hours] = float(row['rain_ungauged']) / 24.0
        sources += sorted(set(observed['source'].dropna()))
    if forecast_points:
        hz = points_hourly_zones(cfg, forecast_points)['rain_ungauged'].reindex(grid)
        last_obs = rate.last_valid_index()
        if last_obs is not None:
            hz[hz.index <= last_obs] = np.nan     # never let a forecast overwrite observations
        rate = rate.fillna(hz)
        sources.append('nws')
    missing = int(rate.isna().sum())
    rate = rate.fillna(0.0)
    day_total = rate.groupby(rate.index.floor('D')).transform('sum')
    keep = (1.0 - abstraction_mm_per_day / day_total.where(day_total > 0)).clip(lower=0).fillna(0.0)
    flux = basin_zones.mm_per_hour_to_cfs((rate * keep).to_numpy(), area)
    return flux, {'area_km2': area, 'sources': sources, 'missing_hours': missing,
                  'rate_mm_per_hour': rate}


def zone_totals(cfg: dict, observed: Optional[pd.DataFrame],
                forecast_points: Optional[Dict[int, pd.DataFrame]],
                today, tz: str, days: int = 7) -> dict:
    """Past-`days` (observed) and next-`days` (NWS) basin-mean rain/snow per zone, mm."""
    today = pd.Timestamp(today).normalize()
    out = {'past': {}, 'next': {}, 'past_through': None}
    if observed is not None and len(observed):
        past = observed[(observed.index > today - pd.Timedelta(days=days)) & (observed.index <= today)]
        out['past'] = {c: round(float(past[c].sum()), 1) for c in FEATURE_COLUMNS if c in past}
        out['past_through'] = str(past.index.max().date()) if len(past) else None
    if forecast_points:
        fut = points_daily_zones(cfg, forecast_points, tz)
        fut = fut[(fut.index > today) & (fut.index <= today + pd.Timedelta(days=days))]
        out['next'] = {c: round(float(fut[c].sum()), 1) for c in FEATURE_COLUMNS}
    return out
