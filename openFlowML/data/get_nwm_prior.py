"""
Physical / operational streamflow forecast priors: NWS RFC + NOAA NWM v3 (Phase 3).

On dam-regulated rivers (Dolores / McPhee, Conejos / Platoro, State Line /
Aspinall, Royal Gorge / Twin Lakes), human release schedules cannot be deduced
from weather alone. Meanwhile, NWS River Forecast Centers manually incorporate
scheduled releases into official RFC forecasts, and NOAA's National Water Model
(NWM v3) publishes 10-day medium-range blended ensemble forecasts for every
NHDPlus river reach in CONUS.

This module supplies one unified `prior_flow` (cfs) feature + `prior_observed`
indicator:
  - Live inference (`fetch_live_prior`):
      1. Official NWS RFC forecast at the gauge's `nws_lid` (when current)
      2. Extended / backed up by NOAA NWM v3 Operational Medium-Range Blend
         via NWPS Reach API (`https://api.water.noaa.gov/nwps/v1/reaches/{reachId}/streamflow?series=medium_range_blend`),
         keyed by the gauge's NWPS `reachId` (or NLDI `comid`) — zero auth!
      3. Bias-aligned onto the gauge's recent observed scale over overlap days
         (same splice principle as `ml_inference.splice_on_training_scale`).
  - Historical training (`fetch_historical_prior`):
      Reads daily-mean streamflow for the gauge's NWM `reachId` / `comid` from
      NOAA's public NWM v3.0 Retrospective Zarr store on AWS Open Data
      (`s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr`) when
      `xarray` + `zarr` + `s3fs` are installed; degrades cleanly to empty
      (`prior_observed = 0`) when those optional packages are absent.

Set OPENFLOW_DISABLE_PRIOR=1 to disable for ablations.
"""

import logging
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

from data import nwps
from data.utils import data_utils

logger = logging.getLogger(__name__)

NWPS_REACH_URL = "https://api.water.noaa.gov/nwps/v1/reaches/{reach_id}/streamflow"
NLDI_SITE_URL = "https://api.water.usgs.gov/nldi/linked-data/nwissite/USGS-{usgs_id}"
NWM_RETRO_ZARR = "s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr"
M3S_TO_CFS = 35.3146667

# Known NHDPlus COMID / NWM reachId cache for all 8 registered training gauges.
KNOWN_REACH_IDS: Dict[str, int] = {
    'USGS:08276500': 17864594,   # Lower Taos Box
    'USGS:09070500': 1236065,    # Colorado River near Dotsero
    'USGS:09114500': 3252637,    # Gunnison River near Gunnison
    'USGS:09163500': 3176490,    # Colorado River near CO-UT State Line (Westwater)
    'USGS:09180000': 18369916,   # Dolores River near Cisco
    'USGS:06719505': 2885178,    # Clear Creek at Golden
    'DWR:ARKCANCO':  949105,     # Arkansas River at Canon City (Royal Gorge, CNCC2)
    'DWR:CONMOGCO':  17876227,   # Conejos River near Mogote
}

_REACH_CACHE: Dict[str, Optional[int]] = dict(KNOWN_REACH_IDS)


def _disabled() -> bool:
    return os.getenv('OPENFLOW_DISABLE_PRIOR', '').strip() in ('1', 'true', 'True')


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=['Date', 'prior_flow'])


def resolve_reach_id(site_id: str, nws_lid: Optional[str] = None,
                     usgs_id: Optional[str] = None) -> Optional[int]:
    """Resolve an OpenFlow site_id to its NWM reachId / NHDPlus COMID."""
    if site_id in _REACH_CACHE:
        return _REACH_CACHE[site_id]
    if nws_lid:
        meta = nwps.get_gauge(nws_lid) or {}
        rid = meta.get('reachId')
        if rid:
            try:
                _REACH_CACHE[site_id] = int(rid)
                return _REACH_CACHE[site_id]
            except (TypeError, ValueError):
                pass
    if not usgs_id and site_id.startswith('USGS:'):
        usgs_id = site_id.split(':', 1)[1]
    if usgs_id:
        resp = data_utils.request_with_retry(NLDI_SITE_URL.format(usgs_id=usgs_id), timeout=30)
        if resp is not None:
            try:
                feats = resp.json().get('features') or []
                comid = feats[0]['properties'].get('comid') if feats else None
                if comid:
                    _REACH_CACHE[site_id] = int(comid)
                    return _REACH_CACHE[site_id]
            except (ValueError, KeyError, IndexError, TypeError):
                pass
    _REACH_CACHE[site_id] = None
    return None


def parse_nwm_reach_series(payload: Optional[dict], series_key: str = 'mediumRangeBlend',
                           tz: str = 'America/Denver') -> pd.Series:
    """
    Parse one block (`mediumRangeBlend`, `shortRange`, `analysisAssimilation`)
    of an NWPS NWM reach JSON response into a daily-mean Series (cfs, Date index).
    """
    block = (payload or {}).get(series_key) or {}
    series = block.get('series') or block.get('mean') or {}
    points = series.get('data') or []
    if not points:
        return pd.Series(dtype=float)
    units = str(series.get('units') or '').lower()
    mult = M3S_TO_CFS if ('m3' in units or 'm³' in units or 'cms' in units) else 1.0
    rows = []
    for p in points:
        t, v = p.get('validTime'), p.get('flow')
        if not t or v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if fv < 0 or fv <= nwps.MISSING:
            continue
        rows.append((t, fv * mult))
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows, columns=['time', 'flow'])
    df['time'] = pd.to_datetime(df['time'], utc=True, errors='coerce')
    df = df.dropna(subset=['time'])
    if df.empty:
        return pd.Series(dtype=float)
    local = df.set_index('time')['flow'].tz_convert(tz)
    daily = local.groupby(local.index.date).mean()
    daily.index = pd.DatetimeIndex(daily.index, name='Date')
    return daily.sort_index()


def fetch_nwm_live_daily(reach_id: int, tz: str = 'America/Denver') -> pd.Series:
    """Fetch live NWM operational analysis + short-range + medium-range blend (daily cfs)."""
    resp = data_utils.request_with_retry(NWPS_REACH_URL.format(reach_id=int(reach_id)), timeout=30)
    parts = []
    if resp is not None:
        try:
            payload = resp.json()
            for key in ('analysisAssimilation', 'shortRange', 'mediumRange', 'mediumRangeBlend'):
                s = parse_nwm_reach_series(payload, series_key=key, tz=tz)
                if len(s):
                    parts.append(s)
        except ValueError:
            pass
    # Also try explicit medium_range_blend query if the combined payload omitted it.
    if not any(len(p) >= 5 for p in parts):
        r2 = data_utils.request_with_retry(
            NWPS_REACH_URL.format(reach_id=int(reach_id)),
            params={'series': 'medium_range_blend'}, timeout=30,
        )
        if r2 is not None:
            try:
                s2 = parse_nwm_reach_series(r2.json(), series_key='mediumRangeBlend', tz=tz)
                if len(s2):
                    parts.append(s2)
            except ValueError:
                pass
    if not parts:
        return pd.Series(dtype=float)
    combined = pd.concat(parts)
    return combined[~combined.index.duplicated(keep='last')].sort_index()


def fetch_live_prior(site_id: str, nws_lid: Optional[str] = None,
                     usgs_id: Optional[str] = None,
                     tz: str = 'America/Denver') -> pd.DataFrame:
    """
    Live daily physics/operator prior DataFrame[Date, prior_flow] in cfs:
    prefers official NWS RFC forecast where current (captures planned dam
    releases), supplemented/extended by NOAA NWM v3 medium-range blend.
    """
    if _disabled():
        return _empty()
    rfc_daily = pd.Series(dtype=float)
    if nws_lid:
        try:
            payload = nwps.get_stageflow(nws_lid)
            fc = nwps.parse_series(payload, 'forecast').dropna(subset=['flow_cfs'])
            if not fc.empty:
                loc = fc.set_index('time')['flow_cfs'].tz_convert(tz)
                rfc_daily = loc.groupby(loc.index.date).mean()
                rfc_daily.index = pd.DatetimeIndex(rfc_daily.index, name='Date')
        except Exception as e:
            logger.debug("RFC live prior lookup skipped for %s: %s", nws_lid, e)

    nwm_daily = pd.Series(dtype=float)
    rid = resolve_reach_id(site_id, nws_lid=nws_lid, usgs_id=usgs_id)
    if rid is not None:
        try:
            nwm_daily = fetch_nwm_live_daily(rid, tz=tz)
        except Exception as e:
            logger.debug("NWM live prior lookup skipped for %s (%s): %s", site_id, rid, e)

    if rfc_daily.empty and nwm_daily.empty:
        return _empty()
    if rfc_daily.empty:
        out = nwm_daily
    elif nwm_daily.empty:
        out = rfc_daily
    else:
        # Prefer RFC on days it covers; extend past RFC end with ratio-aligned NWM.
        both = pd.concat([rfc_daily, nwm_daily], axis=1, keys=['r', 'n']).dropna()
        both = both[(both['r'] > 0) & (both['n'] > 0)]
        ratio = float(np.median(both['r'] / both['n'])) if len(both) else 1.0
        tail = nwm_daily[nwm_daily.index > rfc_daily.index.max()] * ratio
        head = nwm_daily[nwm_daily.index < rfc_daily.index.min()] * ratio
        out = pd.concat([head, rfc_daily, tail]).sort_index()
        out = out[~out.index.duplicated(keep='last')]
    return pd.DataFrame({'Date': out.index, 'prior_flow': out.to_numpy(dtype=float)})


def fetch_historical_prior(site_id: str, start_date, end_date,
                           nws_lid: Optional[str] = None,
                           usgs_id: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch historical daily NWM v3.0 retrospective streamflow (cfs) for
    `site_id` from AWS Open Data Zarr (1979-2023) when `xarray` + `zarr` +
    `s3fs` are installed. Degrades cleanly to empty when unavailable.
    """
    if _disabled():
        logger.info("OPENFLOW_DISABLE_PRIOR set -- skipping NWM historical prior for %s", site_id)
        return _empty()
    try:
        import s3fs
        import xarray as xr
        import zarr  # noqa: F401
    except ImportError:
        logger.info("NWM retrospective optional deps (xarray/zarr/s3fs) absent; skipping for %s", site_id)
        return _empty()
    rid = resolve_reach_id(site_id, nws_lid=nws_lid, usgs_id=usgs_id)
    if rid is None:
        return _empty()
    try:
        fs = s3fs.S3FileSystem(anon=True)
        store = s3fs.S3Map(root='noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr', s3=fs, check=False)
        ds = xr.open_zarr(store, consolidated=True)
        start_s = pd.Timestamp(start_date).strftime('%Y-%m-%d')
        end_s = pd.Timestamp(end_date).strftime('%Y-%m-%d')
        s = (ds['streamflow'].sel(feature_id=int(rid), time=slice(start_s, end_s))
             .resample(time='1D').mean().to_series().dropna())
        if s.empty:
            return _empty()
        df = pd.DataFrame({
            'Date': pd.DatetimeIndex(s.index).normalize(),
            'prior_flow': np.maximum(s.to_numpy(dtype=float) * M3S_TO_CFS, 0.0),
        })
        return df.sort_values('Date').reset_index(drop=True)
    except Exception as e:
        logger.warning("NWM v3 retrospective fetch failed for %s (reach %s): %s", site_id, rid, e)
        return _empty()
