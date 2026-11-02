"""
Thin client for the NOAA National Water Prediction Service (NWPS) gauge API.

    https://api.water.noaa.gov/nwps/v1/gauges/{LID}
    https://api.water.noaa.gov/nwps/v1/gauges/{LID}/stageflow

NWPS serves, per NWS gauge location id (LID):
  - ~30 days of observed stage + flow (15-min to hourly cadence)
  - the current deterministic River Forecast Center (RFC) forecast, when one
    is in effect -- `issuedTime` plus a stage/flow series
  - gauge metadata: name, USGS id, NWM reach id, flood categories

In the stageflow payload, `primary` is STAGE (ft) and `secondary` is FLOW in
`secondaryUnits` (usually "kcfs"). Missing values are reported as -999. The
parsers below normalise all of that into plain cfs / ft with NaN for missing.

Used by the upstream-routing forecaster (routing.py / flow_forecast.py) for
observed + forecast hydrographs at a target gauge and its upstream gauges.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from data.utils import data_utils

logger = logging.getLogger(__name__)

NWPS_BASE_URL = "https://api.water.noaa.gov/nwps/v1"
MISSING = -999

# Multipliers from NWPS unit strings to cfs.
_FLOW_UNITS_TO_CFS = {'kcfs': 1000.0, 'cfs': 1.0}


def _empty_series() -> pd.DataFrame:
    return pd.DataFrame({'time': pd.Series(dtype='datetime64[ns, UTC]'),
                         'flow_cfs': pd.Series(dtype='float64'),
                         'stage_ft': pd.Series(dtype='float64')})


def get_gauge(lid: str) -> Optional[dict]:
    """Gauge metadata (name, usgsId, reachId, flood categories, status)."""
    response = data_utils.request_with_retry(f"{NWPS_BASE_URL}/gauges/{lid}")
    if response is None:
        return None
    try:
        return response.json()
    except ValueError:
        logger.error("NWPS returned non-JSON gauge metadata for %s", lid)
        return None


def get_stageflow(lid: str) -> Optional[dict]:
    """Raw NWPS stageflow payload: {'observed': {...}, 'forecast': {...}}."""
    response = data_utils.request_with_retry(f"{NWPS_BASE_URL}/gauges/{lid}/stageflow")
    if response is None:
        return None
    try:
        return response.json()
    except ValueError:
        logger.error("NWPS returned non-JSON stageflow for %s", lid)
        return None


def parse_series(payload: Optional[dict], kind: str) -> pd.DataFrame:
    """
    Normalise one half ('observed' or 'forecast') of a stageflow payload.

    Returns DataFrame[time (UTC), flow_cfs, stage_ft] sorted by time, with
    NaN wherever NWPS reported -999 / missing. An absent or empty series
    yields an empty frame (e.g. an RFC forecast that is not current).
    """
    if kind not in ('observed', 'forecast'):
        raise ValueError("kind must be 'observed' or 'forecast'")
    block = (payload or {}).get(kind) or {}
    points = block.get('data') or []
    if not points:
        return _empty_series()

    flow_units = (block.get('secondaryUnits') or 'kcfs').strip().lower()
    to_cfs = _FLOW_UNITS_TO_CFS.get(flow_units)
    if to_cfs is None:
        logger.warning("Unknown NWPS flow units %r; assuming kcfs", flow_units)
        to_cfs = 1000.0

    rows = []
    for p in points:
        t = p.get('validTime')
        if not t:
            continue
        flow = p.get('secondary')
        stage = p.get('primary')
        flow = np.nan if flow is None or flow <= MISSING else float(flow) * to_cfs
        stage = np.nan if stage is None or stage <= MISSING else float(stage)
        rows.append((t, flow, stage))
    if not rows:
        return _empty_series()
    df = pd.DataFrame(rows, columns=['time', 'flow_cfs', 'stage_ft'])
    df['time'] = pd.to_datetime(df['time'], utc=True, errors='coerce')
    df = df.dropna(subset=['time']).sort_values('time')
    # Collapse duplicate timestamps (re-transmissions) to their mean.
    df = df.groupby('time', as_index=False)[['flow_cfs', 'stage_ft']].mean()
    return df.reset_index(drop=True)


def forecast_issued_time(payload: Optional[dict]) -> Optional[pd.Timestamp]:
    """Issuance time of the RFC forecast, or None when no forecast is current."""
    block = (payload or {}).get('forecast') or {}
    issued = block.get('issuedTime')
    if not issued or not block.get('data'):
        return None
    ts = pd.to_datetime(issued, utc=True, errors='coerce')
    if pd.isna(ts) or ts.year < 1900:
        return None
    return ts


def flood_thresholds_cfs(gauge: Optional[dict]) -> dict:
    """
    Flood-category flow thresholds in cfs from gauge metadata, e.g.
    {'action': 7750.0, 'minor': 9800.0, ...}. Categories NWPS leaves
    undefined (-999 / missing) are omitted.
    """
    cats = ((gauge or {}).get('flood') or {}).get('categories') or {}
    out = {}
    for name in ('action', 'minor', 'moderate', 'major'):
        flow = (cats.get(name) or {}).get('flow')
        if flow is not None and flow > 0:
            out[name] = float(flow)
    return out


def hourly(df: pd.DataFrame, column: str = 'flow_cfs') -> pd.Series:
    """
    Resample a parsed series onto an hourly UTC grid (mean within the hour,
    then time-interpolate interior gaps). Returns an empty Series for empty
    input.
    """
    if df is None or df.empty or df[column].notna().sum() == 0:
        return pd.Series(dtype='float64', index=pd.DatetimeIndex([], tz='UTC', name='time'))
    s = df.set_index('time')[column].astype('float64')
    s = s.resample('1h').mean()
    return s.interpolate(method='time', limit_area='inside')
