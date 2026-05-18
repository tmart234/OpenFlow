"""
NOAA Colorado Basin River Forecast Center (CBRFC) streamflow forecast baseline.

A second comparison baseline alongside persistence. The model needs to beat
CBRFC's official short-range forecast to claim it's adding value over what an
operational forecaster has access to.

Public API:
    fetch(site_id, anchor_date, horizon_days) -> DataFrame[Date, cbrfc_flow]
        Forecast issued on or before `anchor_date`, predictions for the next
        `horizon_days`. Empty DataFrame when no usable forecast exists.

    baseline_predictions(test_samples) -> Optional[np.ndarray]
        Stack predictions for the entire test set into a (N, horizon, 2)
        tensor (same shape as persistence_baseline output in train.py). Returns
        None when CBRFC coverage is sparse enough that the comparison would
        be meaningless.

Data sources:
  - Live (anchor_date >= today): AHPS public hydrograph XML at water.weather.gov.
    The page only exposes the current issuance, so this is the "what does CBRFC
    think tomorrow's flow is right now" lookup.
  - Historical (anchor_date < today): NWS NWPS forecast archive at
    api.water.noaa.gov, which accepts a `reference_time` query parameter and
    returns the deterministic stage/flow forecast issued at that timestamp.
    This is the path used for backtesting against test-set anchor dates.

LID mapping (OpenFlow site_id -> NWS LID) is curated in cbrfc_lid_map.json
alongside this file. A missing mapping is a silent skip -- CBRFC is an
optional comparison baseline, not a training input.
"""

import json
import logging
import os
from datetime import date, datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd

from data.utils import data_utils

logger = logging.getLogger(__name__)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# AHPS public site forecast page (one page per gauge by NWS LID).
AHPS_FORECAST_URL = "https://water.weather.gov/ahps2/hydrograph_to_xml.php"
# NWPS forecast archive: accepts a historical `reference_time` and returns the
# deterministic stage/flow forecast issued at that timestamp as JSON.
NWPS_FORECAST_URL = "https://api.water.noaa.gov/nwps/v1/gauges/{lid}/stageflow/forecast"
# Default 14-day horizon matches windowing.DECODER_DAYS.
DEFAULT_HORIZON_DAYS = 14

_LID_MAP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'cbrfc_lid_map.json')


def _load_lid_table() -> dict:
    """Load site_id -> NWS LID from disk. Cached on the function."""
    cached = getattr(_load_lid_table, '_cache', None)
    if cached is not None:
        return cached
    try:
        with open(_LID_MAP_PATH) as f:
            raw = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning("Could not read %s (%s); CBRFC baseline disabled", _LID_MAP_PATH, e)
        raw = {}
    # Drop comment / metadata keys (anything starting with underscore).
    table = {k: v for k, v in raw.items() if not k.startswith('_')}
    _load_lid_table._cache = table
    return table


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=['Date', 'cbrfc_flow'])


def _to_date(d) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return datetime.strptime(str(d)[:10], '%Y-%m-%d').date()


def _ahps_lid_for_site(site_id: str) -> Optional[str]:
    """Map an OpenFlow site_id (USGS:XXXX or DWR:XXXX) to its NWS LID, or None."""
    return _load_lid_table().get(site_id)


def fetch_current(site_id: str, horizon_days: int = DEFAULT_HORIZON_DAYS) -> pd.DataFrame:
    """
    Pull the most recent AHPS forecast for `site_id`. Returns DataFrame
    [Date, cbrfc_flow] in cfs, one row per forecast day, or an empty frame
    when the site has no LID mapping or AHPS returned no usable forecast.

    The AHPS public forecast page only exposes the current forecast issuance,
    so this is the "what does CBRFC think tomorrow's flow is right now"
    helper. Historical issuances use the NWPS archive (_fetch_nwps_historical).
    """
    lid = _ahps_lid_for_site(site_id)
    if not lid:
        logger.info("No AHPS LID mapped for %s; CBRFC fetch skipped", site_id)
        return _empty()
    params = {'gage': lid, 'output': 'xml'}
    response = data_utils.request_with_retry(AHPS_FORECAST_URL, params=params)
    if response is None:
        return _empty()
    rows = _parse_ahps_forecast_xml(response.text, horizon_days)
    if not rows:
        return _empty()
    return pd.DataFrame(rows, columns=['Date', 'cbrfc_flow'])


def _parse_ahps_forecast_xml(text: str, horizon_days: int) -> List[tuple]:
    """
    Extract daily forecast (date, cfs) rows from the AHPS hydrograph XML.

    AHPS XML wraps a `<forecast>` block of `<datum>` elements, each with a
    `<valid>` ISO timestamp and a `<primary>` numeric value (typically flow
    in kcfs or stage in ft -- the gauge metadata tells you which). Lifts the
    primary value, collapses sub-daily issuances to daily mean, caps at
    horizon_days days from the issuance date.
    """
    try:
        from xml.etree import ElementTree as ET
    except ImportError:
        return []
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return []
    rows: list = []
    forecast = root.find('forecast')
    if forecast is None:
        return rows
    for datum in forecast.findall('datum'):
        valid = datum.findtext('valid')
        primary = datum.findtext('primary')
        if not valid or primary is None:
            continue
        try:
            d = datetime.fromisoformat(valid.replace('Z', '+00:00')).date()
            v = float(primary)
        except (ValueError, TypeError):
            continue
        rows.append((d.strftime('%Y-%m-%d'), v))
    if not rows:
        return rows
    # Collapse multiple values per day to daily mean.
    df = pd.DataFrame(rows, columns=['Date', 'cbrfc_flow'])
    daily = df.groupby('Date', as_index=False)['cbrfc_flow'].mean()
    daily = daily.sort_values('Date').head(horizon_days)
    return list(daily.itertuples(index=False, name=None))


def _fetch_nwps_historical(lid: str, anchor: date,
                           horizon_days: int) -> List[tuple]:
    """
    Pull the CBRFC forecast issued at `anchor` from the NWPS archive.

    Returns a list of (YYYY-MM-DD, cfs) tuples for the next `horizon_days`
    days after the issuance, or an empty list on any error (404, parse
    failure, no forecast for that anchor). Sub-daily values are collapsed to
    daily mean to match the persistence / model output cadence.
    """
    url = NWPS_FORECAST_URL.format(lid=lid)
    # NWPS issues its deterministic forecast around 12Z; align the reference
    # time there so any anchor lands on a real issuance.
    reference_time = f"{anchor.strftime('%Y-%m-%d')}T12:00:00Z"
    params = {'reference_time': reference_time}
    response = data_utils.request_with_retry(url, params=params)
    if response is None:
        return []
    try:
        payload = response.json()
    except ValueError:
        logger.debug("NWPS returned non-JSON for %s @ %s", lid, anchor)
        return []
    # NWPS schema: {"data": [{"validTime": "...", "primary": "..."}, ...]}.
    # The wrapper key drifts between schema versions; tolerate both shapes.
    data = payload.get('data') or payload.get('forecast', {}).get('data') or []
    if not data:
        return []
    rows = []
    for point in data:
        valid = point.get('validTime') or point.get('valid')
        primary = point.get('primary')
        if not valid or primary is None:
            continue
        try:
            d = datetime.fromisoformat(str(valid).replace('Z', '+00:00')).date()
            v = float(primary)
        except (ValueError, TypeError):
            continue
        rows.append((d.strftime('%Y-%m-%d'), v))
    if not rows:
        return []
    df = pd.DataFrame(rows, columns=['Date', 'cbrfc_flow'])
    daily = df.groupby('Date', as_index=False)['cbrfc_flow'].mean()
    daily = daily.sort_values('Date').head(horizon_days)
    return list(daily.itertuples(index=False, name=None))


def fetch(site_id: str, anchor_date,
          horizon_days: int = DEFAULT_HORIZON_DAYS) -> pd.DataFrame:
    """
    CBRFC forecast for `site_id` issued on or before `anchor_date`, for the
    `horizon_days` days following the anchor.

    For anchor_date >= today, returns the current AHPS issuance (live path).
    For any historical anchor_date, queries the NWPS forecast archive at the
    matching reference_time and returns the issuance from that day.
    """
    anchor = _to_date(anchor_date)
    if anchor >= date.today():
        return fetch_current(site_id, horizon_days=horizon_days)
    lid = _ahps_lid_for_site(site_id)
    if not lid:
        logger.info("No AHPS LID mapped for %s; CBRFC historical skipped", site_id)
        return _empty()
    rows = _fetch_nwps_historical(lid, anchor, horizon_days)
    if not rows:
        return _empty()
    return pd.DataFrame(rows, columns=['Date', 'cbrfc_flow'])


def baseline_predictions(test_samples) -> Optional[np.ndarray]:
    """
    Per-sample CBRFC forecast aligned with `windowing.WindowedSample` test
    items. Returns an (N, horizon, target_features) array on the SCALED flow
    target space (so it lines up with model_pred), or None when fewer than
    one usable forecast was found and the comparison would be vacuous.

    This is the integration point train.py calls; when filled in, it produces
    a third row in the per-horizon MAE table alongside model and persistence.
    The shape mirrors `_persistence_pred_for_samples` in train.py.
    """
    if not test_samples:
        return None
    found = 0
    horizon = test_samples[0].target_Y.shape[0]
    rows = []
    for sample in test_samples:
        # Forecast was issued at the end of the encoder window -- that's
        # `anchor_date + encoder_days` for a sample whose anchor_date is the
        # first encoder day. WindowedSample doesn't store the forecast-issue
        # date explicitly; reconstruct as anchor + (encoder_days - 1).
        forecast_issue = (sample.anchor_date
                          + pd.Timedelta(days=sample.encoder_X.shape[0] - 1))
        forecast = fetch(sample.site_id, forecast_issue, horizon_days=horizon)
        if forecast.empty:
            rows.append(np.full((horizon, sample.target_Y.shape[1]), np.nan,
                                dtype='float32'))
            continue
        found += 1
        # The fetched cbrfc_flow is on the raw cfs scale; train.py knows how
        # to invert the scalers if a non-scaled prediction set is provided.
        # Broadcast the single CBRFC value across both Min Flow + Max Flow
        # target columns until the upstream product distinguishes them.
        values = forecast['cbrfc_flow'].astype('float32').to_numpy()
        if len(values) < horizon:
            padded = np.full(horizon, np.nan, dtype='float32')
            padded[:len(values)] = values
            values = padded
        rows.append(np.tile(values[:horizon, None], (1, sample.target_Y.shape[1])))

    if found == 0:
        return None
    return np.stack(rows).astype('float32')
