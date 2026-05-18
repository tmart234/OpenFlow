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

IMPLEMENTATION STATUS:
    The CBRFC publishes operational deterministic forecasts via the
    Advanced Hydrologic Prediction Service (AHPS) at water.weather.gov, and
    Ensemble Streamflow Prediction (ESP) products through their own portal at
    cbrfc.noaa.gov. Both have *current-day* access; the **historical archive**
    needed for backtesting against our test-set anchor dates is the gap:

      - AHPS does not expose historical issuance via its REST API; the
        archived forecasts live in tarballs at
        https://water.weather.gov/ahps/download.php
      - CBRFC's ESP archive is accessible per-basin via their THREDDS server
        but requires a per-issuance lookup that is meaningfully more involved
        than this stub.

    Filling either path in (the obvious follow-up commit) lets the baseline
    actually evaluate against the test set. Until then, fetch() returns empty
    for any anchor_date != today, and baseline_predictions() returns None so
    train.py skips the comparison cleanly.
"""

import logging
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
# Default 14-day horizon matches windowing.DECODER_DAYS.
DEFAULT_HORIZON_DAYS = 14


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=['Date', 'cbrfc_flow'])


def _to_date(d) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return datetime.strptime(str(d)[:10], '%Y-%m-%d').date()


def _ahps_lid_for_site(site_id: str) -> Optional[str]:
    """
    Map a USGS / DWR site id to an NWS / AHPS LID (5-letter location id).
    The mapping isn't algorithmic; it's a curated lookup. Returns None if no
    LID is known for the site, in which case fetch() returns empty.

    Populate this when wiring CBRFC for a specific gauge:
      {'USGS:09163500': 'CRSC2', ...}
    """
    return _AHPS_LID_TABLE.get(site_id)


_AHPS_LID_TABLE: dict = {
    # site_id -> NWS LID. Empty by default; fill in per gauge as needed.
}


def fetch_current(site_id: str, horizon_days: int = DEFAULT_HORIZON_DAYS) -> pd.DataFrame:
    """
    Pull the most recent AHPS forecast for `site_id`. Returns DataFrame
    [Date, cbrfc_flow] in cfs, one row per forecast day, or an empty frame
    when the site has no LID mapping or AHPS returned no usable forecast.

    The AHPS public forecast page only exposes the current forecast issuance,
    so this is the "what does CBRFC think tomorrow's flow is right now"
    helper. Historical issuances need the archive (see module docstring).
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


def fetch(site_id: str, anchor_date,
          horizon_days: int = DEFAULT_HORIZON_DAYS) -> pd.DataFrame:
    """
    CBRFC forecast for `site_id` issued on or before `anchor_date`, for the
    `horizon_days` days following the anchor.

    For anchor_date == today, this is equivalent to fetch_current. For any
    historical anchor_date, the AHPS API does not expose the issuance; this
    returns empty until the historical archive integration lands (see module
    docstring).
    """
    anchor = _to_date(anchor_date)
    if anchor >= date.today():
        return fetch_current(site_id, horizon_days=horizon_days)
    logger.debug("CBRFC historical forecast for %s @ %s requires archive integration",
                 site_id, anchor)
    return _empty()


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
    # Until the historical CBRFC archive is wired in there's nothing to
    # backtest against; signal "no comparison" cleanly to the caller.
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
