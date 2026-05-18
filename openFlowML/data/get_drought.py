"""
US Drought Monitor (USDM) drought intensity timeseries by HUC8.

Wraps the public USDM data service at usdmdataservices.unl.edu. USDM publishes
weekly snapshots of percent area in each drought category (None / D0 / D1 /
D2 / D3 / D4); we collapse those to a single intensity index per week and
forward-fill to daily.

Public API:
    main(lat, lon, start_date, end_date) -> DataFrame[Date, drought_index]

drought_index = sum_c (D{c}_percent * weight_c) where weight is 1..5 for
D0..D4. Range 0 (no drought anywhere in HUC) to 500 (entire HUC in D4
exceptional drought). 0 is the legitimate default for missing rows.
"""

import argparse
import logging
from datetime import date, datetime
from typing import Optional

import pandas as pd

from data.utils import data_utils
from data import get_swe

logger = logging.getLogger(__name__)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

USDM_URL = "https://usdmdataservices.unl.edu/api/HUCStatistics/GetWeeklyHUCStatistics"

# Ordinal weights for the intensity index. None has weight 0 (it cancels).
_CATEGORY_WEIGHTS = {'D0': 1, 'D1': 2, 'D2': 3, 'D3': 4, 'D4': 5}


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=['Date', 'drought_index'])


def _to_date(d) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return datetime.strptime(str(d)[:10], '%Y-%m-%d').date()


def _format_mdy(d) -> str:
    """USDM API expects M/D/YYYY (no zero-padding)."""
    d = _to_date(d)
    return f"{d.month}/{d.day}/{d.year}"


def get_drought_weekly(huc_id, start_date, end_date, huc_level=8):
    """
    Fetch raw weekly USDM percent-area-by-category records for a HUC.

    Returns a list of dicts (one per ISO week) or [] on any failure.
    """
    if not huc_id:
        return []
    params = {
        'aoi': str(huc_id),
        'hucLevel': str(huc_level),
        'startdate': _format_mdy(start_date),
        'enddate': _format_mdy(end_date),
        'statisticsType': '2',  # percent area by drought category
    }
    headers = {'Accept': 'application/json'}
    response = data_utils.request_with_retry(USDM_URL, params=params, headers=headers)
    if response is None:
        return []
    try:
        payload = response.json()
    except ValueError:
        return []
    if not isinstance(payload, list):
        return []
    return payload


def _record_to_index(record) -> float:
    """
    Collapse a USDM weekly record (percent area per category) into a single
    weighted-intensity index. Missing category fields treated as 0.
    """
    total = 0.0
    for cat, weight in _CATEGORY_WEIGHTS.items():
        try:
            total += float(record.get(cat, 0) or 0) * weight
        except (TypeError, ValueError):
            continue
    return total


def _parse_record_date(record) -> Optional[date]:
    """
    USDM payloads expose the snapshot date under a few possible keys; tolerate
    all of them.
    """
    for key in ('MapDate', 'ValidStart', 'validStart', 'mapDate'):
        raw = record.get(key)
        if not raw:
            continue
        raw = str(raw)[:10]
        for fmt in ('%Y-%m-%d', '%Y%m%d'):
            try:
                return datetime.strptime(raw, fmt).date()
            except ValueError:
                continue
    return None


def get_drought(lat, lon, start_date, end_date, huc_level=8) -> pd.DataFrame:
    """
    End-to-end: lat/lon -> HUC -> USDM weekly intensity -> daily DataFrame.

    Weekly records are forward-filled across the daily index (USDM is a weekly
    snapshot; values stay constant for the week). Empty DataFrame on any
    failure (HUC lookup, API, parsing) so the spine treats it as "no data".
    """
    try:
        huc_id = get_swe.get_huc_id(lat, lon, level=huc_level)
    except Exception as e:
        logger.warning("HUC%d lookup failed for (%s, %s): %s", huc_level, lat, lon, e)
        return _empty()
    if not huc_id:
        logger.warning("Could not resolve HUC%d for (%s, %s)", huc_level, lat, lon)
        return _empty()

    records = get_drought_weekly(huc_id, start_date, end_date, huc_level=huc_level)
    if not records:
        logger.warning("No USDM records for HUC %s in [%s, %s]",
                       huc_id, start_date, end_date)
        return _empty()

    rows = []
    for record in records:
        d = _parse_record_date(record)
        if d is None:
            continue
        rows.append((d.strftime('%Y-%m-%d'), _record_to_index(record)))

    if not rows:
        return _empty()

    weekly = (pd.DataFrame(rows, columns=['Date', 'drought_index'])
              .drop_duplicates(subset=['Date'])
              .sort_values('Date'))
    weekly['Date'] = pd.to_datetime(weekly['Date'])
    weekly = weekly.set_index('Date')

    # Forward-fill weekly snapshots over the daily index.
    daily_index = pd.date_range(
        pd.Timestamp(_to_date(start_date)),
        pd.Timestamp(_to_date(end_date)),
        freq='D', name='Date',
    )
    daily = weekly.reindex(daily_index, method='ffill')
    daily = daily.reset_index()
    daily['Date'] = daily['Date'].dt.strftime('%Y-%m-%d')
    return daily[['Date', 'drought_index']]


def main(lat, lon, start_date, end_date, huc_level=8) -> pd.DataFrame:
    df = get_drought(lat, lon, start_date, end_date, huc_level=huc_level)
    data_utils.preview_data(df)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetch USDM drought intensity timeseries by HUC.')
    parser.add_argument('--lat', type=float, required=True)
    parser.add_argument('--lon', type=float, required=True)
    parser.add_argument('--start-date', type=str, required=True, help='YYYY-MM-DD')
    parser.add_argument('--end-date', type=str, required=True, help='YYYY-MM-DD')
    parser.add_argument('--huc-level', type=int, default=8, choices=[2, 4, 6, 8, 10, 12])
    args = parser.parse_args()
    main(args.lat, args.lon, args.start_date, args.end_date, huc_level=args.huc_level)
