"""
USBR RISE reservoir storage + release timeseries by site.

For regulated rivers, downstream flow is heavily driven by reservoir
operations (releases) and storage state (how full the basin is). This module
maps a site_id to one or more upstream reservoirs via .github/reservoir_mapping.txt
and pulls daily storage (acre-feet) + release (cfs) from the public USBR RISE
catalog API.

Public API:
    main(site_id, start_date, end_date) -> DataFrame[Date, reservoir_storage,
                                                     reservoir_release]

Stations with no mapping entry (e.g. unregulated headwater gauges) get an
empty DataFrame back; combine_data treats that as "0 / not observed", which
is the correct semantic.
"""

import argparse
import logging
import os
from datetime import date, datetime
from typing import List, Optional, Tuple

import pandas as pd

from data.utils import data_utils

logger = logging.getLogger(__name__)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

RISE_RESULT_URL = "https://data.usbr.gov/rise/api/result"

# Reservoir mapping config lives alongside site_ids.txt in .github/.
_DEFAULT_MAPPING = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    '.github', 'reservoir_mapping.txt',
)


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=['Date', 'reservoir_storage', 'reservoir_release'])


def _to_date(d) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return datetime.strptime(str(d)[:10], '%Y-%m-%d').date()


def load_mapping(path: Optional[str] = None) -> dict:
    """
    Parse the reservoir-mapping config into
        {site_id: [(reservoir_name, storage_itemId, release_itemId), ...]}.

    Stations without entries are simply absent from the returned dict, which
    the caller treats as "no reservoir for this station".
    """
    if path is None:
        path = _DEFAULT_MAPPING
    mapping: dict = {}
    if not os.path.exists(path):
        logger.info("No reservoir mapping file at %s -- all stations unregulated", path)
        return mapping
    try:
        with open(path, 'r') as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith('#'):
                    continue
                parts = [p.strip() for p in line.split(',')]
                if len(parts) < 3:
                    logger.warning("Skipping malformed reservoir mapping line: %s", raw.rstrip())
                    continue
                site_id = parts[0]
                reservoir = parts[1]
                storage = parts[2] or None
                release = parts[3] if len(parts) >= 4 and parts[3] else None
                mapping.setdefault(site_id, []).append((reservoir, storage, release))
    except OSError as e:
        logger.warning("Could not read reservoir mapping %s: %s", path, e)
    return mapping


def fetch_rise_series(item_id, start_date, end_date) -> pd.DataFrame:
    """
    Pull a single RISE catalog-item timeseries as a [Date, value] frame.

    RISE's `/result` endpoint paginates; the public API caps page size, so we
    walk forward until the next-page URL is exhausted. Empty DataFrame on any
    failure -- the spine treats it as missing.
    """
    if not item_id:
        return pd.DataFrame(columns=['Date', 'value'])

    start = _to_date(start_date)
    end = _to_date(end_date)
    params = {
        'itemId': str(item_id),
        'after': start.strftime('%Y-%m-%dT00:00:00Z'),
        'before': end.strftime('%Y-%m-%dT23:59:59Z'),
        'order': 'ASC',
        'itemsPerPage': '10000',
    }
    headers = {'Accept': 'application/vnd.api+json'}

    rows: List[Tuple[str, float]] = []
    url = RISE_RESULT_URL
    next_params = params
    safety = 200  # hard cap on pagination loops to avoid runaway requests
    while url and safety > 0:
        safety -= 1
        response = data_utils.request_with_retry(url, params=next_params, headers=headers)
        if response is None:
            break
        try:
            payload = response.json()
        except ValueError:
            break
        for item in (payload.get('data') or []):
            attrs = item.get('attributes') or {}
            ts = attrs.get('dateTime') or attrs.get('resultDateTime')
            val = attrs.get('result')
            if ts is None or val is None:
                continue
            try:
                rows.append((str(ts)[:10], float(val)))
            except (TypeError, ValueError):
                continue
        # Next-page link: JSON:API uses links.next as an absolute URL with
        # query string baked in; once we follow it we drop params.
        links = payload.get('links') or {}
        nxt = links.get('next')
        if not nxt or nxt == url:
            break
        url = nxt
        next_params = None

    if not rows:
        return pd.DataFrame(columns=['Date', 'value'])
    df = pd.DataFrame(rows, columns=['Date', 'value'])
    # Collapse any duplicate dates within the series (hourly -> daily).
    df = df.groupby('Date', as_index=False)['value'].mean()
    return df.sort_values('Date').reset_index(drop=True)


def get_reservoir(site_id, start_date, end_date,
                  mapping_path: Optional[str] = None) -> pd.DataFrame:
    """
    Resolve the reservoir(s) mapped to site_id, fetch storage + release from
    RISE, and return a daily DataFrame[Date, reservoir_storage, reservoir_release]
    over the requested window.

    If multiple reservoirs map to one site, storage and release are summed
    (total water held back upstream, total outflow). Returns an empty frame
    when there is no mapping entry, mirroring the SWE / SMAP graceful pattern.
    """
    mapping = load_mapping(mapping_path)
    entries = mapping.get(site_id)
    if not entries:
        return _empty()

    storage_frames = []
    release_frames = []
    for reservoir_name, storage_id, release_id in entries:
        if storage_id:
            ts = fetch_rise_series(storage_id, start_date, end_date)
            if not ts.empty:
                ts = ts.rename(columns={'value': 'reservoir_storage'})
                storage_frames.append(ts)
        if release_id:
            tr = fetch_rise_series(release_id, start_date, end_date)
            if not tr.empty:
                tr = tr.rename(columns={'value': 'reservoir_release'})
                release_frames.append(tr)
        logger.info("Site %s reservoir %s: storage=%s release=%s",
                    site_id, reservoir_name,
                    'yes' if storage_id else '-',
                    'yes' if release_id else '-')

    def _sum_by_date(frames, col):
        if not frames:
            return pd.DataFrame(columns=['Date', col])
        combined = pd.concat(frames, ignore_index=True)
        return combined.groupby('Date', as_index=False)[col].sum()

    storage_total = _sum_by_date(storage_frames, 'reservoir_storage')
    release_total = _sum_by_date(release_frames, 'reservoir_release')

    if storage_total.empty and release_total.empty:
        return _empty()

    merged = (storage_total
              .merge(release_total, on='Date', how='outer')
              .sort_values('Date')
              .reset_index(drop=True))
    if 'reservoir_storage' not in merged.columns:
        merged['reservoir_storage'] = float('nan')
    if 'reservoir_release' not in merged.columns:
        merged['reservoir_release'] = float('nan')
    return merged[['Date', 'reservoir_storage', 'reservoir_release']]


def main(site_id, start_date, end_date,
         mapping_path: Optional[str] = None) -> pd.DataFrame:
    df = get_reservoir(site_id, start_date, end_date, mapping_path=mapping_path)
    data_utils.preview_data(df)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetch USBR RISE reservoir storage + release for a site.')
    parser.add_argument('--site-id', type=str, required=True, help='e.g. USGS:09163500')
    parser.add_argument('--start-date', type=str, required=True, help='YYYY-MM-DD')
    parser.add_argument('--end-date', type=str, required=True, help='YYYY-MM-DD')
    parser.add_argument('--mapping', type=str, default=None,
                        help='Override path to reservoir_mapping.txt')
    args = parser.parse_args()
    main(args.site_id, args.start_date, args.end_date, mapping_path=args.mapping)
