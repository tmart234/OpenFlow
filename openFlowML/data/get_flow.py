import argparse
from concurrent.futures import ThreadPoolExecutor
import logging
import pandas as pd
from datetime import datetime, timedelta
from data.utils import data_utils

if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

USGS_IV_URL = "https://nwis.waterservices.usgs.gov/nwis/iv/"
USGS_DV_URL = "https://nwis.waterservices.usgs.gov/nwis/dv/"
# The IV service silently returns a partial response for multi-year requests,
# so we fetch it roughly one year at a time and concatenate.
CHUNK_DAYS = 366
# Daily Values responses are ~30x smaller than 15-min IV and handle 10-year
# chunks cleanly without truncation.
DV_CHUNK_DAYS = 10 * 366
# Polite concurrency across yearly IV chunks for one site.
IV_MAX_WORKERS = 4


def _parse_rdb_flow(text):
    """Parse a USGS IV RDB payload into a list of (date_str, flow) tuples."""
    rows = []
    for line in text.splitlines():
        if not line.startswith('USGS'):
            continue
        columns = line.split('\t')
        if len(columns) < 5:
            continue
        try:
            dt = datetime.strptime(columns[2][:16], "%Y-%m-%d %H:%M")
            flow = float(columns[4])
        except ValueError:
            # Header / format-spec row, missing value, or a non-numeric
            # qualifier in the value column -- skip it.
            continue
        if not (flow >= 0):
            # USGS uses negative sentinels (e.g. -999999) for ice/equipment outages.
            continue
        rows.append((dt.strftime("%Y-%m-%d"), flow))
    return rows


def _parse_rdb_dv_flow(text):
    """Parse a USGS Daily Values (statCd=00003) RDB payload into (date_str, flow) tuples."""
    date_idx = 2
    val_idx = 3
    header_seen = False
    rows = []
    for line in text.splitlines():
        if not line or line.startswith('#'):
            continue
        if not header_seen and line.startswith('agency_cd\t'):
            header = line.split('\t')
            if 'datetime' in header:
                date_idx = header.index('datetime')
            mean_cols = [
                i for i, col in enumerate(header)
                if col.endswith('_00060_00003') and not col.endswith('_cd')
            ]
            if mean_cols:
                val_idx = mean_cols[0]
            header_seen = True
            continue
        if not line.startswith('USGS\t'):
            continue
        columns = line.split('\t')
        if len(columns) <= max(date_idx, val_idx):
            continue
        date_raw = columns[date_idx][:10]
        try:
            datetime.strptime(date_raw, "%Y-%m-%d")
            flow = float(columns[val_idx])
        except ValueError:
            continue
        if not (flow >= 0):
            continue
        rows.append((date_raw, flow))
    return rows


def _fetch_iv_chunk(flow_site_id, chunk_start, chunk_end):
    params = {
        "sites": flow_site_id,
        "parameterCd": "00060",
        "startDT": chunk_start.strftime('%Y-%m-%d'),
        "endDT": chunk_end.strftime('%Y-%m-%d'),
        "siteStatus": "all",
        "format": "rdb",
    }
    logger.info("Fetching USGS IV flow for %s %s..%s",
                flow_site_id, params['startDT'], params['endDT'])
    response = data_utils.request_with_retry(USGS_IV_URL, params=params)
    if response is None:
        logger.error("Failed to fetch USGS flow chunk %s..%s for %s",
                     params['startDT'], params['endDT'], flow_site_id)
        return []
    return _parse_rdb_flow(response.text)


def _fetch_iv_records(flow_site_id, start_date, end_date):
    """
    Fetch USGS instantaneous flow as (date_str, flow) records.

    The request is split into ~yearly chunks because the IV service silently
    truncates long-range requests; each chunk is fetched with retries. Multiple
    chunks run through a small bounded thread pool to keep multi-decade fetches
    fast without overwhelming NWIS.
    """
    chunks = list(data_utils.date_chunks(start_date, end_date, CHUNK_DAYS))
    if not chunks:
        return []
    if len(chunks) == 1:
        return _fetch_iv_chunk(flow_site_id, chunks[0][0], chunks[0][1])

    records = []
    workers = min(IV_MAX_WORKERS, len(chunks))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_fetch_iv_chunk, flow_site_id, cs, ce) for cs, ce in chunks]
        for fut in futures:
            records.extend(fut.result())
    return records


def _fetch_dv_daily_mean(flow_site_id, start_date, end_date):
    """Fetch USGS Daily Values mean discharge (statCd=00003) in multi-year chunks."""
    records = []
    for chunk_start, chunk_end in data_utils.date_chunks(start_date, end_date, DV_CHUNK_DAYS):
        params = {
            "sites": flow_site_id,
            "parameterCd": "00060",
            "statCd": "00003",
            "startDT": chunk_start.strftime('%Y-%m-%d'),
            "endDT": chunk_end.strftime('%Y-%m-%d'),
            "siteStatus": "all",
            "format": "rdb",
        }
        logger.info("Fetching USGS DV daily mean for %s %s..%s",
                    flow_site_id, params['startDT'], params['endDT'])
        response = data_utils.request_with_retry(USGS_DV_URL, params=params)
        if response is None:
            logger.warning("Failed to fetch USGS DV chunk %s..%s for %s",
                           params['startDT'], params['endDT'], flow_site_id)
            continue
        records.extend(_parse_rdb_dv_flow(response.text))
    return records


def get_daily_flow_data(flow_site_id, start_date, end_date):
    """Fetch USGS instantaneous flow for a site and aggregate it to daily min/max."""
    records = _fetch_iv_records(flow_site_id, start_date, end_date)
    if not records:
        return pd.DataFrame(columns=['Date', 'Min Flow', 'Max Flow'])

    df = pd.DataFrame(records, columns=['Date', 'flow'])
    daily = df.groupby('Date')['flow'].agg(['min', 'max']).reset_index()
    daily.columns = ['Date', 'Min Flow', 'Max Flow']
    return daily.sort_values('Date').reset_index(drop=True)


def get_daily_mean_flow(flow_site_id, start_date, end_date):
    """
    Fetch USGS daily MEAN discharge as DataFrame[Date, flow]. Used for
    upstream-gauge features, where volume (mean) rather than the intraday range
    is what routes downstream.

    Queries the fast USGS Daily Values (`dv`, statCd=00003) archive first
    (~30x faster than 15-min IV and spans pre-2007 years), tops up any recent
    trailing days past DV's latest published date from IV, and falls back to IV
    when DV returns nothing.
    """
    dv_records = _fetch_dv_daily_mean(flow_site_id, start_date, end_date)
    if dv_records:
        df = pd.DataFrame(dv_records, columns=['Date', 'flow'])
        daily = df.groupby('Date', as_index=False)['flow'].mean()
        last_dv = pd.Timestamp(daily['Date'].max()).normalize()
        end_ts = pd.Timestamp(end_date).normalize()
        gap_days = int((end_ts - last_dv).days)
        if 0 < gap_days <= 14:
            tail_start = (last_dv + pd.Timedelta(days=1)).to_pydatetime()
            iv_tail = _fetch_iv_records(flow_site_id, tail_start, end_date)
            if iv_tail:
                tail_df = (pd.DataFrame(iv_tail, columns=['Date', 'flow'])
                           .groupby('Date', as_index=False)['flow'].mean())
                tail_df = tail_df[tail_df['Date'] > daily['Date'].max()]
                if not tail_df.empty:
                    daily = pd.concat([daily, tail_df], ignore_index=True)
        return daily.sort_values('Date').reset_index(drop=True)

    records = _fetch_iv_records(flow_site_id, start_date, end_date)
    if not records:
        return pd.DataFrame(columns=['Date', 'flow'])
    df = pd.DataFrame(records, columns=['Date', 'flow'])
    daily = df.groupby('Date', as_index=False)['flow'].mean()
    return daily.sort_values('Date').reset_index(drop=True)


def main(flow_site_id, start_date, end_date):
    df = get_daily_flow_data(flow_site_id, start_date, end_date)
    data_utils.preview_data(df)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetch daily min/max flow for a USGS site.')
    parser.add_argument('--flow_site_id', type=str, required=True, help='USGS flow site ID (ex: 09114500)')
    parser.add_argument('--start_date', type=str, required=True, help='Start date YYYY-MM-DD')
    parser.add_argument('--end_date', type=str, required=True, help='End date YYYY-MM-DD')
    args = parser.parse_args()
    start = datetime.strptime(args.start_date, '%Y-%m-%d')
    end = datetime.strptime(args.end_date, '%Y-%m-%d')
    main(args.flow_site_id, start, end)
