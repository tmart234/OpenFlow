import argparse
import logging
import pandas as pd
from datetime import datetime
from data.utils import data_utils

if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

USGS_IV_URL = "https://nwis.waterservices.usgs.gov/nwis/iv/"
# The IV service silently returns a partial response for multi-year requests,
# so we fetch it roughly one year at a time and concatenate.
CHUNK_DAYS = 366


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
        rows.append((dt.strftime("%Y-%m-%d"), flow))
    return rows


def get_daily_flow_data(flow_site_id, start_date, end_date):
    """
    Fetch USGS instantaneous flow for a site and aggregate it to daily min/max.

    The request is split into ~yearly chunks because the IV service silently
    truncates long-range requests; each chunk is fetched with retries.
    """
    records = []
    for chunk_start, chunk_end in data_utils.date_chunks(start_date, end_date, CHUNK_DAYS):
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
            continue
        records.extend(_parse_rdb_flow(response.text))

    if not records:
        return pd.DataFrame(columns=['Date', 'Min Flow', 'Max Flow'])

    df = pd.DataFrame(records, columns=['Date', 'flow'])
    daily = df.groupby('Date')['flow'].agg(['min', 'max']).reset_index()
    daily.columns = ['Date', 'Min Flow', 'Max Flow']
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
