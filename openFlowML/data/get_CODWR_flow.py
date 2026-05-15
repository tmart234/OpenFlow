import argparse
import logging
import pandas as pd
from datetime import datetime, timedelta
from data.utils import data_utils

if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

# The telemetry hourly endpoint only retains ~30 days of recent data;
# surfacewatertsday is the historical daily archive (per the project wiki).
DWR_DAILY_URL = "https://dwr.state.co.us/Rest/GET/api/v2/surfacewater/surfacewatertsday/"
PAGE_SIZE = 50000


def _empty_frame():
    return pd.DataFrame(columns=['Date', 'Min Flow', 'Max Flow'])


def get_historical_data(abbrev, start_date, end_date):
    """
    Fetch daily streamflow for a CO DWR station from the surfacewatertsday API.

    DWR's daily archive carries a single representative value per day rather
    than an intraday range, so Min Flow and Max Flow are both set to that value.
    Results are paginated; all pages are collected.
    """
    records = []
    page = 1
    while True:
        params = {
            "format": "json",
            "abbrev": abbrev,
            "min-measDate": start_date.strftime('%Y-%m-%d'),
            "max-measDate": end_date.strftime('%Y-%m-%d'),
            "pageSize": PAGE_SIZE,
            "pageIndex": page,
        }
        logger.info("Fetching DWR daily flow for %s (page %d)", abbrev, page)
        response = data_utils.request_with_retry(DWR_DAILY_URL, params=params)
        if response is None:
            logger.error("Failed to fetch DWR daily data for %s (page %d)", abbrev, page)
            break
        try:
            payload = response.json()
        except ValueError:
            logger.error("DWR returned a non-JSON response for %s", abbrev)
            break
        records.extend(payload.get('ResultList', []))
        page_count = payload.get('PageCount', 1) or 1
        if page >= page_count:
            break
        page += 1

    if not records:
        return _empty_frame()

    rows = []
    for record in records:
        meas_date = record.get('measDate')
        value = record.get('value')
        if meas_date is None or value is None:
            continue
        try:
            date = pd.to_datetime(meas_date).strftime('%Y-%m-%d')
            flow = float(value)
        except (ValueError, TypeError):
            continue
        rows.append((date, flow))

    if not rows:
        return _empty_frame()

    df = pd.DataFrame(rows, columns=['Date', 'flow'])
    daily = df.groupby('Date')['flow'].agg(['min', 'max']).reset_index()
    daily.columns = ['Date', 'Min Flow', 'Max Flow']
    return daily.sort_values('Date').reset_index(drop=True)


def main(abbrev, start_date=None, end_date=None):
    if start_date is None:
        start_date = datetime.now() - timedelta(days=5 * 365)
    if end_date is None:
        end_date = datetime.now()
    df = get_historical_data(abbrev, start_date, end_date)
    data_utils.preview_data(df)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetch historical daily flow for a CO DWR station.')
    parser.add_argument('--abbrev', type=str, required=True, help='Station abbreviation (ex: ARKCANCO)')
    parser.add_argument('--start_date', type=str, default=None, help='Start date YYYY-MM-DD')
    parser.add_argument('--end_date', type=str, default=None, help='End date YYYY-MM-DD')
    args = parser.parse_args()
    start = datetime.strptime(args.start_date, '%Y-%m-%d') if args.start_date else None
    end = datetime.strptime(args.end_date, '%Y-%m-%d') if args.end_date else None
    main(args.abbrev, start, end)
