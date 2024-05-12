import requests
import argparse
import pandas as pd
import logging
from datetime import datetime, timedelta
import ml_utils

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_daily_flow_data(flow_site_id, start_date, end_date):
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')
    url = f"https://nwis.waterservices.usgs.gov/nwis/iv/?sites={flow_site_id}&parameterCd=00060&startDT={start_date}&endDT={end_date}&siteStatus=all&format=rdb"
    logging.info(f"Fetching flow data from: {url}")
    response = requests.get(url)
    content = response.text.splitlines()

    dates, min_flows, max_flows = [], [], []
    for index, line in enumerate(content):
        if line.startswith('USGS'):
            data_lines = content[index:]
            break
    else:
        return pd.DataFrame(columns=['Date', 'Min Discharge', 'Max Discharge'])  # Return empty DataFrame if no data lines found

    for line in data_lines:
        columns = line.split('\t')
        if len(columns) >= 5:
            datetime_str = columns[2][:16]  # Extracts "YYYY-MM-DD HH:MM" part
            try:
                dt = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M")
            except ValueError:
                logging.error(f"Skipping line due to unexpected datetime format: {line}")
                continue  # Skip this line entirely
            date_str = dt.strftime("%Y-%m-%d")

            flow = float(columns[4])
            if date_str in dates:
                index = dates.index(date_str)
                min_flows[index] = min(min_flows[index], flow)
                max_flows[index] = max(max_flows[index], flow)
            else:
                dates.append(date_str)
                min_flows.append(flow)
                max_flows.append(flow)

    df = pd.DataFrame({'Date': dates, 'Min Discharge': min_flows, 'Max Discharge': max_flows})
    return df

def main(flow_site_id='09114500', start_date=None, end_date=None):
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=7*365 + 7)).strftime('%Y-%m-%d')
    if end_date is None:
        end_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

    df = get_daily_flow_data(flow_site_id, start_date, end_date)
    ml_utils.preview_data(df)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetch daily flow data for a given USGS site.')
    parser.add_argument('--flow_site_id', type=str, default='09114500', help='USGS flow site ID (default: 09114500)')
    parser.add_argument('--start_date', type=str, default=None, help='Start date in the format YYYY-MM-DD')
    parser.add_argument('--end_date', type=str, default=None, help='End date in the format YYYY-MM-DD')
    args = parser.parse_args()
    main(args.flow_site_id, args.start_date, args.end_date)