import requests
import argparse
import pandas as pd
import logging
from datetime import datetime, timedelta
import ml_utils

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_historical_data(abbrev, start_date, end_date):
    # need hourly data to get min and max
    base_url = "https://dwr.state.co.us/Rest/GET/api/v2/telemetrystations/telemetrytimeserieshour/"
    
    params = {
        "format": "json",
        "dateFormat": "spaceSepToSeconds",
        "fields": "abbrev,parameter,measDate,measValue,measUnit",
        "encoding": "deflate",
        "abbrev": abbrev,
        "parameter": "DISCHRG",
        "min-measDate": start_date.strftime('%Y-%m-%d'),
        "max-measDate": end_date.strftime('%Y-%m-%d')
    }
    
    logging.info(f"Fetching historical data from: {base_url}")
    response = requests.get(base_url, params=params)
    logging.info(f"URL being requested: {response.request.url}")

    if response.status_code == 200:
        try:
            data = response.json()
            if not data or 'ResultList' not in data:
                logging.info("No data or unexpected data format returned for the given query.")
                return pd.DataFrame(columns=['Date', 'Min Discharge', 'Max Discharge'])
            
            date_dict = {}
            for record in data['ResultList']:
                date = datetime.strptime(record['measDate'], "%Y-%m-%d %H:%M:%S").date()
                flow = float(record['measValue'])
                if date in date_dict:
                    date_dict[date].append(flow)
                else:
                    date_dict[date] = [flow]
            
            dates = list(date_dict.keys())
            min_flows = [min(flows) for flows in date_dict.values()]
            max_flows = [max(flows) for flows in date_dict.values()]
            
            return pd.DataFrame({'Date': dates, 'Min Discharge': min_flows, 'Max Discharge': max_flows})
        except (ValueError, KeyError) as e:
            logging.error(f"Failed to parse JSON from response: {e}")
            return pd.DataFrame(columns=['Date', 'Min Discharge', 'Max Discharge'])
    else:
        logging.error(f"HTTP Error {response.status_code}: {response.text}")
        return pd.DataFrame(columns=['Date', 'Min Discharge', 'Max Discharge'])
    
def main(abbrev, start_date=None, end_date=None):
    if start_date is None:
        start_date = datetime.now() - timedelta(days=5*365)
    if end_date is None:
        end_date = datetime.now()
    
    df = get_historical_data(abbrev, start_date, end_date)
    ml_utils.preview_data(df)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetch historical data for a given abbreviation.')
    # ex: ARKCANCO
    parser.add_argument('--abbrev', type=str, required=True, help='Abbreviation for the station or measurement type')
    parser.add_argument('--start_date', type=str, default=True, help='Start date in the format MM/DD/YYYY')
    parser.add_argument('--end_date', type=str, default=True, help='End date in the format MM/DD/YYYY')
    args = parser.parse_args()
    
    # Convert input string dates to datetime objects if provided
    start_dt = datetime.strptime(args.start_date, '%m/%d/%Y') if args.start_date else None
    end_dt = datetime.strptime(args.end_date, '%m/%d/%Y') if args.end_date else None

    main(args.abbrev, start_dt, end_dt)
