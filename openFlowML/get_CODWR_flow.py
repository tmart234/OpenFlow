import requests
import argparse
import pandas as pd
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_historical_data(abbrev, start_date, end_date):
    base_url = "https://dwr.state.co.us/Rest/GET/api/v2/surfacewater/surfacewatertsday/"
    
    params = {
        "format": "json",
        "dateFormat": "dateOnly",
        "fields": "abbrev,measDate,value,measUnit",
        "encoding": "deflate",
        "abbrev": abbrev,
        "min-measDate": start_date.strftime('%m/%d/%Y'),
        "max-measDate": end_date.strftime('%m/%d/%Y')
    }
    
    logging.info(f"Fetching historical data from: {base_url}")
    response = requests.get(base_url, params=params)
    #logging.info(f"URL being requested: {response.request.url}")

    if response.status_code == 200:
        try:
            data = response.json()
            if not data:
                logging.info("No data returned for the given query - this may be expected based on the dates queried.")
                return pd.DataFrame(columns=['Date', 'Discharge'])
            
            dates, discharge_values = [], []
            for record in data:
                meas_date = record['measDate']
                discharge = record['value']
                dates.append(meas_date)
                discharge_values.append(discharge)
            return pd.DataFrame({'Date': dates, 'Discharge': discharge_values})
        except ValueError:
            logging.error("Failed to parse JSON from response.")
            return pd.DataFrame(columns=['Date', 'Discharge'])
        except TypeError:
            # Instead of logging this as an error, log as info or warning if needed
            logging.info("Unexpected data format in JSON response, likely empty data.")
            return pd.DataFrame(columns=['Date', 'Discharge'])
    else:
        logging.error(f"HTTP Error {response.status_code}: {response.text}")
        return pd.DataFrame(columns=['Date', 'Discharge'])

def main(abbrev, start_date=None, end_date=None):
    if start_date is None:
        start_date = datetime.now() - timedelta(days=5*365)
    if end_date is None:
        end_date = datetime.now()
    
    df = get_historical_data(abbrev, start_date, end_date)
    logging.info(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetch historical data for a given abbreviation.')
    # ex: ARKCANCO
    parser.add_argument('--abbrev', type=str, required=True, help='Abbreviation for the station or measurement type')
    parser.add_argument('--start_date', type=str, default=None, help='Start date in the format MM/DD/YYYY')
    parser.add_argument('--end_date', type=str, default=None, help='End date in the format MM/DD/YYYY')
    args = parser.parse_args()
    
    # Convert input string dates to datetime objects if provided
    start_dt = datetime.strptime(args.start_date, '%m/%d/%Y') if args.start_date else None
    end_dt = datetime.strptime(args.end_date, '%m/%d/%Y') if args.end_date else None

    main(args.abbrev, start_dt, end_dt)
