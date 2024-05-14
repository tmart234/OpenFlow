import requests
import argparse
from swe_dicts import basins, subbasins
import pandas as pd
from datetime import datetime, timedelta
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

""" 
Given a basin or sub-basin (HU6 or HU8) and a date, look up historic SWE for a range of dates
"""

def is_leap(year):
    """Return True if year is a leap year."""
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def normalize_name(name):
    """Normalize names to match keys in the dictionary."""
    return name.lower().replace("/", "-") if name else None

def fetch_swe_data(url):
    """Fetch SWE data from a given URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return json.loads(response.text)
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed: {e}")
        return None

def get_swe_for_date(data, target_date, years_range):
    year_values = {}
    for year, value in data.items():
        if year.isdigit() and int(year) in years_range:
            year_values[year] = value
    return year_values

def get_swe_for_date_range(data, start_date, end_date):
    start_date = datetime.strptime(start_date, "%Y-%m-%d").date() if isinstance(start_date, str) else start_date
    end_date = datetime.strptime(end_date, "%Y-%m-%d").date() if isinstance(end_date, str) else end_date
    years_range = range(start_date.year, end_date.year + 1)
    
    all_dates = []
    all_values = []

    for entry in data:
        entry_date_str = entry['date']
        for year in years_range:
            try:
                if entry_date_str == "02-29" and not is_leap(year):
                    continue
                date_str = f"{year}-{entry_date_str}"
                entry_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                if start_date <= entry_date <= end_date:
                    year_data = entry.get(str(year), None)
                    if year_data is not None:
                        all_dates.append(entry_date)
                        all_values.append(year_data)
            except ValueError as e:
                logging.error(f"Error processing date {date_str}: {e}")
                continue
    
    # Create DataFrame
    result_df = pd.DataFrame({
        'Date': all_dates,
        'SWE Value': all_values
    })

    # Sort DataFrame by date
    result_df.sort_values('Date', inplace=True)
    return result_df


"""
def fetch_swe_station(start_date="2020-01-01", end_date="2021-01-01", station_id="360", state="MT"):
    url_template = "https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/customMultiTimeSeriesGroupByStationReport/daily/start_of_period/{station_id}:{state}:SNTL%7Cid=%22%22%7Cname/{start_date},{end_date}/WTEQ::value"
    url = url_template.format(
        start_date=start_date,
        end_date=end_date,
        station_id=station_id,
        state=state,
    )

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad responses
        content = response.text.splitlines()
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed: {e}")
        return {}

    data_dict = {}
    for line in content:
        if ',' not in line or line.strip().startswith('#'):
            continue  # skip comments and empty lines
        try:
            date_str, value = line.split(',')
            # Additional parsing to skip headers or malformed lines
            if value.replace('.', '', 1).isdigit():
                data_dict[date_str] = float(value)
        except ValueError as e:
            logging.error(f"Error processing line: {line} - {e}")

    return data_dict
"""

def main(basin_name, start_date, end_date):
    normalized_basin_name = normalize_name(basin_name)
    if normalized_basin_name in basins:
        url = basins[normalized_basin_name]
        data = fetch_swe_data(url)
        if data:
            result_df = get_swe_for_date_range(data, start_date, end_date)
            if not result_df.empty:  # Use .empty to check if the DataFrame has data
                print(result_df)
            else:
                print("No SWE data found within the specified date range.")
        else:
            logging.warn("Failed to fetch data.")
    else:
        logging.error(f"No data URL found for basin: {normalized_basin_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetch SWE data for a specified basin and date range.')
    parser.add_argument('--basin_name', type=str, default='South Platte', help='Name of the basin or sub-basin')
    parser.add_argument('--start_date', type=str, default=(datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d'), help='Start date in the format YYYY-MM-DD')
    parser.add_argument('--end_date', type=str, default=datetime.now().strftime('%Y-%m-%d'), help='End date in the format YYYY-MM-DD')
    args = parser.parse_args()

    # Ensure dates are properly converted to datetime.date objects
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()

    result_df = main(args.basin_name, start_date, end_date)
    if not result_df.empty:
        print(result_df)
    else:
        print("No SWE data found for the given parameters.")