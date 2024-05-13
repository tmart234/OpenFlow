import requests
import argparse
from swe_dicts import basins, subbasins
import pandas as pd
from datetime import datetime, timedelta

""" 
Given a basin or sub-basin (HU6 or HU8) and a date, look up historic SWE
"""

def normalize_name(name):
    """Normalize names to match keys in the dictionary."""
    return name.lower().replace("/", "-") if name else None

def extract_swe_for_date(data, target_date):
    """Extract SWE values for a specific date from a data dictionary."""
    target_month_day = target_date.strftime('%m-%d')
    if data['date'] != target_month_day:
        print(f"No data available for {target_month_day}")
        return None
    
    current_year = datetime.now().year
    swe_value = data.get(str(current_year), None)
    if swe_value is None:
        print(f"No SWE data found for the year {current_year}.")
    else:
        print(f"SWE value for {target_month_day} of {current_year} is {swe_value}.")
    return swe_value

# fetch SWE from a station
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
        return {"error": str(e)}

    data_dict = {}

    for line in content:
        if ',' not in line and not line.strip().startswith('#'):
            continue  # skip lines without a comma
        try:
            date_str, value = line.split(',')
        except Exception as e:
            print(f"Error processing line: {line}")
            print(f"Exception: {e}")
            continue

        # Check if the second element is a number
        if not value.replace('.', '', 1).isdigit():
            continue

        data_dict[date_str] = float(value)

    return data_dict

def main(basin_name, target_date):
    normalized_basin_name = normalize_name(basin_name)
    if normalized_basin_name in basins:
        url = basins[normalized_basin_name]
        result_df = get_swe_data(url, target_date)
        return result_df
    else:
        print(f"No data URL found for basin: {normalized_basin_name}")
        return pd.DataFrame()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetch SWE data for a specified basin and date range.')
    parser.add_argument('--basin_name', type=str, default='South Platte', help='Name of the basin or sub-basin')
    parser.add_argument('--start_date', type=str, default="2020-01-01", help='Start date in the format YYYY-MM-DD')
    parser.add_argument('--end_date', type=str, default="2022-01-01", help='End date in the format YYYY-MM-DD')
    args = parser.parse_args()

    result_df = main(args.basin_name, args.start_date)
    if not result_df.empty:
        print(result_df)
    else:
        print("No SWE data found for the given parameters.")