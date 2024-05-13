import requests
import argparse
from swe_dicts import basins, subbasins
import pandas as pd
from datetime import datetime, timedelta
import json

""" 
Given a basin or sub-basin (HU6 or HU8) and a date, look up historic SWE for a range of dates
"""

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
        print(f"Request failed: {e}")
        return None

def get_swe_for_date(data, target_date):
    """Extract the SWE values for all available years for a specific date."""
    for entry in data:
        if entry['date'] == target_date:
            return {year: entry[year] for year in entry if year.isdigit()}  # Capture all year entries
    return None

def get_swe_for_date_range(data, start_date, end_date):
    """Get SWE values for a range of dates for all years available."""
    # Ensure start_date and end_date are datetime.date objects
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    elif isinstance(start_date, datetime):
        start_date = start_date.date()

    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
    elif isinstance(end_date, datetime):
        end_date = end_date.date()

    swe_values = {}

    # Iterate through the data for each date within the range
    for entry in data:
        try:
            # Parse the 'date' from the entry as just a month and day combination
            # Assuming that all entries are valid dates in non-leap years
            temp_date = datetime.strptime(entry['date'] + '-2000', "%m-%d-%Y").date()
            # Adjust the year to the current year for comparison
            entry_date = temp_date.replace(year=datetime.now().year)
        except ValueError as ve:
            print(f"Skipping invalid date in data: {entry['date']} - {ve}")
            continue

        if start_date <= entry_date <= end_date:
            year_data = get_swe_for_date(data, entry['date'])
            if year_data:
                swe_values[entry['date']] = year_data

    return swe_values


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

def main(basin_name, start_date, end_date):
    normalized_basin_name = normalize_name(basin_name)
    if normalized_basin_name in basins:
        url = basins[normalized_basin_name]
        data = fetch_swe_data(url)
        if data:
            result_data = get_swe_for_date_range(data, start_date, end_date)
            if result_data:
                result_df = pd.DataFrame(result_data)
                return result_df
            else:
                print("No SWE data found within the specified date range.")
        else:
            print("Failed to fetch data.")
    else:
        print(f"No data URL found for basin: {normalized_basin_name}")
    return pd.DataFrame()  # Ensure that a DataFrame is returned even if no data is found


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetch SWE data for a specified basin and date range.')
    parser.add_argument('--basin_name', type=str, default='South Platte', help='Name of the basin or sub-basin')
    parser.add_argument('--start_date', type=str, help='Start date in the format YYYY-MM-DD')
    parser.add_argument('--end_date', type=str, help='End date in the format YYYY-MM-DD')
    args = parser.parse_args()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7*365)

    result_df = main(args.basin_name, start_date, end_date)
    if not result_df.empty:
        print(result_df)
    else:
        print("No SWE data found for the given parameters.")