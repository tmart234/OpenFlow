import os
from datetime import datetime, timedelta
import get_flow
import get_noaa
import normalize_data
import pandas as pd
from get_coordinates import get_coordinates
import logging
import re
import numpy as np

""" 
Takes multiuple individual data components and combindes into a dataset

scaling/Performance concerns: noaa script, pd.concat
 """

# This function will merge NOAA and flow data.
def merge_dataframes(noaa_data, flow_data, station_id):
     # Check if 'Date' column is in both dataframes
    if 'Date' not in noaa_data or 'Date' not in flow_data:
        raise ValueError("'Date' column missing in one of the dataframes")
    try:
        # Convert 'Date' columns to datetime format
        noaa_data['Date'] = pd.to_datetime(noaa_data['Date'], errors='coerce')
        flow_data['Date'] = pd.to_datetime(flow_data['Date'], errors='coerce')

        # Check for NaN values after conversion
        if noaa_data['Date'].isnull().any() or flow_data['Date'].isnull().any():
            raise ValueError("NaN values found in 'Date' column after conversion to datetime")

        # Add station_id to both dataframes
        noaa_data['stationID'] = station_id
        flow_data['stationID'] = station_id

        # Set 'Date' as the index
        noaa_data.set_index(['Date', 'stationID'], inplace=True)
        flow_data.set_index(['Date', 'stationID'], inplace=True)

        # Use an outer join to merge so we keep all dates and station IDs, filling missing values with NaN
        combined_data = pd.merge(noaa_data, flow_data, left_index=True, right_index=True, how='outer')

        # After merging, handle missing data for key columns.
        # You can choose to fill with mean, median, or a placeholder like -9999
        for col in ['TMAX', 'TMIN', 'Min Flow', 'Max Flow']:
            if col in combined_data.columns:
                # Filling missing values with the mean of the column
                combined_data[col].fillna(combined_data[col].mean(), inplace=True)

        # Reset index to bring 'Date' and 'stationID' back as columns
        combined_data.reset_index(inplace=True)

        return combined_data
    except Exception as e:
        logging.error(f"Error merging dataframes for station ID {station_id}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error
 
# This function will handle fetching and processing data for a single site ID.
def fetch_and_process_data(site_id, start_date, end_date):
    coords_dict = get_coordinates(site_id)
    latitude = coords_dict['latitude']
    longitude = coords_dict['longitude']

    # Fetch NOAA data
    closest_noaa_station, noaa_data = get_noaa.main(latitude, longitude, start_date, end_date)

        # Clean NOAA data
    if not noaa_data.empty:
        # Directly convert 'TMAX' and 'TMIN' columns to numeric, with non-numeric values turned into NaNs
        noaa_data['TMAX'] = pd.to_numeric(noaa_data['TMAX'], errors='coerce')
        noaa_data['TMIN'] = pd.to_numeric(noaa_data['TMIN'], errors='coerce')

    # Fetch flow data
    flow_data = get_flow.get_daily_flow_data(site_id, start_date, end_date)

    # Diagnostic logging to check 'Date' column in NOAA data
    if 'Date' in noaa_data.columns:
        logging.info(f"'Date' column present in NOAA data for site ID {site_id}")
    else:
        logging.error(f"'Date' column missing in NOAA data for site ID {site_id}")

    # Diagnostic logging to check 'Date' column in flow data
    if 'Date' in flow_data.columns:
        logging.info(f"'Date' column present in flow data for site ID {site_id}")
    else:
        logging.error(f"'Date' column missing in flow data for site ID {site_id}")


    if noaa_data.empty or flow_data.empty:
        logging.warning(f"No data available for site ID {site_id}. Skipping...")
        return None, None

    # Check for and handle missing or non-numeric values in key columns
    for df in [noaa_data, flow_data]:
        for col in ['TMAX', 'TMIN', 'Min Flow', 'Max Flow']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col].fillna(df[col].mean(), inplace=True)  # Fill missing values with mean

    return noaa_data, flow_data
 
 # Additional function to display the beginning and ending of the dataframe
def preview_data(df, num_rows=4):
    logging.info("First few rows:")
    logging.info(df.head(num_rows))
    logging.info("\nLast few rows:")
    logging.info(df.tail(num_rows))

def get_site_ids(filename=None):
    if filename is None:
        filename = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.github', 'site_ids.txt')
    with open(filename, 'r') as f:
        return [line.strip() for line in f]

# This function will save the combined data from all site IDs.
def save_combined_data(all_data, base_path):
    final_data = pd.DataFrame()
    for site_id, data in all_data.items():
        final_data = pd.concat([final_data, data])

    # Preview the combined data
    preview_data(final_data)

    # Save the combined raw data to a CSV file
    combined_data_file_path = os.path.join(base_path, 'openFlowML', 'combined_data_all_sites.csv')
    final_data.to_csv(combined_data_file_path, index=False)

    # Apply normalization which includes one-hot encoding within the normalization function
    normalized_data = normalize_data(combined_data_file_path, final_data)

    # Save the normalized data to a separate CSV file
    normalized_data_path = os.path.join(base_path, 'openFlowML', 'normalized_data.csv')
    normalized_data.to_csv(normalized_data_path, index=False)

    return normalized_data

# usable for for GH actions or local testing workflow
def get_base_path():
    if 'GITHUB_WORKSPACE' in os.environ:
        return os.environ['GITHUB_WORKSPACE']
    else:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def parse_datetime(datetime_str):
    # Use regular expressions to extract the datetime part
    match = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', datetime_str)
    if match:
        datetime_str = match.group(0)
        try:
            # Try parsing the datetime
            return datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
        except ValueError as e:
            print(f"Error parsing datetime: {e}")
            return None
    else:
        print(f"No valid datetime found in string: {datetime_str}")
        return None
    
def get_site_ids(filename=None):
    if filename is None:
        # Construct the relative path to the site_ids.txt
        filename = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.github', 'site_ids.txt')
        
    with open(filename, 'r') as f:
        return [line.strip() for line in f]

def main(training_num_years = 7):
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    all_data = {}
    
    # Extract and encode site ids
    site_ids = get_site_ids()
    
    base_path = get_base_path()

    # Get dates for the last n years
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=training_num_years*365)).strftime('%Y-%m-%d')

    # Iterate over each site ID
    for site_id in site_ids:
        try:
            coords_dict = get_coordinates(site_id)
            latitude = coords_dict['latitude']
            longitude = coords_dict['longitude']

            # Call NOAA and get_flow
            closest_noaa_station, noaa_data = get_noaa.main(latitude, longitude, start_date, end_date)
            flow_data = get_flow.get_daily_flow_data(site_id, start_date, end_date)

            # Skip if no data is available
            if noaa_data.empty or flow_data.empty:
                logging.warning(f"No data available for site ID {site_id}. Skipping...")
                continue

            # Merge NOAA and flow data
            combined_data = merge_dataframes(noaa_data, flow_data, site_id)
            all_data[site_id] = combined_data
        except Exception as e:
            logging.error(f"An error occurred for site ID {site_id}: {e}")

    # Save combined data if available
    if all_data:
        final_data = save_combined_data(all_data, base_path)
        return final_data
    else:
        logging.error("No combined data for all sites")
        return None

if __name__ == "__main__":
    main()