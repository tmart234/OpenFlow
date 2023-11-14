import os
from datetime import datetime, timedelta
import get_flow
import get_noaa
import normalize_data
import pandas as pd
from get_coordinates import get_coordinates
import sys
import logging
import re
from tensorflow.keras.preprocessing.text import Tokenizer

""" 
Takes multiuple individual data components and combindes into a dataset

scaling/Performance concerns: noaa script, pd.concat
 """
# This function will merge NOAA and flow data.
def merge_dataframes(noaa_data, flow_data):
    try:
        # Ensure 'Date' columns are in datetime format
        noaa_data['Date'] = pd.to_datetime(noaa_data['Date'])
        flow_data['Date'] = pd.to_datetime(flow_data['Date'])

        # Set 'Date' as the index
        noaa_data.set_index('Date', inplace=True)
        flow_data.set_index('Date', inplace=True)

        # Merge the dataframes
        combined_data = pd.merge(noaa_data, flow_data, left_index=True, right_index=True, how='outer')
        combined_data.reset_index(inplace=True)

        return combined_data
    except Exception as e:
        logging.error(f"Error merging dataframes: {e}")
        return None
 
# This function will handle fetching and processing data for a single site ID.
def fetch_and_process_data(site_id, start_date, end_date):
    coords_dict = get_coordinates(site_id)
    latitude = coords_dict['latitude']
    longitude = coords_dict['longitude']

    # Fetch NOAA data
    closest_noaa_station, noaa_data = get_noaa.main(latitude, longitude, start_date, end_date)

    # Fetch flow data
    flow_data = get_flow.get_daily_flow_data(site_id, start_date, end_date)

    # Check if 'Date' column exists in both dataframes
    if 'Date' not in noaa_data.columns:
        logging.error(f"'Date' column missing in NOAA data for site ID {site_id}")
        return None, None
    if 'Date' not in flow_data.columns:
        logging.error(f"'Date' column missing in flow data for site ID {site_id}")
        return None, None

    if noaa_data.empty or flow_data.empty:
        logging.warning(f"No data available for site ID {site_id}. Skipping...")
        return None, None

    return noaa_data, flow_data
 
 # Additional function to display the beginning and ending of the dataframe
def preview_data(df, num_rows=4):
    print("First few rows:")
    print(df.head(num_rows))
    print("\nLast few rows:")
    print(df.tail(num_rows))

def get_site_ids_with_embedding(filename=None):
    if filename is None:
        filename = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.github', 'site_ids.txt')
        
    with open(filename, 'r') as f:
        site_ids = [line.strip() for line in f]
        
    tokenizer = Tokenizer(char_level=False)
    tokenizer.fit_on_texts(site_ids)
    return tokenizer, [tokenizer.texts_to_sequences([site_id])[0][0] for site_id in site_ids]

# This function will save the combined data from all site IDs.
def save_combined_data(all_data, base_path):
    final_data = pd.DataFrame()
    for site_id, data in all_data.items():
        final_data = pd.concat([final_data, data])

    # Preview the combined data
    preview_data(final_data)

    combined_data_file_path = os.path.join(base_path, 'openFlowML', 'combined_data_all_sites.csv')
    final_data.to_csv(combined_data_file_path, index=False)

    # Save normalized data as a separate CSV
    normalized_data_path = os.path.join(base_path, 'openFlowML', 'normalized_data.csv')
    final_data = normalize_data.normalize_data(combined_data_file_path, final_data)
    final_data.to_csv(normalized_data_path, index=False)

    return final_data

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
    tokenizer, encoded_site_ids = get_site_ids_with_embedding()
    site_ids = get_site_ids()
    
    base_path = get_base_path()

    # Get dates for the last n years
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=training_num_years*365)).strftime('%Y-%m-%d')

    for site_id, encoded_site_id in zip(site_ids, encoded_site_ids):
        try:
            coords_dict = get_coordinates(site_id)
            latitude = coords_dict['latitude']
            longitude = coords_dict['longitude']

            # Call noaa and get_flow
            closest_noaa_station, noaa_data = get_noaa.main(latitude, longitude, start_date, end_date)
            flow_data = get_flow.get_daily_flow_data(site_id, start_date, end_date)

            if noaa_data.empty or flow_data.empty:
                logging.warning(f"No data available for site ID {site_id}. Skipping...")
                continue

            # Embedding the site_id for later use in the model
            noaa_data['site_id_encoded'] = encoded_site_id
            flow_data['site_id_encoded'] = encoded_site_id

            combined_data = merge_dataframes(noaa_data, flow_data)
            all_data[site_id] = combined_data
        except Exception as e:
            logging.error(f"An error occurred for site ID {site_id}: {e}")

    if all_data:
        final_data = save_combined_data(all_data, base_path)
        return final_data
    else:
        logging.error("No combined data for all sites")
        return None

if __name__ == "__main__":
    main()
