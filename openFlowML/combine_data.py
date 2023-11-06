import os
from datetime import datetime, timedelta
import get_flow
import get_noaa
import normalize_data
import pandas as pd
from get_coordinates import get_coordinates
import sys
import logging

""" 
Takes multiuple individual data components and combindes into a dataset

scaling/Performance concerns: noaa script, pd.concat
 """

def parse_date(x):
    try:
        return datetime.strptime(x, "%Y-%m-%d %H:%M")
    except ValueError:
        # Handle other date formats or raise the exception if necessary
        return x
    
def get_site_ids(filename=None):
    if filename is None:
        # Construct the relative path to the site_ids.txt
        filename = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.github', 'site_ids.txt')
        
    with open(filename, 'r') as f:
        return [line.strip() for line in f]

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    all_data = []
    # Extract site ids
    site_ids = get_site_ids()
    for site_id in site_ids:
        try:
            # Check if 'GITHUB_WORKSPACE' is in the environment variables
            if 'GITHUB_WORKSPACE' in os.environ:
                base_path = os.environ['GITHUB_WORKSPACE']
            # if not use script's current executing directory
            else:
                # Get the directory of the currently executing script
                base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
            # Get dates for the last 5 years
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
            
            # Fetch coords
            coords_dict = get_coordinates(site_id)
            latitude = coords_dict['latitude']
            longitude = coords_dict['longitude']

            # Call noaa
            temp_data = {}
            closest_noaa_station, temp_data = get_noaa.main(latitude, longitude, start_date, end_date)
            if not temp_data.empty:
                noaa_data = pd.DataFrame.from_dict(temp_data)
                noaa_data.reset_index(inplace=True)
                noaa_data.rename(columns={'index': 'Date'}, inplace=True)
            else:
                logging.warning(f"No NOAA data available for site ID {site_id}. Skipping...")
                continue

            # Call the get_flow function
            flow_data = get_flow.get_daily_flow_data(site_id, start_date, end_date)
            if flow_data.empty:
                logging.warning(f"No Flow data available for site ID {site_id}. Skipping...")
                continue
            
            flow_data.rename(columns={'index': 'Date'}, inplace=True)
            
            # Ensure 'Date' columns are in datetime format
            noaa_data['Date'] = pd.to_datetime(noaa_data['Date'])
            flow_data['Date'] = pd.to_datetime(flow_data['Date'])

            # Ensure the dates are the index for both dataframes for proper alignment
            flow_data.set_index('Date', inplace=True)
            noaa_data.set_index('Date', inplace=True)

            # Merge the two dataframes on the index (which is 'Date' for both)
            combined_data = pd.merge(noaa_data, flow_data, left_index=True, right_index=True, how='outer')
            combined_data.reset_index(inplace=True)

            all_data.append(combined_data)

        except Exception as e:
             # Log error
            logging.error(f"An error occurred for site ID {site_id}: {e}")

    # Combine data from all site ids and save
    if all_data:
        final_data = pd.concat(all_data)
        combined_data_file_path = os.path.join(base_path, 'openFlowML', 'combined_data_all_sites.csv')
        final_data.to_csv(combined_data_file_path, index=False)

        # Save normalized data as a separate CSV
        normalized_data_path = os.path.join(base_path, 'openFlowML', 'normalized_data.csv')
        final_data = normalize_data.normalize_data(combined_data_file_path, final_data)
        final_data.to_csv(normalized_data_path, index=False)

        return normalized_data_path  # Return the path to the normalized data

    else:
        logging.error("No combined data for all sites")
        return None

if __name__ == "__main__":
    main()
