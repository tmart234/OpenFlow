import os
import subprocess
from datetime import datetime, timedelta
import get_flow_dict  # You might want to import the relevant function from this module
import pandas as pd
from get_coordinates import get_coordinates

""" 
Takes multiuple individual data components and combindes into a dataset
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
            site_id = "09163500"  # Default, but you can modify as needed
            
            # Fetch coords
            coords_dict = get_coordinates(site_id)
            latitude = coords_dict['latitude']
            longitude = coords_dict['longitude']

            # Call the NOAA script with the correct path
            noaa_script_path = os.path.join(base_path, 'openFlowML', 'get_noaa_dict.py')
            subprocess.run(['python', noaa_script_path, latitude, longitude, start_date, end_date, site_id], check=True)
            
            # Call the get_flow function directly instead of subprocess
            flow_dict = get_flow_dict.get_daily_flow_data(site_id, start_date, end_date)  # Using the correct function from your get_flow_dict module
            flow_data = pd.DataFrame.from_dict(flow_dict, orient='index')
            flow_data.reset_index(inplace=True)
            flow_data.rename(columns={'index': 'Date'}, inplace=True)

            # Load NOAA CSV with custom date parser
            noaa_data = pd.read_csv(f"{site_id}_noaa_data.csv", parse_dates=['Date'], date_parser=parse_date)  # Adjust the column name 'Date' if it's different in your CSV

            # Ensure the dates are the index for both dataframes for proper alignment
            flow_data.set_index('Date', inplace=True)
            noaa_data.set_index('Date', inplace=True)

            # Combine and save the data for the current site_id
            combined_data = pd.concat([noaa_data, flow_data], axis=1)
            combined_data.reset_index(inplace=True)  # If you want 'Date' to be a column again and not an index
            all_data.append(combined_data)
        except Exception as e:
            # Log error
            print(f"An error occurred for site ID {site_id}: {e}")

    # Combine data from all site ids and save
    if all_data:
        final_data = pd.concat(all_data)
        final_data.to_csv('combined_data_all_sites.csv', index=False)

if __name__ == "__main__":
    main()
