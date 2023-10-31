import os
import sys
import subprocess
from datetime import datetime, timedelta
import get_noaa_dict
import get_flow_dict  # You might want to import the relevant function from this module
import pandas as pd

def main():
    # Get dates for the last 5 years
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    site_id = "09058000"  # Default, but you can modify as needed
    
    # TODO: get this from script
    latitude = 39.7392  # Default latitude (for Denver, CO as an example)
    longitude = -104.9903  # Default longitude

    # Call the NOAA script with the correct path
    noaa_script_path = '/home/runner/work/OpenFlowColorado/openFlowML/get_noaa_dict.py'
    subprocess.run(['python', noaa_script_path, str(latitude), str(longitude), start_date, end_date])
    
    # Call the get_flow function directly instead of subprocess
    flow_dict = get_flow_dict.get_daily_flow_data(site_id, start_date, end_date)  # Using the correct function from your get_flow_dict module
    flow_data = pd.DataFrame.from_dict(flow_dict, orient='index')
    flow_data.reset_index(inplace=True)
    flow_data.rename(columns={'index': 'Date'}, inplace=True)

    # Load NOAA CSV
    noaa_data = pd.read_csv('noaa_output.csv')  # Assuming this is the filename, adjust as necessary

    # Ensure the dates are the index for both dataframes for proper alignment
    flow_data.set_index('Date', inplace=True)
    noaa_data.set_index('Date', inplace=True)

    # Combine and save the data
    combined_data = pd.concat([noaa_data, flow_data], axis=1)
    combined_data.reset_index(inplace=True)  # If you want 'Date' to be a column again and not an index
    combined_data.to_csv('combined_data.csv', index=False)

if __name__ == "__main__":
    main()
