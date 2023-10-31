import os
import sys
import subprocess
from datetime import datetime, timedelta
import get_noaa_dict
import get_flow_dict
import pandas as pd

def main():
    # Get dates for the last 5 years
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    site_id = "09058000"  # Default, but you can modify as needed
    # TODO: get this from script
    latitude = 39.7392  # Default latitude (for Denver, CO as an example)
    longitude = -104.9903  # Default longitude

    # Call the NOAA script
    subprocess.run(['python', 'get_noaa_dict.py', str(latitude), str(longitude), start_date, end_date])

    # Call the get_flow script
    subprocess.run(['python', 'get_flow_dict.py', start_date, end_date, site_id])

    # Load both CSVs
    noaa_data = pd.read_csv('path_to_noaa_output.csv')
    flow_data = pd.read_csv('path_to_flow_output.csv')

    # Combine and save the data (You might need more advanced merging depending on your data format)
    combined_data = pd.concat([noaa_data, flow_data], axis=1)
    combined_data.to_csv('combined_data.csv', index=False)

if __name__ == "__main__":
    main()
