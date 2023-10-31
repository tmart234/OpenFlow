import os
import sys
import subprocess
from datetime import datetime, timedelta

def main():
    # Get dates for the last 5 years
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    
    site_id = "09058000"  # Default, but you can modify as needed

    # Call the NOAA script
    subprocess.run(['python', 'noaa.py', start_date, end_date, site_id])
    
    # Call the get_flow script
    subprocess.run(['python', 'get_flow.py', start_date, end_date, site_id])

    # Load both CSVs
    noaa_data = pd.read_csv('path_to_noaa_output.csv')
    flow_data = pd.read_csv('path_to_flow_output.csv')

    # Combine and save the data (You might need more advanced merging depending on your data format)
    combined_data = pd.concat([noaa_data, flow_data], axis=1)
    combined_data.to_csv('combined_data.csv', index=False)

if __name__ == "__main__":
    main()
