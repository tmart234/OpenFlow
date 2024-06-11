import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import io
import h5py
import logging
import argparse

# Set up the plotting theme
sns.set_theme(style="darkgrid")
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def search_smap_data(start_date, end_date, bounding_box, bearer_token, short_name="SPL2SMP_E", version="006"):
    """
    Searches for SMAP data using the CMR API.
    """
    cmr_url = "https://cmr.earthdata.nasa.gov/search/granules.json"
    headers = {
        "Authorization": f"Bearer {bearer_token}"
    }
    params = {
        "short_name": short_name,
        "version": version,
        "temporal": f"{start_date}T00:00:00Z,{end_date}T23:59:59Z",
        "bounding_box": bounding_box,
        "page_size": 100,
        "page_num": 1
    }
    
    response = requests.get(cmr_url, headers=headers, params=params)
    logging.info(f"CMR URL: {response.url}")
    response.raise_for_status()
    return response.json()

def get_smap_soil_moisture_data(granule_url, bearer_token):
    """
    Retrieves soil moisture data from the SMAP granule URL.
    """
    headers = {
        "Authorization": f"Bearer {bearer_token}"
    }
    
    response = requests.get(granule_url, headers=headers)
    response.raise_for_status()
    return response.content

def debug_hdf5_structure(data):
    """
    Debug function to print the structure of the HDF5 file.
    """
    with h5py.File(io.BytesIO(data), 'r') as f:
        def print_structure(name, obj):
            logging.debug(name)
        f.visititems(print_structure)

def process_smap_data(data):
    """
    Processes the SMAP data from HDF5 format into a pandas DataFrame.
    """
    with h5py.File(io.BytesIO(data), 'r') as f:
        # Debug the structure of the HDF5 file
        logging.info("HDF5 file structure:")
        f.visititems(lambda name, obj: logging.info(name))
        
        # Extract data from Soil_Moisture_Retrieval_Data group
        soil_moisture_data = f['Soil_Moisture_Retrieval_Data']['soil_moisture_option3'][:]
        latitude_data = f['Soil_Moisture_Retrieval_Data']['latitude'][:]
        longitude_data = f['Soil_Moisture_Retrieval_Data']['longitude'][:]
        date_data = f['Soil_Moisture_Retrieval_Data']['tb_time_utc'][:].astype('U')
        
        # Attempt to extract quality flag data if it exists
        quality_flag_data = None
        if 'retrieval_qual_flag_option3' in f['Soil_Moisture_Retrieval_Data']:
            quality_flag_data = f['Soil_Moisture_Retrieval_Data']['retrieval_qual_flag_option3'][:]
        else:
            quality_flag_data = f['Soil_Moisture_Retrieval_Data']['retrieval_qual_flag'][:]
    
    df = pd.DataFrame({
        'date': date_data,
        'latitude': latitude_data,
        'longitude': longitude_data,
        'soil_moisture': soil_moisture_data,
        'quality_flag': quality_flag_data
    })
    
    # Filter out invalid soil moisture values and ensure recommended quality
    df = df[(df['soil_moisture'] != -9999.0) & (df['quality_flag'] & 0b1 == 0)]
    
    return df

def clean_and_convert_dates(date_series):
    """
    Cleans and converts a series of date strings to datetime objects, filtering out invalid formats.
    """
    def try_parsing_date(text):
        for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
        return None

    cleaned_dates = date_series.apply(try_parsing_date)
    return cleaned_dates.dropna()

def plot_soil_moisture(df):
    """
    Plots the soil moisture data.
    """
    df['date'] = clean_and_convert_dates(df['date'])
    
    if df.empty:
        logging.info("No valid soil moisture data available to plot.")
        return
    
    plt.figure(figsize=(14, 7))
    plt.scatter(df['date'], df['soil_moisture'], marker='o')
    plt.title('Daily Soil Moisture')
    plt.xlabel('Date')
    plt.ylabel('Soil Moisture')
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.gcf().autofmt_xdate()
    plt.show()

def get_smap_timeseries(bearer_token, start_date, end_date, latitude, longitude):
    """
    Main function to retrieve and process SMAP soil moisture data.
    """
    bounding_box = f"{longitude-0.1},{latitude-0.1},{longitude+0.1},{latitude+0.1}"
    
    logging.info(f"Bounding Box: {bounding_box}")
    logging.info(f"Start Date: {start_date}")
    logging.info(f"End Date: {end_date}")

    search_results = search_smap_data(start_date, end_date, bounding_box, bearer_token)

    # Check if any granules are found
    if 'feed' in search_results and 'entry' in search_results['feed'] and len(search_results['feed']['entry']) > 0:
        granule_url = search_results['feed']['entry'][0]['links'][0]['href']
        logging.info(f"Granule URL: {granule_url}")

        # Get soil moisture data from the first granule (for simplicity)
        soil_moisture_data = get_smap_soil_moisture_data(granule_url, bearer_token)

        # Debug the HDF5 file structure
        debug_hdf5_structure(soil_moisture_data)

        # Process the data
        df_soil_moisture = process_smap_data(soil_moisture_data)
        logging.info(df_soil_moisture.head())

        # Plot the data
        plot_soil_moisture(df_soil_moisture)
        
        return df_soil_moisture
    else:
        logging.warning("No granules found for the given search criteria.")
        return pd.DataFrame()

def main(bearer_token, start_date, end_date, latitude, longitude):
    df_soil_moisture = get_smap_timeseries(bearer_token, start_date, end_date, latitude, longitude)
    if not df_soil_moisture.empty:
        logging.info("Soil moisture data retrieved successfully.")
    else:
        logging.warning("Failed to retrieve soil moisture data.")
    return df_soil_moisture

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetch SMAP soil moisture data.')
    parser.add_argument('--bearer_token', type=str, required=True, help='Bearer token for authentication')
    parser.add_argument('--start_date', type=str, required=True, help='Start date in the format YYYY-MM-DD')
    parser.add_argument('--end_date', type=str, required=True, help='End date in the format YYYY-MM-DD')
    parser.add_argument('--latitude', type=float, required=True, help='Latitude of the location')
    parser.add_argument('--longitude', type=float, required=True, help='Longitude of the location')
    args = parser.parse_args()

    main(args.bearer_token, args.start_date, args.end_date, args.latitude, args.longitude)