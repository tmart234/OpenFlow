import logging
import os
import h5py
from dotenv import load_dotenv

 # Additional function to display the beginning and ending of the dataframe
def preview_data(df, num_rows=4):
    logging.info("First few rows:")
    logging.info(df.head(num_rows))
    logging.info("\nLast few rows:")
    logging.info(df.tail(num_rows))

def get_smap_data_bounds(hdf_file):
    """
    Get the actual bounding box of the SMAP data from the HDF file, excluding fill values.
    """
    try:
        with h5py.File(hdf_file, 'r') as file:
            for time_of_day in ['AM', 'PM']:
                try:
                    lat_dataset = file[f'Soil_Moisture_Retrieval_Data_{time_of_day}/latitude']
                    lon_dataset = file[f'Soil_Moisture_Retrieval_Data_{time_of_day}/longitude']
                    
                    # Filter out fill values (assuming -9999.0 is the fill value)
                    valid_lat = lat_dataset[lat_dataset[:] != -9999.0]
                    valid_lon = lon_dataset[lon_dataset[:] != -9999.0]
                    
                    if len(valid_lat) > 0 and len(valid_lon) > 0:
                        min_lat = valid_lat.min()
                        max_lat = valid_lat.max()
                        min_lon = valid_lon.min()
                        max_lon = valid_lon.max()
                        
                        logging.info(f"Actual SMAP data bounds: Lon ({min_lon}, {max_lon}), Lat ({min_lat}, {max_lat})")
                        return (min_lon, min_lat, max_lon, max_lat)
                    else:
                        logging.warning(f"No valid data found for {time_of_day}")
                except KeyError:
                    continue
        logging.error("Could not find valid latitude or longitude data in the file")
        return None
    except Exception as e:
        logging.error(f"Error getting SMAP data bounds: {e}")
        return None

def load_vars():
    # Load environment variables from cred.env
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cred_env_path = os.path.join(script_dir, 'creds.env')
    if os.path.exists(cred_env_path):
        load_dotenv(cred_env_path)
        logging.info(f"Loaded environment variables from {cred_env_path}")
    else:
        logging.error(f"creds.env file not found at {cred_env_path}")
        exit(1)

    # Check if environment variables are set
    if not os.getenv("EARTHDATA_USERNAME") or not os.getenv("EARTHDATA_PASSWORD") \
        or not os.getenv("EROS_API_KEY") or not os.getenv("EROS_USERNAME") \
        or not os.getenv("EROS_USERNAME"):
        logging.error("environment variables are not set in the cred.env file")
        exit(1)
    else:
        logging.info("environment variables are set")
