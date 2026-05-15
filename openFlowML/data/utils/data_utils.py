import logging
import os
import time
import requests
from datetime import datetime, timedelta, timezone

# NOTE: h5py, python-dotenv and earthaccess are heavy, optional dependencies
# used only by the soil-moisture data modules. They are imported lazily inside
# the functions that need them so that lightweight consumers (e.g. combine_data
# importing preview_data) work with only the core requirements installed.

# Global variables to store token and expiration
_token = None
_expiration = None

 # Additional function to display the beginning and ending of the dataframe
def preview_data(df, num_rows=4):
    logging.info("First few rows:")
    logging.info(df.head(num_rows))
    logging.info("\nLast few rows:")
    logging.info(df.tail(num_rows))


def request_with_retry(url, params=None, headers=None, timeout=60, retries=3, backoff=2):
    """
    GET a URL with a timeout and simple exponential backoff.

    Returns the Response on HTTP 200, or None if every attempt failed. The flow
    fetchers depend on this because their upstream services (USGS IV, CO DWR)
    are slow and occasionally flaky on long-range requests.
    """
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
            if response.status_code == 200:
                return response
            logging.warning("HTTP %s from %s (attempt %d/%d)",
                            response.status_code, url, attempt + 1, retries)
        except requests.exceptions.RequestException as e:
            logging.warning("Request error for %s: %s (attempt %d/%d)",
                            url, e, attempt + 1, retries)
        if attempt < retries - 1:
            time.sleep(backoff ** attempt)
    logging.error("Giving up on %s after %d attempts", url, retries)
    return None


def date_chunks(start_date, end_date, max_days=366):
    """
    Yield (chunk_start, chunk_end) datetime pairs covering [start_date, end_date]
    in windows of at most `max_days` days.

    Long-range requests to the USGS IV service are silently truncated, so the
    flow fetchers request one bounded chunk at a time and concatenate.
    """
    current = start_date
    while current <= end_date:
        chunk_end = min(current + timedelta(days=max_days - 1), end_date)
        yield current, chunk_end
        current = chunk_end + timedelta(days=1)

def get_earthdata_auth():
    """
    Create and return an authenticated earthaccess Auth instance.
    """
    from earthaccess import Auth
    auth = Auth()
    
    username = os.getenv("EARTHDATA_USERNAME")
    password = os.getenv("EARTHDATA_PASSWORD")
    
    logging.info(f"Attempting to authenticate with username: {username}")
    
    if username and password:
        if auth.login(strategy="environment"):
            logging.info("Successfully authenticated using environment variables")
            return auth
        else:
            logging.warning("Authentication failed using environment variables")
    else:
        logging.warning("Environment variables not set or empty")
    
    raise RuntimeError("Failed to authenticate with NASA Earthdata Login")


def appeears_login():
    """
    Log in to AppEEARS and obtain a token.
    """
    global _token, _expiration
    url = "https://appeears.earthdatacloud.nasa.gov/api/login"
    username = os.getenv("EARTHDATA_USERNAME")
    password = os.getenv("EARTHDATA_PASSWORD")

    if not username or not password:
        raise ValueError("EARTHDATA_USERNAME and EARTHDATA_PASSWORD environment variables must be set.")

    try:
        response = requests.post(url, auth=(username, password))
        response.raise_for_status()
        data = response.json()
        _token = data['token']
        # Parse the expiration date string manually
        expiration_str = data['expiration']
        # Remove the 'Z' at the end and split the string
        date_part, time_part = expiration_str[:-1].split('T')
        year, month, day = map(int, date_part.split('-'))
        hour, minute, second = map(int, time_part.split(':'))
        _expiration = datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
        return _token
    except requests.RequestException as e:
        print(f"Login failed: {e}")
        return None

def appeears_logout():
    """
    Log out from AppEEARS and invalidate the current token.
    """
    global _token, _expiration
    if not _token:
        print("No active session to log out from.")
        return True

    url = "https://appeears.earthdatacloud.nasa.gov/api/logout"
    headers = {'Authorization': f'Bearer {_token}'}
    try:
        response = requests.post(url, headers=headers)
        if response.status_code == 204:
            _token = None
            _expiration = None
            return True
        else:
            print(f"Logout failed: {response.text}")
            return False
    except requests.RequestException as e:
        print(f"Logout failed: {e}")
        return False

def get_smap_data_bounds(hdf_file):
    """
    Get the actual bounding box of the SMAP data from the HDF file, excluding fill values.
    """
    import h5py
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
    """
    Load credentials from creds.env into the environment.

    Returns True if all required variables are present, False otherwise. This
    function must never exit the process: importing a data module should not
    kill the interpreter just because credentials are absent (e.g. in CI or
    when running the test suite).
    """
    from dotenv import load_dotenv

    script_dir = os.path.dirname(os.path.abspath(__file__))
    cred_env_path = os.path.join(script_dir, 'creds.env')
    if os.path.exists(cred_env_path):
        load_dotenv(cred_env_path)
        logging.info(f"Loaded environment variables from {cred_env_path}")
    else:
        logging.warning(f"creds.env file not found at {cred_env_path}")

    required = ["EARTHDATA_USERNAME", "EARTHDATA_PASSWORD",
                "EROS_API_KEY", "EROS_USERNAME", "EROS_PASSWORD"]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        logging.warning(f"Missing required environment variables: {missing}")
        return False
    logging.info("environment variables are set")
    return True
