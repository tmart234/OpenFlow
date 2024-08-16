import datetime
import numpy as np
import h5py
from shapely.geometry import Polygon, Point
import logging
import argparse
from io import BytesIO
from earthaccess import *
from dataUtils.get_poly import get_huc8_polygon, simplify_polygon
from dataUtils.data_utils import load_earthdata_vars
import os
import tempfile
import requests
# Conditionally import matplotlib
import importlib.util
matplotlib_spec = importlib.util.find_spec("matplotlib")
matplotlib_available = matplotlib_spec is not None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_earthdata_vars()

def get_earthdata_auth():
    """
    Create and return an authenticated earthaccess Auth instance.
    """
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
    
    # If environment variables fail, try .netrc file
    if auth.login(strategy="netrc"):
        logging.info("Successfully authenticated using .netrc file")
        return auth
    
    # If both methods fail, fall back to interactive login
    if auth.login(strategy="interactive"):
        logging.info("Successfully authenticated interactively")
        return auth
    
    raise RuntimeError("Failed to authenticate with NASA Earthdata Login")

def search_and_download_smap_data(start_date, end_date, auth):
    """
    Search for SMAP L3 data between the given dates and download it to a temporary directory.
    """
    try:
        # Check if we're already logged in
        if not auth.authenticated:
            logging.info("Not logged in, attempting to log in...")
            if not auth.login(strategy="environment"):  # Explicitly use environment strategy
                raise RuntimeError("Failed to authenticate with NASA Earthdata Login")
        else:
            logging.info("Already authenticated, proceeding with search and download")

        # Create DataCollections instance
        collections = DataCollections(auth)
        
        # Search for SMAP L3 soil moisture collections
        smap_collections = collections.keyword("SMAP L3 Soil Moisture").get()
        
        if not smap_collections:
            logging.error("No SMAP L3 Soil Moisture collections found")
            return None

        # Use the first collection found
        collection = smap_collections[0]
        
        # Log collection details using available attributes
        logging.info(f"Using collection: {collection.concept_id()}")

        # Create DataGranules instance
        granules = DataGranules(auth)

        # Search for granules using the collection's concept_id
        granule_query = (granules
                         .concept_id(collection.concept_id())
                         .temporal(start_date, end_date))
        
        # Get all granules
        all_granules = granule_query.get_all()
        
        if not all_granules:
            logging.error(f"No SMAP data found from {start_date} to {end_date}")
            return None
        
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            logging.info(f"Created temporary directory: {temp_dir}")
            
            # Download the data
            try:
                downloaded_files = download(all_granules, local_path=temp_dir)
            except Exception as e:
                logging.error(f"Error during download: {str(e)}")
                return None
            
            if downloaded_files:
                logging.info(f"Successfully downloaded: {downloaded_files}")
                return downloaded_files
            else:
                logging.error("Failed to download SMAP data")
                return None
    
    except Exception as e:
        logging.error(f"Error searching or downloading SMAP data: {e}")
        logging.error("Traceback: ", exc_info=True)
        return None

def extract_soil_moisture(hdf_file, polygon):
    """
    Extract soil moisture data for the given polygon from the HDF file.
    """
    try:
        with h5py.File(hdf_file, 'r') as file:
            soil_moisture = file['Soil_Moisture_Retrieval_Data']['soil_moisture'][:]
            lat = file['cell_lat'][:]
            lon = file['cell_lon'][:]
            
            shape = Polygon(polygon)
            mask = np.zeros_like(soil_moisture, dtype=bool)
            
            for i in range(soil_moisture.shape[0]):
                for j in range(soil_moisture.shape[1]):
                    if shape.contains(Point(lon[j], lat[i])):
                        mask[i, j] = True
            
            valid_data = soil_moisture[mask & (soil_moisture != -9999)]
            if len(valid_data) > 0:
                return np.mean(valid_data), soil_moisture, lat, lon, mask
            else:
                logging.warning("No valid soil moisture data found within the polygon")
                return None, None, None, None, None
    except Exception as e:
        logging.error(f"Error extracting soil moisture data: {e}")
        return None, None, None, None, None

def visualize_soil_moisture(polygon, soil_moisture, lat, lon, mask, average_moisture):
    if not matplotlib_available:
        logging.error("Matplotlib is not installed. Cannot create visualization.")
        return

    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as mplPolygon
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the entire soil moisture data
    im = ax.imshow(soil_moisture, cmap='YlGnBu', extent=[lon.min(), lon.max(), lat.min(), lat.max()], 
                   origin='lower', alpha=0.5)
    
    # Highlight the polygon
    poly = mplPolygon(polygon, facecolor='none', edgecolor='red', linewidth=2)
    ax.add_patch(poly)
    
    # Set the extent to focus on the polygon
    min_lon, min_lat = np.min(polygon, axis=0)
    max_lon, max_lat = np.max(polygon, axis=0)
    ax.set_xlim(min_lon - 1, max_lon + 1)
    ax.set_ylim(min_lat - 1, max_lat + 1)
    
    plt.colorbar(im, label='Soil Moisture')
    plt.title(f'Soil Moisture Map\nAverage within polygon: {average_moisture:.4f}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    plt.show()

def main(start_date, end_date, lat, lon, visual):
    auth = get_earthdata_auth()
    # The function will raise an exception if authentication fails, so we don't need to check if auth is None

    # Get and simplify the HUC8 polygon
    huc8_polygon = get_huc8_polygon(lat, lon)
    if not huc8_polygon:
        logging.error("Failed to retrieve HUC8 polygon")
        return

    simplified_polygon = simplify_polygon(huc8_polygon)

    temp_file_path = search_and_download_smap_data(start_date, end_date, auth)
    if temp_file_path:
        soil_moisture, full_soil_moisture, lat_data, lon_data, mask = extract_soil_moisture(temp_file_path, simplified_polygon)
        if soil_moisture is not None:
            logging.info(f"Average soil moisture for the given polygon between {start_date} and {end_date}: {soil_moisture:.4f}")
            
            if visual:
                if matplotlib_available:
                    visualize_soil_moisture(simplified_polygon, full_soil_moisture, lat_data, lon_data, mask, soil_moisture)
                else:
                    logging.warning("Matplotlib is not installed. Skipping visualization.")
        else:
            logging.error("Failed to calculate soil moisture")
    else:
        logging.error("Failed to find or download SMAP data")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate average soil moisture for a HUC8 polygon from SMAP L3 data.')
    parser.add_argument('--start-date', type=lambda d: datetime.datetime.strptime(d, '%Y-%m-%d').date(), required=True, help='Start Date in YYYY-MM-DD format')
    parser.add_argument('--end-date', type=lambda d: datetime.datetime.strptime(d, '%Y-%m-%d').date(), required=True, help='End Date in YYYY-MM-DD format')
    parser.add_argument('--lat', type=float, required=True, help='Latitude of the point within the desired HUC8 polygon')
    parser.add_argument('--lon', type=float, required=True, help='Longitude of the point within the desired HUC8 polygon')
    parser.add_argument('--visual', action='store_true', help='Enable matplotlib visualization')
    args = parser.parse_args()

    main(args.start_date, args.end_date, args.lat, args.lon, args.visual)