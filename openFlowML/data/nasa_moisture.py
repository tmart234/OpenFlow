import datetime
import numpy as np
import h5py
import pytz
from shapely.geometry import Polygon, Point, box
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

import earthaccess

def search_and_download_smap_data(start_date, end_date, auth, simplified_polygon):
    """
    Search for SMAP L3 data between the given dates that intersect with the given polygon,
    and download the smallest intersecting granule.
    """
    try:
        # Ensure we're authenticated
        if not auth.authenticated:
            logging.info("Not logged in, attempting to log in...")
            if not auth.login(strategy="environment"):
                raise RuntimeError("Failed to authenticate with NASA Earthdata Login")
        else:
            logging.info("Already authenticated, proceeding with search and download")

        # Search for SPL3SMP_E collection
        collection_query = earthaccess.DataCollections().short_name("SPL3SMP_E").version("006")
        collections = collection_query.get()

        if not collections:
            logging.error("SMAP L3 SM_P_E collection not found")
            return None

        collection = collections[0]
        concept_id = collection.concept_id()
        logging.info(f"Found SMAP_L3_SM_P_E collection with concept_id: {concept_id}")


        # Calculate bounding box from simplified_polygon
        lons, lats = zip(*simplified_polygon)
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)

        # Now search for granules using DataGranules
        granule_query = (earthaccess.DataGranules()
                         .concept_id(concept_id) 
                         .temporal(start_date, end_date)
                         .bounding_box(min_lon, min_lat, max_lon, max_lat))
                
        granule_hits = granule_query.hits()
        logging.info(f"Number of granules found: {granule_hits}")

        if granule_hits == 0:
            logging.warning(f"No SMAP data found from {start_date} to {end_date}")
            return None

        granules = granule_query.get_all()

        if not granules:
            logging.warning(f"No granules retrieved despite positive hit count")
            return None

        # Log the number of granules found
        logging.info(f"Retrieved {len(granules)} granules")

        # Find the smallest granule
        smallest_granule = min(granules, key=lambda g: g.size())
        logging.info(f"Smallest intersecting granule size: {smallest_granule.size()} MB")

        # Create a temporary directory that won't be automatically deleted
        temp_dir = tempfile.mkdtemp()
        logging.info(f"Created temporary directory: {temp_dir}")

        # Download the smallest granule
        try:
            downloaded_files = earthaccess.download(smallest_granule, local_path=temp_dir)
        except Exception as e:
            logging.error(f"Error during download: {str(e)}")
            return None

        if downloaded_files:
            downloaded_file = downloaded_files[0]
            logging.info(f"Successfully downloaded: {downloaded_file}")
            
            # Verify that the file exists
            if os.path.exists(downloaded_file):
                logging.info(f"File exists at {downloaded_file}")
                file_size = os.path.getsize(downloaded_file)
                logging.info(f"File size: {file_size} bytes")
            else:
                logging.error(f"File does not exist at {downloaded_file}")
            
            return downloaded_file
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
        logging.info(f"Attempting to open file: {hdf_file}")
        
        if not os.path.exists(hdf_file):
            logging.error(f"File does not exist: {hdf_file}")
            return None, None, None, None, None

        with h5py.File(hdf_file, 'r') as file:
            logging.info("Successfully opened HDF5 file")
            
            # List all groups in the file
            logging.info("Groups in the HDF5 file:")
            file.visit(lambda name: logging.info(name))

            # Attempt to access the soil moisture dataset
            try:
                soil_moisture = file['Soil_Moisture_Retrieval_Data']['soil_moisture'][:]
                logging.info(f"Soil moisture data shape: {soil_moisture.shape}")
            except KeyError:
                logging.error("Could not find 'Soil_Moisture_Retrieval_Data/soil_moisture' in the file")
                return None, None, None, None, None

            # Attempt to access latitude and longitude datasets
            try:
                lat = file['cell_lat'][:]
                lon = file['cell_lon'][:]
                logging.info(f"Latitude data shape: {lat.shape}")
                logging.info(f"Longitude data shape: {lon.shape}")
            except KeyError:
                logging.error("Could not find 'cell_lat' or 'cell_lon' in the file")
                return None, None, None, None, None
    except Exception as e:
        logging.error(f"Error extracting soil moisture data: {e}")
        return None, None, None, None, None

def list_nsidc_collections():
    try:
        nsidc_query = earthaccess.collection_query().daac("NSIDC")
        collections = nsidc_query.get()
        
        logging.info(f"Found {len(collections)} collections from NSIDC-DAAC:")
        for collection in collections:
            logging.info(f"- Short Name: {collection['umm']['ShortName']}, Version: {collection['umm'].get('Version', 'N/A')}")
        
        return collections
    except Exception as e:
        logging.error(f"Error listing NSIDC collections: {e}")
        return None

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

    #list_nsidc_collections()

    # Get the HUC8 polygon
    huc8_polygon = get_huc8_polygon(lat, lon)
    if not huc8_polygon:
        logging.error("Failed to retrieve HUC8 polygon")
        return

    # Simplify the polygon and format it for earthaccess
    simplified_polygon = simplify_polygon(huc8_polygon)
    logging.debug(f"Simplified polygon: {simplified_polygon}")

    # Pass the simplified_polygon to the search_and_download_smap_data function
    downloaded_file = search_and_download_smap_data(start_date, end_date, auth, simplified_polygon)
    
    if downloaded_file:
        soil_moisture, full_soil_moisture, lat_data, lon_data, mask = extract_soil_moisture(downloaded_file, simplified_polygon)
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