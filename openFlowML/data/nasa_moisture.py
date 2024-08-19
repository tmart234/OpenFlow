import datetime
import numpy as np
import h5py
import shutil
from shapely.ops import transform
from shapely.geometry import Polygon, Point
import logging
import argparse
from earthaccess import *
from dataUtils.get_poly import check_polygon_intersection, get_huc8_polygon, validate_polygon, simplify_polygon
from dataUtils.data_utils import load_vars, get_smap_data_bounds
import os
import tempfile
# Conditionally import matplotlib
import importlib.util
from shapely.ops import transform
import pyproj
matplotlib_spec = importlib.util.find_spec("matplotlib")
matplotlib_available = matplotlib_spec is not None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_vars()


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

        # Download only the smallest granule
        try:
            downloaded_files = earthaccess.download(smallest_granule, local_path=temp_dir)
        except Exception as e:
            logging.error(f"Error during download: {str(e)}")
            if temp_dir:
                shutil.rmtree(temp_dir)
            return None, None

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
                if temp_dir:
                    shutil.rmtree(temp_dir)
                return None, None
            
            return downloaded_file, temp_dir
        else:
            logging.error("Failed to download SMAP data")
            if temp_dir:
                shutil.rmtree(temp_dir)
            return None, None

    except Exception as e:
        logging.error(f"Error searching or downloading SMAP data: {e}")
        logging.error("Traceback: ", exc_info=True)
        if temp_dir:
            shutil.rmtree(temp_dir)
        return None, None

def extract_soil_moisture(hdf_file, polygon):
    """
    Extract soil moisture data for the given polygon from the HDF file.
    """
    try:
        logging.info(f"Attempting to open file: {hdf_file}")
        
        if not os.path.exists(hdf_file):
            logging.error(f"File does not exist: {hdf_file}")
            return None

        with h5py.File(hdf_file, 'r') as file:
            logging.info("Successfully opened HDF5 file")
            
            # Attempt to access the soil moisture dataset (try both AM and PM)
            for time_of_day in ['AM', 'PM']:
                try:
                    soil_moisture_dataset = file[f'Soil_Moisture_Retrieval_Data_{time_of_day}/soil_moisture']
                    lat_dataset = file[f'Soil_Moisture_Retrieval_Data_{time_of_day}/latitude']
                    lon_dataset = file[f'Soil_Moisture_Retrieval_Data_{time_of_day}/longitude']
                    logging.info(f"Found soil moisture data for {time_of_day}")
                    break
                except KeyError:
                    logging.warning(f"Could not find soil moisture data for {time_of_day}")
            else:
                logging.error("Could not find soil moisture, latitude, or longitude data in the file")
                return None

            # Get dataset shapes and create a polygon object
            shape = soil_moisture_dataset.shape
            polygon_obj = Polygon(polygon)

            # Log the resolution of the data
            lat_res = (lat_dataset[:].max() - lat_dataset[:].min()) / shape[0]
            lon_res = (lon_dataset[:].max() - lon_dataset[:].min()) / shape[1]
            logging.info(f"SMAP data resolution: {lat_res:.6f} degrees latitude, {lon_res:.6f} degrees longitude")

            # Calculate the approximate size of the polygon
            poly_bounds = polygon_obj.bounds
            poly_width = poly_bounds[2] - poly_bounds[0]
            poly_height = poly_bounds[3] - poly_bounds[1]
            logging.info(f"Polygon size: {poly_width:.6f} degrees longitude, {poly_height:.6f} degrees latitude")

            # Create a transformer object for converting between coordinate systems
            transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

            # Transform the polygon to the same coordinate system as the data
            polygon_transformed = transform(transformer.transform, polygon_obj)

            # Initialize variables for calculation
            total_moisture = 0
            count = 0

            # Process data in chunks to reduce memory usage
            chunk_size = 1000
            for i in range(0, shape[0], chunk_size):
                for j in range(0, shape[1], chunk_size):
                    # Load data in chunks
                    lat_chunk = lat_dataset[i:i+chunk_size, j:j+chunk_size]
                    lon_chunk = lon_dataset[i:i+chunk_size, j:j+chunk_size]
                    soil_moisture_chunk = soil_moisture_dataset[i:i+chunk_size, j:j+chunk_size]

                    # Create masks for valid data
                    valid_mask = (lat_chunk != -9999.0) & (lon_chunk != -9999.0) & (soil_moisture_chunk != -9999.0)
                    
                    # Apply masks
                    lat_valid = lat_chunk[valid_mask]
                    lon_valid = lon_chunk[valid_mask]
                    soil_moisture_valid = soil_moisture_chunk[valid_mask]

                    # Log the number of valid points in this chunk
                    logging.info(f"Valid points in chunk {i}/{shape[0]}, {j}/{shape[1]}: {np.sum(valid_mask)}")

                    # Check points against the polygon
                    for lat, lon, moisture in zip(lat_valid.flat, lon_valid.flat, soil_moisture_valid.flat):
                        point = Point(transformer.transform(lon, lat))
                        if polygon_transformed.contains(point):
                            total_moisture += moisture
                            count += 1

                    # Log progress and intermediate results
                    if count > 0:
                        logging.info(f"Processed chunk {i}/{shape[0]}, {j}/{shape[1]}. Points in polygon so far: {count}")

            if count > 0:
                average_moisture = total_moisture / count
                logging.info(f"Number of points found inside the polygon: {count}")
                logging.info(f"Average soil moisture: {average_moisture:.4f}")
                return average_moisture
            else:
                logging.warning("No valid soil moisture data found within the polygon")
                return None

    except Exception as e:
        logging.error(f"Error extracting soil moisture data: {e}")
        logging.error("Traceback: ", exc_info=True)
        return None

# Add this function to check data availability in the polygon area
def check_data_availability(hdf_file, polygon):
    try:
        with h5py.File(hdf_file, 'r') as file:
            for time_of_day in ['AM', 'PM']:
                try:
                    soil_moisture = file[f'Soil_Moisture_Retrieval_Data_{time_of_day}/soil_moisture'][:]
                    lat = file[f'Soil_Moisture_Retrieval_Data_{time_of_day}/latitude'][:]
                    lon = file[f'Soil_Moisture_Retrieval_Data_{time_of_day}/longitude'][:]
                    break
                except KeyError:
                    continue
            else:
                logging.error("Could not find soil moisture data")
                return

        # Get polygon bounds
        min_lon, min_lat, max_lon, max_lat = Polygon(polygon).bounds

        # Create a mask for the area of interest
        area_mask = (lat >= min_lat) & (lat <= max_lat) & (lon >= min_lon) & (lon <= max_lon)

        # Check for valid data in the area of interest
        valid_data_mask = (soil_moisture != -9999.0) & area_mask

        total_points = np.sum(area_mask)
        valid_points = np.sum(valid_data_mask)

        logging.info(f"Total points in area of interest: {total_points}")
        logging.info(f"Valid data points in area of interest: {valid_points}")
        logging.info(f"Percentage of valid data: {valid_points/total_points*100:.2f}%")

    except Exception as e:
        logging.error(f"Error checking data availability: {e}")

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

def visualize_soil_moisture_simple(polygon, average_moisture):
    """
    Create a simple visualization of the polygon and average soil moisture.
    """
    if not matplotlib_available:
        logging.error("Matplotlib is not installed. Cannot create visualization.")
        return

    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as mplPolygon
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the polygon
    poly = mplPolygon(polygon, facecolor='none', edgecolor='red', linewidth=2)
    ax.add_patch(poly)
    
    # Set the extent to focus on the polygon
    min_lon, min_lat = np.min(polygon, axis=0)
    max_lon, max_lat = np.max(polygon, axis=0)
    ax.set_xlim(min_lon - 0.1, max_lon + 0.1)
    ax.set_ylim(min_lat - 0.1, max_lat + 0.1)
    
    plt.title(f'HUC8 Polygon\nAverage Soil Moisture: {average_moisture:.4f}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    plt.show()

def main(start_date, end_date, lat, lon, visual):
    auth = get_earthdata_auth()

    # Get the HUC8 polygon
    huc8_polygon = get_huc8_polygon(lat, lon)
    if not huc8_polygon:
        logging.error("Failed to retrieve HUC8 polygon")
        return

    # Simplify & validate the polygon and format it for earthaccess
    simplified_polygon = simplify_polygon(huc8_polygon)
    logging.debug(f"Simplified polygon: {simplified_polygon}")
    simplified_polygon = validate_polygon(simplified_polygon)
    logging.info(f"Validated polygon coordinates: {simplified_polygon}")

    # Calculate and log the bounding box of the polygon
    lons, lats = zip(*simplified_polygon)
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    logging.info(f"Polygon bounding box: Lon ({min_lon}, {max_lon}), Lat ({min_lat}, {max_lat})")

    # Pass the simplified_polygon to the search_and_download_smap_data function
    downloaded_file, temp_dir = search_and_download_smap_data(start_date, end_date, auth, simplified_polygon)
    
    if downloaded_file and temp_dir:
        try:
            # Extract soil moisture data
            average_soil_moisture = extract_soil_moisture(downloaded_file, simplified_polygon)
            
            if average_soil_moisture is not None:
                logging.info(f"Average soil moisture for the given polygon between {start_date} and {end_date}: {average_soil_moisture:.4f}")
                
                if visual:
                    if matplotlib_available:
                        visualize_soil_moisture_simple(simplified_polygon, average_soil_moisture)
                    else:
                        logging.warning("Matplotlib is not installed. Skipping visualization.")
            else:
                logging.error("Failed to calculate soil moisture. Check if the polygon intersects with available data.")
                
            # Additional debugging: Check SMAP data bounds
            smap_bounds = get_smap_data_bounds(downloaded_file)
            if smap_bounds:
                logging.info(f"SMAP data bounds: Lon ({smap_bounds[0]}, {smap_bounds[2]}), Lat ({smap_bounds[1]}, {smap_bounds[3]})")
                if check_polygon_intersection(simplified_polygon, smap_bounds):
                    logging.info("Polygon intersects with SMAP data bounds.")
                else:
                    logging.warning("Polygon does not intersect with SMAP data bounds. This may explain the lack of data.")
            
        finally:
            # Clean up: remove the temporary directory
            try:
                shutil.rmtree(temp_dir)
                logging.info(f"Removed temporary directory: {temp_dir}")
            except Exception as e:
                logging.warning(f"Failed to remove temporary directory: {e}")
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