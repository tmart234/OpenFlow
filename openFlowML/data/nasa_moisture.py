import datetime
import numpy as np
import h5py
import shutil
from scipy.spatial import cKDTree
from shapely.ops import transform
from shapely.geometry import Polygon, Point
import logging
import argparse
from earthaccess import *
from dataUtils.get_poly import check_polygon_intersection, get_huc_polygon, validate_polygon, simplify_polygon
from dataUtils.data_utils import load_vars, get_earthdata_auth, get_smap_data_bounds
import os
import tempfile
# Conditionally import matplotlib
import importlib.util
from shapely.ops import transform
import earthaccess
import pyproj
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as mplPolygon
matplotlib_spec = importlib.util.find_spec("matplotlib")
matplotlib_available = matplotlib_spec is not None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_vars()

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

def extract_soil_moisture(hdf_file, polygon, max_distance=0.1):
    try:
        with h5py.File(hdf_file, 'r') as file:
            for dataset_name in file:
                if 'Soil_Moisture_Retrieval_Data' in dataset_name:
                    soil_moisture = file[f'{dataset_name}/soil_moisture'][:]
                    lat = file[f'{dataset_name}/latitude'][:]
                    lon = file[f'{dataset_name}/longitude'][:]
                    logging.info(f"Found soil moisture data in {dataset_name}")
                    break
            else:
                logging.error("Could not find soil moisture data in the file")
                return None

        polygon_obj = Polygon(polygon)
        minx, miny, maxx, maxy = polygon_obj.bounds

        # Start with the polygon bounds and gradually expand
        for distance in np.linspace(0, max_distance, 5):
            expanded_bounds = (minx-distance, miny-distance, maxx+distance, maxy+distance)
            mask = (lon >= expanded_bounds[0]) & (lon <= expanded_bounds[2]) & \
                   (lat >= expanded_bounds[1]) & (lat <= expanded_bounds[3])
            
            lons = lon[mask]
            lats = lat[mask]
            soil_moisture_masked = soil_moisture[mask]

            valid = (soil_moisture_masked != -9999.0) & (lons != -9999.0) & (lats != -9999.0)
            lons = lons[valid]
            lats = lats[valid]
            soil_moisture_valid = soil_moisture_masked[valid]

            logging.info(f"Found {len(lons)} valid points within expanded bounds (distance: {distance})")

            if len(lons) > 0:
                tree = cKDTree(np.column_stack((lons, lats)))
                expanded_polygon = polygon_obj.buffer(distance)
                mask_polygon = tree.query_ball_point(expanded_polygon.exterior.coords, r=0.01)
                mask_polygon = np.unique(np.concatenate(mask_polygon))

                soil_moisture_in_polygon = soil_moisture_valid[mask_polygon]

                if len(soil_moisture_in_polygon) > 0:
                    average_moisture = np.mean(soil_moisture_in_polygon)
                    logging.info(f"Found {len(soil_moisture_in_polygon)} points inside or near the polygon")
                    logging.info(f"Average soil moisture: {average_moisture:.4f}")
                    return average_moisture, expanded_polygon

        logging.warning("No valid soil moisture data found within or near the polygon")
        return None, None

    except Exception as e:
        logging.error(f"Error extracting soil moisture data: {e}")
        logging.error("Traceback: ", exc_info=True)
        return None, None

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
        poly = Polygon(polygon)
        min_lon, min_lat, max_lon, max_lat = poly.bounds

        # Create masks for the area of interest and its surroundings
        area_mask = (lat >= min_lat) & (lat <= max_lat) & (lon >= min_lon) & (lon <= max_lon)
        surrounding_mask = (lat >= min_lat-1) & (lat <= max_lat+1) & (lon >= min_lon-1) & (lon <= max_lon+1)

        # Check for valid data
        valid_data_mask = (soil_moisture != -9999.0)

        # Calculate statistics
        total_points = np.sum(area_mask)
        valid_points = np.sum(valid_data_mask & area_mask)
        surrounding_valid_points = np.sum(valid_data_mask & surrounding_mask)

        logging.info(f"Total points in area of interest: {total_points}")
        logging.info(f"Valid data points in area of interest: {valid_points}")
        logging.info(f"Percentage of valid data in area: {valid_points/total_points*100:.2f}%")
        logging.info(f"Valid data points in surrounding area: {surrounding_valid_points}")

        if valid_points == 0:
            logging.warning("No valid data points found within the polygon.")
            if surrounding_valid_points > 0:
                logging.info("However, valid data points found in the surrounding area.")
            else:
                logging.warning("No valid data points found in the surrounding area either.")

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

def visualize_smap_and_polygon(hdf_file, polygon):
    try:
        with h5py.File(hdf_file, 'r') as file:
            for dataset_name in file:
                if 'Soil_Moisture_Retrieval_Data' in dataset_name:
                    soil_moisture = file[f'{dataset_name}/soil_moisture'][:]
                    lat = file[f'{dataset_name}/latitude'][:]
                    lon = file[f'{dataset_name}/longitude'][:]
                    break
            else:
                logging.error("Could not find soil moisture data in the file")
                return

        # Create a mask for valid data
        valid_mask = (soil_moisture != -9999.0) & (lat != -9999.0) & (lon != -9999.0)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot SMAP data points
        sc = ax.scatter(lon[valid_mask], lat[valid_mask], c=soil_moisture[valid_mask], 
                        cmap='viridis', s=1, alpha=0.5)
        plt.colorbar(sc, label='Soil Moisture')
        
        # Plot the polygon
        poly = mplPolygon(polygon, facecolor='none', edgecolor='red', linewidth=2)
        ax.add_patch(poly)
        
        # Set the extent to focus on the area around the polygon
        poly_bounds = Polygon(polygon).bounds
        ax.set_xlim(poly_bounds[0] - 1, poly_bounds[2] + 1)
        ax.set_ylim(poly_bounds[1] - 1, poly_bounds[3] + 1)
        
        plt.title('SMAP Data and Polygon')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.show()
        
    except Exception as e:
        logging.error(f"Error visualizing SMAP data and polygon: {e}")

def main(start_date, end_date, lat, lon, visual):
    auth = get_earthdata_auth()

    huc8_polygon = get_huc_polygon(lat, lon, huc_level=8)
    if not huc8_polygon:
        logging.error("Failed to retrieve HUC8 polygon")
        return

    simplified_polygon = simplify_polygon(huc8_polygon)
    logging.info(f"Simplified polygon coordinates: {simplified_polygon}")

    downloaded_file, temp_dir = search_and_download_smap_data(start_date, end_date, auth, simplified_polygon)
    
    if downloaded_file and temp_dir:
        try:
            # Visualize SMAP data and polygon
            visualize_smap_and_polygon(downloaded_file, simplified_polygon)
            
            # Extract soil moisture data
            average_soil_moisture, used_polygon = extract_soil_moisture(downloaded_file, simplified_polygon)
            
            if average_soil_moisture is not None:
                logging.info(f"Average soil moisture: {average_soil_moisture:.4f}")
                if visual and matplotlib_available:
                    pass
                    #visualize_soil_moisture_simple(used_polygon, average_soil_moisture)
            else:
                logging.error("Failed to calculate soil moisture.")
            
        finally:
            # Clean up: remove the temporary directory
            shutil.rmtree(temp_dir)
            logging.info(f"Removed temporary directory: {temp_dir}")
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