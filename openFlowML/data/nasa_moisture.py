import datetime
import numpy as np
import h5py
from shapely.geometry import Polygon, Point
import logging
import argparse
from io import BytesIO
from earthaccess import Auth, DataCollections
from dataUtils.get_poly import get_huc8_polygon, simplify_polygon

# Conditionally import matplotlib
import importlib.util
matplotlib_spec = importlib.util.find_spec("matplotlib")
matplotlib_available = matplotlib_spec is not None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def search_and_download_smap_data(date, auth):
    """
    Search for the most recent SMAP L3 data up to the given date and download it.
    """
    collection = DataCollections.smap_l3_soil_moisture
    end_date = date
    start_date = end_date - datetime.timedelta(days=7)  # Look for data up to a week before the specified date
    
    try:
        granules = collection.search(
            temporal=(start_date, end_date),
            bounding_box=(-180, -90, 180, 90),  # Global coverage
            limit=1,
            sort_key="-start_date"
        )
        
        if not granules:
            logging.error(f"No SMAP data found from {start_date} to {end_date}")
            return None
        
        granule = granules[0]
        logging.info(f"Found SMAP data for date: {granule.date}")
        
        # Download the data
        data = granule.download(auth)
        return BytesIO(data)
    
    except Exception as e:
        logging.error(f"Error searching or downloading SMAP data: {e}")
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

def main(date, lat, lon, visual):
    # Set up authentication
    auth = Auth()
    auth.login()

    # Get and simplify the HUC8 polygon
    huc8_polygon = get_huc8_polygon(lat, lon)
    if not huc8_polygon:
        logging.error("Failed to retrieve HUC8 polygon")
        return

    simplified_polygon = simplify_polygon(huc8_polygon)
    logging.info(f"Simplified polygon: {simplified_polygon}")

    hdf_file = search_and_download_smap_data(date, auth)
    if hdf_file:
        soil_moisture, full_soil_moisture, lat_data, lon_data, mask = extract_soil_moisture(hdf_file, simplified_polygon)
        if soil_moisture is not None:
            logging.info(f"Average soil moisture for the given polygon on or before {date}: {soil_moisture:.4f}")
            
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
    parser.add_argument('--date', type=lambda d: datetime.datetime.strptime(d, '%Y-%m-%d').date(), required=True, help='Date in YYYY-MM-DD format')
    parser.add_argument('--lat', type=float, required=True, help='Latitude of the point within the desired HUC8 polygon')
    parser.add_argument('--lon', type=float, required=True, help='Longitude of the point within the desired HUC8 polygon')
    parser.add_argument('--visual', action='store_true', help='Enable matplotlib visualization')
    args = parser.parse_args()

    main(args.date, args.lat, args.lon, args.visual)