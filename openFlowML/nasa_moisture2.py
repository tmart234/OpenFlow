import datetime
import numpy as np
import h5py
from shapely.geometry import Polygon, Point
import logging
import argparse
from io import BytesIO
from earthdata import Auth, DataGranule, DataCollections

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
                return np.mean(valid_data)
            else:
                logging.warning("No valid soil moisture data found within the polygon")
                return None
    except Exception as e:
        logging.error(f"Error extracting soil moisture data: {e}")
        return None

def main(date, polygon):
    # Set up authentication
    auth = Auth()
    auth.login()

    hdf_file = search_and_download_smap_data(date, auth)
    if hdf_file:
        soil_moisture = extract_soil_moisture(hdf_file, polygon)
        if soil_moisture is not None:
            logging.info(f"Average soil moisture for the given polygon on or before {date}: {soil_moisture:.4f}")
        else:
            logging.error("Failed to calculate soil moisture")
    else:
        logging.error("Failed to find or download SMAP data")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate average soil moisture for a polygon from SMAP L3 data.')
    parser.add_argument('--date', type=lambda d: datetime.datetime.strptime(d, '%Y-%m-%d').date(), required=True, help='Date in YYYY-MM-DD format')
    parser.add_argument('--polygon', type=float, nargs='+', required=True, help='Polygon coordinates as a flat list of alternating longitude and latitude values')
    args = parser.parse_args()

    # Reshape the flat list of coordinates into pairs
    polygon = list(zip(args.polygon[::2], args.polygon[1::2]))
    
    main(args.date, polygon)