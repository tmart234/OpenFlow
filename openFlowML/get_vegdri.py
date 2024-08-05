import requests
import argparse
import pandas as pd
import logging
from datetime import datetime
import json
from get_poly import *

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_vegdri_data(geometry, date):
    """
    Get VegDRI data for a given coordinate or polygon shape and time.
    """
    if not (geometry and date):
        logging.error("Missing required parameters")
        return None

    try:
        geometry_json = json.loads(geometry)
    except ValueError:
        # Assume geometry is a single coordinate
        try:
            latitude, longitude = map(float, geometry.split(','))
            geometry_json = {
                "type": "Point",
                "coordinates": [longitude, latitude]
            }
        except ValueError:
            logging.error("Invalid geometry. Must be a GeoJSON object or a single coordinate (latitude,longitude)")
            return None

    if 'type' not in geometry_json:
        logging.error("Invalid GeoJSON object. Must have a 'type' property")
        return None

    if geometry_json['type'] not in ['Point', 'Polygon']:
        logging.error("Only Point and Polygon geometry types are supported")
        return None

    if geometry_json['type'] == 'Point':
        if 'coordinates' not in geometry_json:
            logging.error("Invalid GeoJSON Point object. Must have a 'coordinates' property")
            return None
        if len(geometry_json['coordinates']) != 2:
            logging.error("Invalid GeoJSON Point object. Coordinates must be in the format [longitude, latitude]")
            return None
        if not (-180 <= geometry_json['coordinates'][0] <= 180):
            logging.error("Longitude must be between -180 and 180")
            return None
        if not (-90 <= geometry_json['coordinates'][1] <= 90):
            logging.error("Latitude must be between -90 and 90")
            return None

    if geometry_json['type'] == 'Polygon':
        if 'coordinates' not in geometry_json:
            logging.error("Invalid GeoJSON Polygon object. Must have a 'coordinates' property")
            return None
        if not isinstance(geometry_json['coordinates'], list):
            logging.error("Invalid GeoJSON Polygon object. Coordinates must be a list of lists")
            return None

    try:
        datetime.strptime(date, '%Y-%m-%d')
    except ValueError:
        logging.error("Invalid date format. Must be YYYY-MM-DD")
        return None

    api_url = "https://vegdri.cr.usgs.gov/api/v1/data"
    params = {
        "geometry": json.dumps(geometry_json),
        "date": date,
        "format": "json"
    }

    logging.info(f"Fetching VegDRI data from: {api_url} with params: {params}")
    try:
        response = requests.get(api_url, params=params)
        print(response.url)
    except requests.exceptions.RequestException as e:
        logging.error(f"Error making request: {e}")
        return None

    if response.status_code == 200:
        try:
            return response.json()
        except requests.exceptions.JSONDecodeError:
            logging.error("Failed to parse JSON response")
            return None
    else:
        logging.error(f"Error: {response.status_code} - {response.text}")
        return None

def main(lat, lon, date):
    huc8_polygon = get_huc8_polygon(lat, lon)
    if huc8_polygon:
        simplified_polygon = simplify_polygon(huc8_polygon)
        logging.debug(f"Simplified polygon: {simplified_polygon}")
        geometry = json.dumps({
            "type": "Polygon",
            "coordinates": [simplified_polygon]
        })
        data = get_vegdri_data(geometry, date)
        if data:
            logging.info(f"Received VegDRI data: {data}")
            try:
                df = pd.json_normalize(data)
                logging.info(f"Data converted to DataFrame: {df.head()}")
                return df
            except ValueError:
                logging.error("Failed to convert data to DataFrame")
                return None
        else:
            logging.error("No data received")
            return None
    else:
        logging.error("No HUC8 polygon found")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetch HUC8 polygon and VegDRI data for a given latitude, longitude, and date.')
    parser.add_argument('--lat', type=float, required=True, help='Latitude')
    parser.add_argument('--lon', type=float, required=True, help='Longitude')
    parser.add_argument('--date', type=str, required=True, help='Date in the format YYYY-MM-DD')
    args = parser.parse_args()
    main(args.lat, args.lon, args.date)