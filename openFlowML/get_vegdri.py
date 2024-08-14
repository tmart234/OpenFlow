import requests
import argparse
import pandas as pd
import logging
from datetime import datetime, timedelta
import json
from shapely.geometry import Polygon, Point
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from get_poly import get_huc8_polygon, simplify_polygon

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_vegdri_data(geometry, date):
    """
    Get VegDRI data for a given geometry and date.
    """
    if not (geometry and date):
        logging.error("Missing required parameters")
        return None

    try:
        geometry_json = json.loads(geometry)
    except ValueError:
        logging.error("Invalid geometry. Must be a GeoJSON object")
        return None

    if geometry_json['type'] not in ['Point', 'Polygon']:
        logging.error("Only Point and Polygon geometry types are supported")
        return None

    # Convert geometry to shapely object
    if geometry_json['type'] == 'Point':
        shape = Point(geometry_json['coordinates'])
    else:  # Polygon
        shape = Polygon(geometry_json['coordinates'][0])

    # Get the bounding box
    minx, miny, maxx, maxy = shape.bounds

    # Set up the request parameters
    base_url = "https://vegdri.cr.usgs.gov/arcgis/rest/services/VegDRI/VegDRI_Current/ImageServer/exportImage"
    
    params = {
        "bbox": f"{minx},{miny},{maxx},{maxy}",
        "bboxSR": 4326,
        "size": "500,500",
        "imageSR": 4326,
        "time": date,
        "format": "tiff",
        "pixelType": "F32",
        "noData": "",
        "noDataInterpretation": "esriNoDataMatchAny",
        "interpolation": "RSP_BilinearInterpolation",
        "compression": "",
        "compressionQuality": 100,
        "bandIds": "",
        "mosaicRule": "",
        "renderingRule": "",
        "f": "json"
    }

    logging.info(f"Fetching VegDRI data from: {base_url} with params: {params}")
    try:
        response = requests.get(base_url, params=params)
        print(response.url)
    except requests.exceptions.RequestException as e:
        logging.error(f"Error making request: {e}")
        return None

    if response.status_code == 200:
        try:
            data = response.json()
            if 'href' in data:
                # Download the TIFF file
                tiff_response = requests.get(data['href'])
                if tiff_response.status_code == 200:
                    # Save the TIFF file
                    with open('vegdri_data.tiff', 'wb') as f:
                        f.write(tiff_response.content)
                    logging.info("VegDRI data saved as 'vegdri_data.tiff'")
                    
                    # Read the TIFF file and extract data for the geometry
                    with rasterio.open('vegdri_data.tiff') as src:
                        out_image, out_transform = mask(src, [shape], crop=True)
                        out_meta = src.meta.copy()
                        out_meta.update({"driver": "GTiff",
                                         "height": out_image.shape[1],
                                         "width": out_image.shape[2],
                                         "transform": out_transform})
                    
                    # Convert the masked data to a pandas DataFrame
                    df = pd.DataFrame(out_image[0].flatten(), columns=['VegDRI'])
                    df['latitude'], df['longitude'] = rasterio.transform.xy(out_transform, 
                                                                            range(out_meta['height']),
                                                                            range(out_meta['width']))
                    return df
                else:
                    logging.error("Failed to download TIFF file")
                    return None
            else:
                logging.error("No image URL in the response")
                return None
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
        if data is not None:
            logging.info(f"Received VegDRI data: {data.head()}")
            return data
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