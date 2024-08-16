import requests
import argparse
import logging
from shapely.geometry import Polygon
import numpy as np

# Configure logging
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

def get_huc8_polygon(lat, lon):
    """
    Get the HUC8 polygon for a given latitude and longitude.
    """
    url = "https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer/4/query"
    params = {
        "f": "json",
        "geometry": "{},{}".format(lon, lat),
        "geometryType": "esriGeometryPoint",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "outSR": 4326,
        "returnGeometry": True,
        "inSR": 4326
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error making request: {e}")
        return None

    try:
        data = response.json()
        features = data["features"]
        if features:
            feature = features[0]
            polygon = feature["geometry"]["rings"][0]
            return polygon
        else:
            return None
    except (KeyError, ValueError):
        logging.error("Failed to parse response")
        return None

from shapely.geometry import Polygon
import numpy as np

def simplify_polygon(polygon, max_points=100, tolerance=0.001):
    # Create a Shapely polygon
    shapely_polygon = Polygon(polygon)
    
    # Simplify the polygon
    simplified = shapely_polygon.simplify(tolerance=tolerance, preserve_topology=True)
    
    # Extract coordinates
    coords = list(simplified.exterior.coords)
    
    # Ensure counter-clockwise orientation
    if not is_ccw(coords):
        coords = coords[::-1]
    
    # If the first and last points are not the same, add the first point at the end
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    
    # Limit the number of points if necessary
    if len(coords) > max_points:
        coords = coords[:max_points]
        coords.append(coords[0])  # Ensure it's still closed
    
    # Round coordinates to 6 decimal places and convert to list of (lon, lat) tuples
    coords = [(round(float(lon), 6), round(float(lat), 6)) for lon, lat in coords]
    
    return coords

def is_ccw(coords):
    """Check if coordinates are in counter-clockwise order."""
    s = 0
    for i in range(len(coords) - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        s += (x2 - x1) * (y2 + y1)
    return s < 0

def is_point_on_line(p1, p2, p):
    x0, y0 = p
    x1, y1 = p1
    x2, y2 = p2
    return (x2 - x1) * (y1 - y0) == (x1 - x0) * (y2 - y1)

def simplify_polygon_rdp(polygon, epsilon):
    if len(polygon) < 3:
        return polygon
    dmax = 0
    index = 0
    for i in range(1, len(polygon) - 1):
        d = perpendicular_distance(polygon[0], polygon[-1], polygon[i])
        if d > dmax:
            index = i
            dmax = d
    if dmax > epsilon:
        results = simplify_polygon_rdp(polygon[:index+1], epsilon) + simplify_polygon_rdp(polygon[index:], epsilon)[1:]
    else:
        results = [polygon[0], polygon[-1]]
    return results

def perpendicular_distance(p1, p2, p):
    """
    Calculate the perpendicular distance from point p to the line segment p1-p2.
    """
    x0, y0 = p
    x1, y1 = p1
    x2, y2 = p2
    denominator = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
    if denominator == 0:
        return 0  # or some other default value
    return abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)) / denominator

def main(lat, lon):
    huc8_polygon = get_huc8_polygon(lat, lon)
    if huc8_polygon:
        simplified_polygon = simplify_polygon(huc8_polygon)
        logging.debug(f"Simplified polygon: {simplified_polygon}")
        return simplified_polygon
    else:
        logging.error("No HUC8 polygon found")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetch HUC8 polygon for a given latitude and longitude.')
    parser.add_argument('--lat', type=float, required=True, help='Latitude')
    parser.add_argument('--lon', type=float, required=True, help='Longitude')
    args = parser.parse_args()
    main(args.lat, args.lon)