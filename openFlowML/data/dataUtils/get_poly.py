import requests
import argparse
import logging
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_polygon(polygon):
    """
    Validate and clean up polygon coordinates.
    """
    # Remove any duplicate consecutive points
    cleaned = [polygon[0]]
    for point in polygon[1:]:
        if point != cleaned[-1]:
            cleaned.append(point)
    
    # Ensure the polygon is closed
    if cleaned[0] != cleaned[-1]:
        cleaned.append(cleaned[0])
    
    # Check if polygon is counter-clockwise, reverse if not
    if not is_ccw(cleaned):
        cleaned = cleaned[::-1]
    
    return cleaned

def visualize_data_and_polygon(lat, lon, soil_moisture, polygon):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 8))
    plt.scatter(lon, lat, c=soil_moisture, cmap='viridis', s=1)
    plt.colorbar(label='Soil Moisture')
    
    poly = Polygon(polygon, facecolor='none', edgecolor='red')
    plt.gca().add_patch(poly)
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('SMAP Data and HUC8 Polygon')
    plt.show()

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
        data = response.json()
        features = data.get("features", [])
        if features:
            feature = features[0]
            polygon = feature["geometry"]["rings"][0]
            logger.info(f"Retrieved HUC8 polygon with {len(polygon)} points")
            return polygon
        else:
            logger.warning("No HUC8 polygon found for the given coordinates")
            return None
    except Exception as e:
        logger.error(f"Error retrieving HUC8 polygon: {e}")
        return None

def simplify_polygon(polygon, max_points=200, tolerance=0.0001):
    """
    Simplify the polygon using Shapely's simplify method and ensure it's properly closed.
    """
    shapely_polygon = Polygon(polygon)
    simplified = shapely_polygon.simplify(tolerance=tolerance, preserve_topology=True)
    
    # Extract coordinates and ensure the polygon is closed
    coords = list(simplified.exterior.coords)
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    
    # Limit the number of points if necessary
    if len(coords) > max_points:
        coords = coords[:max_points]
        coords.append(coords[0])  # Ensure it's still closed
    
    # Round coordinates to 6 decimal places
    coords = [(round(float(lon), 6), round(float(lat), 6)) for lon, lat in coords]
    
    logger.info(f"Simplified polygon to {len(coords)} points")
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

def check_polygon_intersection(polygon, data_bounds):
    """
    Check if the polygon intersects with the SMAP data bounds.
    """
    polygon_shape = Polygon(polygon)
    data_box = box(*data_bounds)
    intersection = polygon_shape.intersection(data_box)
    
    if not intersection.is_empty:
        logger.info("Polygon intersects with SMAP data bounds")
        return True
    else:
        logger.warning("Polygon does not intersect with SMAP data bounds")
        return False

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

def main(lat, lon, data_bounds=None):
    huc8_polygon = get_huc8_polygon(lat, lon)
    if huc8_polygon:
        simplified_polygon = simplify_polygon(huc8_polygon)
        logger.debug(f"Simplified polygon: {simplified_polygon}")
        
        if data_bounds:
            intersects = check_polygon_intersection(simplified_polygon, data_bounds)
            if not intersects:
                logger.warning("The polygon does not intersect with the SMAP data. No soil moisture data may be available for this area.")
        
        return simplified_polygon
    else:
        logger.error("No HUC8 polygon found")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetch and simplify HUC8 polygon for a given latitude and longitude.')
    parser.add_argument('--lat', type=float, required=True, help='Latitude')
    parser.add_argument('--lon', type=float, required=True, help='Longitude')
    args = parser.parse_args()
    
    # Example SMAP data bounds (you should replace these with actual bounds from your SMAP data)
    smap_data_bounds = (-180, -90, 180, 90)
    
    result = main(args.lat, args.lon, smap_data_bounds)
    if result:
        print(f"Simplified HUC8 polygon: {result}")
    else:
        print("Failed to retrieve or process HUC8 polygon")