import requests
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def simplify_polygon(polygon, max_points=100):
    if len(polygon) <= max_points:
        return polygon
    simplified_polygon = []
    for point in polygon:
        if len(simplified_polygon) < 2 or not is_point_on_line(simplified_polygon[-2], simplified_polygon[-1], point):
            simplified_polygon.append(point)        
    return simplified_polygon

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
        logging.info(f"Simplified polygon: {simplified_polygon}")
        return simplified_polygon
    else:
        logging.error("No HUC8 polygon found")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetch HUC8 polygon for a given latitude and longitude.')
    parser.add_argument('--lat', type=float, required=True, help='Latitude')
    parser.add_argument('--lon', type=float, required=True, help='Longitude')
    args = parser.parse_args()
    main(args.lat, args.lon)