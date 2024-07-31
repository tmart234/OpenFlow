import requests
import json

def get_huc8_polygon(lat, lon):
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

    response = requests.get(url, params=params)
    print(response.url)
    
    if response.status_code == 200:
        data = response.json()
        with open('huc8_polygon.json', 'w') as f:
            json.dump(data, f, indent=4)
        print("JSON saved to huc8_polygon.json")
        features = data["features"]
        if features:
            feature = features[0]
            polygon = feature["geometry"]["rings"][0]
            return polygon
        else:
            return None
    else:
        return None

# Example usage
lat = 39.7392
lon = -104.9903
huc8_polygon = get_huc8_polygon(lat, lon)
print(huc8_polygon)