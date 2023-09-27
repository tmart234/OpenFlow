import geopandas as gpd
from shapely.geometry import Point
import requests
from zipfile import ZipFile
from io import BytesIO
import os

# URL of the zipped shapefile
shapefile_url = 'https://geo.colorado.edu/apps/geolibrary/datasets/WBDHU12.zip'

def download_and_unzip(url, download_dir):
    response = requests.get(url)
    
    if response.status_code == 200:
        # Get the file extension from the URL
        file_extension = os.path.splitext(url)[1]
        
        if file_extension.lower() == '.zip':
            # If it's a zip file, proceed with unzipping
            with ZipFile(BytesIO(response.content)) as zip_file:
                zip_file.extractall(download_dir)
                # Assuming the shapefile is named 'WBDHU12.shp' in the zip archive
                shapefile_path = os.path.join(download_dir, 'WBDHU12.shp')
                return shapefile_path
        elif file_extension.lower() == '.shp':
            # If it's already a shapefile, return the path directly
            return url
        else:
            print("Unsupported file format. Please provide a .zip or .shp file.")
            return None
    else:
        print("Failed to download the file from the URL.")
        return None

def find_watershed(gps_coordinate, shapefile_path):
    # Load the shapefile using geopandas
    watersheds = gpd.read_file(shapefile_path)

    # Print the column names to see what's available
    print("Available columns in the shapefile:", watersheds.columns)

    # Create a Point object from the GPS coordinate
    point = Point(gps_coordinate)

    # Iterate through each watershed polygon and check if the point is within it
    for idx, watershed in watersheds.iterrows():
        if point.within(watershed.geometry):  # Use 'geometry' for the polygon
            return watershed  # Return the entire watershed data

    return "Not Found"  # If no watershed is found

# Directory to store the downloaded and unzipped shapefile
download_directory = 'shapefile_data'

# Ensure the download directory exists
os.makedirs(download_directory, exist_ok=True)

# Download and unzip the shapefile, and get the path to the shapefile
shapefile_path = download_and_unzip(shapefile_url, download_directory)

if shapefile_path is not None:
    # Example GPS coordinate for the SWE station (39.181624, -106.282648)
    gps_coordinate = (-106.282648, 39.181624)
    
    # Find the watershed for the GPS coordinate
    result = find_watershed(gps_coordinate, shapefile_path)
    if isinstance(result, str) and result == "Not Found":
        print("The GPS coordinate is not within any watershed.")
    else:
        print("The GPS coordinate is in the watershed with data:")
        # Split the Name column into two variables based on the dash "-"
        name_parts = result.Name.split('-')
        # Assign the parts to separate variables
        part1 = name_parts[0].strip()  # Removes leading/trailing whitespace
        part2 = name_parts[1].strip()
        print(part2 + " near " + part1)
