import geopandas as gpd
from shapely.geometry import Point
import requests
from zipfile import ZipFile, BadZipFile
from io import BytesIO
import os
import tempfile

WATERSHED_URLS = {
    "colorado_headwaters": "https://nwcc-apps.sc.egov.usda.gov/awdb/basin-plots/POR/WTEQ/assocHUCco_8/colorado_headwaters.json",
    "yampa_white_little_snake": "https://nwcc-apps.sc.egov.usda.gov/awdb/basin-plots/POR/WTEQ/assocHUCco_8/yampa-white-little_snake.json",
    "arkansas": "https://nwcc-apps.sc.egov.usda.gov/awdb/basin-plots/POR/WTEQ/assocHUCco_8/arkansas.json",
    # Add more mappings as needed
}

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

    # Create a Point object from the GPS coordinate
    point = Point(gps_coordinate)

    # Iterate through each watershed polygon and check if the point is within it
    for idx, watershed in watersheds.iterrows():
        if point.within(watershed.geometry):  # Use 'geometry' for the polygon
            return watershed  # Return the entire watershed data

    return None  # If no watershed is found

def get_hu12_watershed(gps_coordinate):
    # Directory to store the downloaded and unzipped shapefile
    download_directory = 'shapefile_data'

    # Ensure the download directory exists
    os.makedirs(download_directory, exist_ok=True)

    # URL of the zipped shapefile
    shapefile_url = 'https://geo.colorado.edu/apps/geolibrary/datasets/WBDHU12.zip'

    # Download and unzip the shapefile, and get the path to the shapefile
    shapefile_path = download_and_unzip(shapefile_url, download_directory)

    if shapefile_path is not None:
        # Find the watershed for the GPS coordinate
        result = find_watershed(gps_coordinate, shapefile_path)
        if result is None:
            return None  # No watershed found
        else:
            # Split the Name column into two variables based on the dash "-"
            name_parts = result.Name.split('-')
            # Assign the parts to separate variables
            part1 = name_parts[0].strip()  # Removes leading/trailing whitespace
            part2 = name_parts[1].strip()
            return (part1, part2)
    else:
        return None  # Error in downloading or unzipping the shapefile
    
def get_hu2_watershed(url, coords):
    # Send a GET request to the server
    response = requests.get(url)
    
    # Check response status and content type
    print("HTTP Status Code:", response.status_code)
    print("HTTP Headers:", response.headers)

    if response.headers.get('Content-Type', '').lower() == 'application/zip':
        try:
            with ZipFile(BytesIO(response.content)) as the_zip:
                print("Files in the zip:", the_zip.namelist())
                
                # Assuming there is a specific .gdb file to look for
                file_name = [name for name in the_zip.namelist() if name.endswith('.gdb') and 'WBDHU2' in name]
                if not file_name:
                    print("No suitable .gdb file found in the zip.")
                    return None

                # Extract to a temporary directory using extractall method
                with tempfile.TemporaryDirectory() as temp_dir:
                    the_zip.extractall(temp_dir)
                    gdb_path = os.path.join(temp_dir, file_name[0])
                    gdf = gpd.read_file(gdb_path, layer='WBDHU2')
                    
                    point = Point(coords)
                    contained_watersheds = gdf[gdf.geometry.contains(point)]
                    
                    if not contained_watersheds.empty:
                        return contained_watersheds.iloc[0]['name'], contained_watersheds.iloc[0]['huc2']
                    else:
                        return None

        except BadZipFile:
            print("The downloaded file is not a zip file.")
    else:
        print("Downloaded content is not a zip file.")
        return None

if __name__ == "__main__":
    # Example GPS coordinate for the SWE station (39.181624, -106.282648)
    gps_coordinate = (-106.282648, 39.181624)
    wbdhu2_path = 'https://nrcs.app.box.com/v/huc/file/1300081266989'

    # Get watershed information for the GPS coordinate
    # hu12_result = get_hu12_watershed(gps_coordinate)
    hu2_result = get_hu2_watershed(wbdhu2_path, gps_coordinate)
    hu12_result = None
    
    if hu12_result is None:
        print("Error: Unable to find HU12 watershed information.")
    else:
        part1, part2 = hu12_result
        print(f"HU12 found: {part2} near {part1}")
              
    print(hu2_result)
    
