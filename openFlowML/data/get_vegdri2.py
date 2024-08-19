import requests
import json
import os
import time
from datetime import datetime, timedelta
from dataUtils.data_utils import load_vars

load_vars()

# Configuration
SERVICE_URL = "https://m2m.cr.usgs.gov/api/api/json/stable/"
USERNAME = os.getenv("EROS_USERNAME")
PASSWORD = os.getenv("EROS_PASSWORD")
DATASET_NAME = "VEGDRI"
MAX_RESULTS = 1  # We only need the latest dataset
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

def send_request(endpoint, data, api_key=None):
    url = SERVICE_URL + endpoint
    headers = {'X-Auth-Token': api_key} if api_key else None
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(url, json=data, headers=headers)
            response.raise_for_status()
            result = response.json()
            
            if result.get('errorCode'):
                raise Exception(f"{result['errorCode']}: {result['errorMessage']}")
            
            return result['data']
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < MAX_RETRIES - 1:
                print(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                print("Max retries reached. Unable to complete the request.")
                raise

def login():
    if not USERNAME or not PASSWORD:
        raise ValueError("EROS_USERNAME and EROS_PASSWORD environment variables must be set")
    
    payload = {'username': USERNAME, 'password': PASSWORD}
    api_key = send_request("login", payload)
    print("Logged in successfully")
    return api_key

def logout(api_key):
    send_request("logout", None, api_key)
    print("Logged out successfully")

def search_scenes(api_key):
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    payload = {
        'datasetName': DATASET_NAME,
        'maxResults': MAX_RESULTS,
        'startingNumber': 1,
        'sceneFilter': {
            'acquisitionFilter': {
                'start': start_date,
                'end': end_date
            }
        },
        'sortOrder': 'DESC'
    }
    
    scenes = send_request("scene-search", payload, api_key)
    return scenes['results']

def get_download_options(api_key, entity_id):
    payload = {
        'datasetName': DATASET_NAME,
        'entityIds': [entity_id]
    }
    return send_request("download-options", payload, api_key)

def request_download(api_key, downloads):
    payload = {
        'downloads': downloads,
        'label': f"VEGDRI_download_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    }
    return send_request("download-request", payload, api_key)

def download_file(url, filename):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(filename, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print(f"Downloaded: {filename}")

def main():
    api_key = login()
    
    try:
        scenes = search_scenes(api_key)
        if not scenes:
            print("No scenes found.")
            return
        
        latest_scene = scenes[0]
        entity_id = latest_scene['entityId']
        
        download_options = get_download_options(api_key, entity_id)
        if not download_options:
            print("No download options available.")
            return
        
        downloads = [{
            'entityId': entity_id,
            'productId': download_options[0]['id']  # Assuming the first option is the one we want
        }]
        
        download_results = request_download(api_key, downloads)
        
        if download_results['availableDownloads']:
            url = download_results['availableDownloads'][0]['url']
            filename = f"VEGDRI_{latest_scene['acquisitionDate']}.zip"
            download_file(url, filename)
        else:
            print("Download not immediately available. Please check the USGS Earth Explorer site later.")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        if 'api_key' in locals():
            logout(api_key)

if __name__ == "__main__":
    main()