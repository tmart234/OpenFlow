import requests
import json
import os
import tempfile
import time
import shutil
from datetime import datetime, timedelta
from dataUtils.data_utils import load_vars
import logging

load_vars()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    
    logger.info(f"Sending request to URL: {url}")
    logger.debug(f"Headers: {headers}")
    logger.debug(f"Data: {data}")
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(url, json=data, headers=headers)
            logger.debug(f"Response status code: {response.status_code}")
            logger.debug(f"Response headers: {response.headers}")
            logger.debug(f"Response content: {response.text}")
            
            response.raise_for_status()
            result = response.json()
            
            if result.get('errorCode'):
                raise Exception(f"{result['errorCode']}: {result['errorMessage']}")
            
            return result['data']
        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < MAX_RETRIES - 1:
                logger.info(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                logger.error("Max retries reached. Unable to complete the request.")
                raise

def login():
    if not USERNAME or not PASSWORD:
        raise ValueError("EROS_USERNAME and EROS_PASSWORD environment variables must be set")
    
    payload = {'username': USERNAME, 'password': PASSWORD}
    api_key = send_request("login", payload)
    logger.info(f"Logged in successfully. API Key: {api_key}")
    return api_key

def logout(api_key):
    send_request("logout", None, api_key)
    logger.info("Logged out successfully")

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
    try:
        return send_request("download-options", payload, api_key)
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 403:
            logger.warning("Unable to access download options. Your account may not have the necessary permissions.")
            return None
        else:
            raise

def request_download(api_key, downloads):
    payload = {
        'downloads': downloads,
        'label': f"VEGDRI_download_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    }
    return send_request("download-request", payload, api_key)

def download_file(url, final_filename):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as temp_file:
        logger.info(f"Downloading to temporary file: {temp_file.name}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        for chunk in response.iter_content(chunk_size=8192):
            temp_file.write(chunk)
    
    try:
        shutil.move(temp_file.name, final_filename)
        logger.info(f"File successfully moved to: {final_filename}")
    except Exception as e:
        logger.error(f"Failed to move file: {str(e)}")
        os.unlink(temp_file.name)
        raise

def main():
    api_key = login()
    
    try:
        scenes = search_scenes(api_key)
        if not scenes:
            logger.info("No scenes found.")
            return
        
        latest_scene = scenes[0]
        entity_id = latest_scene['entityId']
        
        download_options = get_download_options(api_key, entity_id)
        if not download_options:
            logger.info("No download options available.")
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
            logger.info("Download not immediately available. Please check the USGS Earth Explorer site later.")
    
    except Exception as e:
        logger.exception(f"An error occurred: {str(e)}")
    finally:
        if 'api_key' in locals():
            logout(api_key)

if __name__ == "__main__":
    main()