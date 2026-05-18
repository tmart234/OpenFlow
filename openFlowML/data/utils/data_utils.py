import logging
import os
import time
import requests
from datetime import timedelta

# NOTE: python-dotenv is a heavy optional dependency used only by load_vars
# (credential loading for the soil-moisture path). It is imported lazily so
# lightweight consumers like combine_data work with only the core requirements.


# Additional function to display the beginning and ending of the dataframe
def preview_data(df, num_rows=4):
    logging.info("First few rows:")
    logging.info(df.head(num_rows))
    logging.info("\nLast few rows:")
    logging.info(df.tail(num_rows))


def request_with_retry(url, params=None, headers=None, timeout=60, retries=3, backoff=2):
    """
    GET a URL with a timeout and simple exponential backoff.

    Returns the Response on HTTP 200, or None if every attempt failed. The flow
    fetchers depend on this because their upstream services (USGS IV, CO DWR)
    are slow and occasionally flaky on long-range requests.
    """
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
            if response.status_code == 200:
                return response
            logging.warning("HTTP %s from %s (attempt %d/%d)",
                            response.status_code, url, attempt + 1, retries)
        except requests.exceptions.RequestException as e:
            logging.warning("Request error for %s: %s (attempt %d/%d)",
                            url, e, attempt + 1, retries)
        if attempt < retries - 1:
            time.sleep(backoff ** attempt)
    logging.error("Giving up on %s after %d attempts", url, retries)
    return None


def date_chunks(start_date, end_date, max_days=366):
    """
    Yield (chunk_start, chunk_end) datetime pairs covering [start_date, end_date]
    in windows of at most `max_days` days.

    Long-range requests to the USGS IV service are silently truncated, so the
    flow fetchers request one bounded chunk at a time and concatenate.
    """
    current = start_date
    while current <= end_date:
        chunk_end = min(current + timedelta(days=max_days - 1), end_date)
        yield current, chunk_end
        current = chunk_end + timedelta(days=1)


def load_vars():
    """
    Load credentials from creds.env into the environment.

    Returns True if all required variables are present, False otherwise. This
    function must never exit the process: importing a data module should not
    kill the interpreter just because credentials are absent (e.g. in CI or
    when running the test suite).
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        logging.warning("python-dotenv not installed; skipping creds.env load")
        load_dotenv = None

    script_dir = os.path.dirname(os.path.abspath(__file__))
    cred_env_path = os.path.join(script_dir, 'creds.env')
    if load_dotenv is not None and os.path.exists(cred_env_path):
        load_dotenv(cred_env_path)
        logging.info(f"Loaded environment variables from {cred_env_path}")
    elif load_dotenv is not None:
        logging.warning(f"creds.env file not found at {cred_env_path}")

    required = ["EARTHDATA_USERNAME", "EARTHDATA_PASSWORD",
                "EROS_API_KEY", "EROS_USERNAME", "EROS_PASSWORD"]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        logging.warning(f"Missing required environment variables: {missing}")
        return False
    logging.info("environment variables are set")
    return True
