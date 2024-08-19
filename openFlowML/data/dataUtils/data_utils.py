import logging
import os
from dotenv import load_dotenv

 # Additional function to display the beginning and ending of the dataframe
def preview_data(df, num_rows=4):
    logging.info("First few rows:")
    logging.info(df.head(num_rows))
    logging.info("\nLast few rows:")
    logging.info(df.tail(num_rows))

def load_vars():
    # Load environment variables from cred.env
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cred_env_path = os.path.join(script_dir, 'creds.env')
    if os.path.exists(cred_env_path):
        load_dotenv(cred_env_path)
        logging.info(f"Loaded environment variables from {cred_env_path}")
    else:
        logging.error(f"creds.env file not found at {cred_env_path}")
        exit(1)

    # Check if environment variables are set
    if not os.getenv("EARTHDATA_USERNAME") or not os.getenv("EARTHDATA_PASSWORD") \
        or not os.getenv("EROS_API_KEY") or not os.getenv("EROS_USERNAME") \
        or not os.getenv("EROS_USERNAME"):
        logging.error("environment variables are not set in the cred.env file")
        exit(1)
    else:
        logging.info("environment variables are set")
