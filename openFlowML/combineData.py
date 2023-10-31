# combineData.py

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def combine_and_normalize_data(temp_file_path, flow_file_path, output_file_path):
    # Read the CSVs
    temp_data = pd.read_csv(temp_file_path)
    flow_data = pd.read_csv(flow_file_path)

    # Merge on the 'Date' column
    combined_data = pd.merge(temp_data, flow_data, on='Date', how='outer')

    # Columns to normalize
    columns_to_normalize = ['TMIN', 'TMAX', 'Min Flow', 'Max Flow']

    # Normalize the data
    for column in columns_to_normalize:
        scaler = StandardScaler()
        combined_data[column] = scaler.fit_transform(combined_data[column].values.reshape(-1, 1))

    # Save the combined, normalized data
    combined_data.to_csv(output_file_path, index=False)

if __name__ == "__main__":
    TEMP_INPUT_PATH = os.path.join(os.getenv('GITHUB_WORKSPACE'), 'openFlowML', 'temperature_data.csv')
    FLOW_INPUT_PATH = os.path.join(os.getenv('GITHUB_WORKSPACE'), 'openFlowML', 'daily_flow_data.csv')
    OUTPUT_PATH = os.path.join(os.getenv('GITHUB_WORKSPACE'), 'openFlowML', 'combined_normalized_data.csv')

    combine_and_normalize_data(TEMP_INPUT_PATH, FLOW_INPUT_PATH, OUTPUT_PATH)
    
    # Update the CSV_FILE_PATH to point to the combined normalized data for the artifact step
    with open(os.getenv('GITHUB_ENV'), 'a') as env_file:
        env_file.write(f"CSV_FILE_PATH={OUTPUT_PATH}\n")
