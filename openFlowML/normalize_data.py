import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def normalize_temperature_data(input_file_path, output_file_path):
    # Read the CSV
    data = pd.read_csv(input_file_path)

    # Ensure 'TMIN' and 'TMAX' columns are in your CSV
    columns_to_normalize = ['TMIN', 'TMAX']

    # Create a scaler for each column
    scalers = {}
    for column in columns_to_normalize:
        scaler = StandardScaler()
        data[column] = scaler.fit_transform(data[column].values.reshape(-1, 1))
        scalers[column] = scaler

    # Save the normalized data
    data.to_csv(output_file_path, index=False)

if __name__ == "__main__":
    INPUT_PATH = os.getenv('CSV_FILE_PATH')  # Retrieve the path from the environment variable
    OUTPUT_PATH = INPUT_PATH.replace('.csv', '_normalized.csv')  # Name the normalized CSV
    normalize_temperature_data(INPUT_PATH, OUTPUT_PATH)
    
    # Update the CSV_FILE_PATH to point to the normalized data for the artifact step
    with open(os.getenv('GITHUB_ENV'), 'a') as env_file:
        env_file.write(f"CSV_FILE_PATH={OUTPUT_PATH}\n")

