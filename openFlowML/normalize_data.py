import pandas as pd
from sklearn.preprocessing import StandardScaler

def normalize_temperature_data(input_file_path, output_file_path):
    # Read the CSV
    data = pd.read_csv(input_file_path)

    # Ensure temperatures is a column in your CSV
    scaler = StandardScaler()
    data['temperatures'] = scaler.fit_transform(data['temperatures'].values.reshape(-1, 1))

    # Save the normalized data
    data.to_csv(output_file_path, index=False)

if __name__ == "__main__":
    INPUT_PATH = 'path_to_input_csv'  # Modify accordingly
    OUTPUT_PATH = 'path_to_output_csv'  # Modify accordingly
    normalize_temperature_data(INPUT_PATH, OUTPUT_PATH)
