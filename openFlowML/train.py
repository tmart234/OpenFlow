import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Input, Concatenate, LSTM, Dropout, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import combine_data
import tf2onnx
import os

def save_as_h5(model, output_filename="lstm_model.h5"):
    # Save the Keras model to HDF5 format
    model.save(output_filename)
    
def save_as_onnx(model, output_filename="lstm_model.onnx"):
    # Define the input signature for the model
    input_signature = (tf.TensorSpec((None, model.input_shape[1], model.input_shape[2]), tf.float32, name="input"),)
    
    # Convert the Keras model to ONNX format using tf2onnx
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature, opset=13)
    
    # Save the ONNX model to a file
    with open(output_filename, "wb") as f:
        f.write(onnx_model.SerializeToString())

# make mix/max flow prediction for next 14 days, 
# take in a window of 14 day future temps, 
# and take in window of 60 days of past flow data
def reshape_data_for_lstm(data, 
                          historical_flow_timesteps=60, 
                          forecast_temperature_timesteps=14, 
                          forecast_flow_timesteps=14):
    X, Y = [], []

    # Ensure columns exist in data
    required_columns = ["Min Flow", "Max Flow", "TMIN", "TMAX"]
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Expected '{col}' column in the data.")
    
    total_required_days = historical_flow_timesteps + forecast_temperature_timesteps + forecast_flow_timesteps

    for i in range(len(data) - total_required_days + 1):
        # Extracting historical flow data and future temperature data
        input_features = data.iloc[i:i+total_required_days][required_columns].values
        X.append(input_features)

        # Extracting the flow data (both Min and Max) for the next 14 days (the target)
        future_min_flow = data.iloc[i+historical_flow_timesteps:i+historical_flow_timesteps+forecast_flow_timesteps]["Min Flow"].values
        future_max_flow = data.iloc[i+historical_flow_timesteps:i+historical_flow_timesteps+forecast_flow_timesteps]["Max Flow"].values
        
        Y.append(np.stack([future_min_flow, future_max_flow], axis=-1))

    return np.array(X), np.array(Y)

# output a flattened prediction of shape [?, forecast_horizon * 2]
# then immediately reshaped it to the desired [?, forecast_horizon, 2].
# Reshape the data to be suitable for LSTM training
def reshape_data_for_lstm(data, historical_flow_timesteps=60, forecast_temperature_timesteps=14, forecast_flow_timesteps=14):
    X, Y = [], []
    # Define required columns
    required_columns = ["Min Flow", "Max Flow", "TMIN", "TMAX"]

    # Check if required columns exist
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Expected '{col}' column in the data.")

    # Calculate the total required days for the input window
    total_required_days = historical_flow_timesteps + forecast_temperature_timesteps

    # Loop over the dataset and prepare input-output pairs
    for i in range(len(data) - total_required_days + 1):
        # Extract input features based on required columns
        input_features = data.iloc[i:i+total_required_days][required_columns].values
        X.append(input_features)

        # Extract future flow data for target output
        future_min_flow = data.iloc[i+historical_flow_timesteps:i+historical_flow_timesteps+forecast_flow_timesteps]["Min Flow"].values
        future_max_flow = data.iloc[i+historical_flow_timesteps:i+historical_flow_timesteps+forecast_flow_timesteps]["Max Flow"].values
        Y.append(np.stack([future_min_flow, future_max_flow], axis=-1))

    return np.array(X), np.array(Y)

# Build LSTM model
def build_lstm_model(input_shape, forecast_horizon=14):
    # Define input layer for historical data
    historical_input = Input(shape=input_shape, name='historical_input')

    # Define LSTM layers with dropout for regularization
    lstm_out = LSTM(50, activation='relu', return_sequences=True)(historical_input)
    lstm_out = Dropout(0.2)(lstm_out)
    lstm_out = LSTM(50, activation='relu')(lstm_out)
    lstm_out = Dropout(0.2)(lstm_out)

    # Dense output layer to predict future flow data
    dense_out = Dense(forecast_horizon * 2, activation="linear")(lstm_out)
    output = tf.keras.layers.Reshape((forecast_horizon, 2))(dense_out)

    # Compile the model
    model = Sequential(inputs=historical_input, outputs=output)
    model.compile(optimizer='adam', loss='mse')
    
    return model

# Main function to execute the training process
def main():
    # Load and combine data (assuming this function returns a pandas DataFrame)
    data = combine_data.main()
    if isinstance(data, str):
        print("Error: Data is recognized as string. Expected pandas DataFrame.")
        exit(1)

    # Reshape data for LSTM training
    X, Y = reshape_data_for_lstm(data)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    # Build LSTM model
    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]), forecast_horizon=14)

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Save the trained model
    save_as_h5(model)


# Call main
if __name__ == '__main__':
    main()




