import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import numpy as np
import combine_data
import tf2onnx
import os

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
def build_lstm_model(input_shape, forecast_horizon=14):
    model = Sequential()

    # Add the first LSTM layer
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))

    # Add a second LSTM layer
    model.add(LSTM(50, activation='relu'))
    model.add(Dropout(0.2))

    # Add Dense layer for prediction of Min and Max flow for the forecast horizon
    model.add(Dense(forecast_horizon * 2, activation="linear"))
    model.add(tf.keras.layers.Reshape((forecast_horizon, 2)))  # Reshape to [batch_size, forecast_horizon, 2]

    model.compile(optimizer='adam', loss='mse')
    return model
    
def train_lstm_model(data, epochs=10, batch_size=32, validation_split=0.2):
    X, Y = reshape_data_for_lstm(data)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=validation_split)

    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1)

    return model, history 


if __name__ == '__main__':
    data = combine_data.main()
    if isinstance(data, str):
        print("Error: Data is recognized as string. Expected pandas DataFrame.")
        exit(1)  # Exit the script

    # Assuming columns 'Flow' and 'Temperature' are present in the combined data
    model, history = train_lstm_model(data)
    save_as_onnx(model)




