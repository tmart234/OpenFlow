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
def build_lstm_model_with_embedding(input_shape, forecast_horizon=14, num_stations=1000, embedding_dim=5):
    # Input layers for historical data and station IDs
    historical_input = Input(shape=input_shape, name='historical_input')
    station_input = Input(shape=(1,), name='station_input')
    
    # Embedding for station IDs
    station_embedding = Embedding(input_dim=num_stations, output_dim=embedding_dim, input_length=1)(station_input)
    station_embedding = tf.keras.layers.Reshape((embedding_dim,))(station_embedding)
    
    # Repeat embedding for each timestep and concatenate with historical data
    station_embedding_repeated = tf.keras.layers.RepeatVector(input_shape[0])(station_embedding)
    combined_input = Concatenate()([historical_input, station_embedding_repeated])
    
    # LSTM layers to process combined input
    lstm_out = LSTM(50, activation='relu', return_sequences=True)(combined_input)
    lstm_out = Dropout(0.2)(lstm_out)
    lstm_out = LSTM(50, activation='relu')(lstm_out)
    lstm_out = Dropout(0.2)(lstm_out)
    
    # Dense layer to predict flow, reshaped to forecast horizon
    dense_out = Dense(forecast_horizon * 2, activation="linear")(lstm_out)
    output = tf.keras.layers.Reshape((forecast_horizon, 2))(dense_out)
    
    # Create and compile the model
    model = Model(inputs=[historical_input, station_input], outputs=output)
    model.compile(optimizer='adam', loss='mse')
    
    return model
    
def train_lstm_model(data, epochs=10, batch_size=32, validation_split=0.2):
    # Extract station IDs and map them to integers
    station_ids = data['Station ID'].values
    label_encoder = LabelEncoder()
    station_ids_encoded = label_encoder.fit_transform(station_ids)
    num_stations = num_stations or len(label_encoder.classes_)

    # Process the data for the LSTM
    X, Y = reshape_data_for_lstm(data)
    
    # The input for the model will now be a tuple consisting of the data and the station IDs
    X_train, X_test, y_train, y_test, station_ids_train, station_ids_test = train_test_split(
        X, Y, station_ids_encoded, test_size=validation_split)

    # Build the LSTM model with embedding for station IDs
    model = build_lstm_model_with_embedding(input_shape=(X_train.shape[1], X_train.shape[2]), 
                                            forecast_horizon=14, 
                                            num_stations=num_stations,  # Dynamically set
                                            embedding_dim=5)

    # Fit the model with the station IDs as additional input
    history = model.fit([X_train, station_ids_train], y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=([X_test, station_ids_test], y_test), verbose=1)

    return model, history

def main():
    data = combine_data.main()  # Your function to read and combine data
    if isinstance(data, str):
        print("Error: Data is recognized as string. Expected pandas DataFrame.")
        exit(1)

    # Extract unique station IDs from the data
    station_ids = data['Station ID'].unique()
    num_stations = len(station_ids)
    
    # Convert station IDs to integer indices
    label_encoder = LabelEncoder()
    data['Station ID'] = label_encoder.fit_transform(data['Station ID'])

    # Now 'data' contains a 'Station ID' column with integer indices
    # and you have the number of unique stations
    model, history = train_lstm_model(data, num_stations=num_stations)
    save_as_h5(model)

# Call main
if __name__ == '__main__':
    main()




