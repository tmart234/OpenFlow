# convert_to_coreml.py
import coremltools as ct
import sys

def convert_model(model_path, mlmodel_output_path, input_shape):
    # Load the Keras model
    model = ct.models.keras.load_model(model_path)

    # Convert the model to Core ML format
    mlmodel = ct.convert(model, inputs=[ct.TensorType(shape=input_shape)])

    # Save the Core ML model
    mlmodel.save(mlmodel_output_path)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python convert_to_coreml.py <h5_model_path> <mlmodel_output_path> <input_shape>")
    else:
        model_path = sys.argv[1]
        mlmodel_output_path = sys.argv[2]
        input_shape = eval(sys.argv[3])  # Convert string input to tuple
        convert_model(h5_model_path, mlmodel_output_path, input_shape)
