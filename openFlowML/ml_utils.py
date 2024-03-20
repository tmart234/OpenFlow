import coremltools as ct
import tensorflow as tf
import sys
import os

def convert_model(model_path, mlmodel_output_path, input_shape):
    # Check if the input model file exists
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Input model file not found: {model_path}")

    # Check if the input model file is empty
    if os.path.getsize(model_path) == 0:
        raise ValueError(f"Input model file is empty: {model_path}")

    # Load the Keras model using TensorFlow
    model = tf.keras.models.load_model(model_path)

    # Convert the model to Core ML format
    if len(input_shape) == 3:
        # If input shape is already 3D, use it directly
        mlmodel = ct.convert(model, inputs=[ct.TensorType(shape=input_shape)])
    else:
        # If input shape is 2D, add a sequence length dimension of 1
        input_shape_with_sequence = (1,) + tuple(input_shape)
        mlmodel = ct.convert(model, inputs=[ct.TensorType(shape=input_shape_with_sequence)])


    # Check if the output directory exists, create it if necessary
    output_dir = os.path.dirname(mlmodel_output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Save the Core ML model
    mlmodel.save(mlmodel_output_path)

    # Check if the output model file was created successfully
    if not os.path.isfile(mlmodel_output_path):
        raise RuntimeError(f"Failed to create output model file: {mlmodel_output_path}")

    # Check if the output model file is empty
    if os.path.getsize(mlmodel_output_path) == 0:
        raise RuntimeError(f"Output model file is empty: {mlmodel_output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python ml_utils.py <model_path> <mlmodel_output_path> <input_shape>")
        sys.exit(1)

    model_path = sys.argv[1]
    mlmodel_output_path = sys.argv[2]
    input_shape = eval(sys.argv[3])  # Convert string input to tuple

    try:
        convert_model(model_path, mlmodel_output_path, input_shape)
        print(f"Model converted successfully. Output saved to: {mlmodel_output_path}")
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        sys.exit(1)
