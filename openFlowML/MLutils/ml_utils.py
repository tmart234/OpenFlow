import coremltools as ct
import tensorflow as tf
import sys
import os
import logging

# Check if the root logger already has handlers (configured in another module)
if not logging.getLogger().hasHandlers():
    # If not, set up basic logging configuration
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get logger for this module
logger = logging.getLogger(__name__)

def convert_model(model_path, mlmodel_output_path, input_shape):
    logger.info(f"Starting model conversion: {model_path} -> {mlmodel_output_path}")
    
    # Check if the input model file exists
    if not os.path.isfile(model_path):
        logger.error(f"Input model file not found: {model_path}")
        raise FileNotFoundError(f"Input model file not found: {model_path}")

    # Check if the input model file is empty
    if os.path.getsize(model_path) == 0:
        logger.error(f"Input model file is empty: {model_path}")
        raise ValueError(f"Input model file is empty: {model_path}")

    # Load the Keras model using TensorFlow
    logger.info("Loading Keras model...")
    model = tf.keras.models.load_model(model_path)

    # Convert the model to Core ML format
    logger.info("Converting model to Core ML format...")
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
    logger.info(f"Saving Core ML model to {mlmodel_output_path}...")
    mlmodel.save(mlmodel_output_path)

    # Check if the output model file was created successfully
    if not os.path.isfile(mlmodel_output_path):
        logger.error(f"Failed to create output model file: {mlmodel_output_path}")
        raise RuntimeError(f"Failed to create output model file: {mlmodel_output_path}")

    # Check if the output model file is empty
    if os.path.getsize(mlmodel_output_path) == 0:
        logger.error(f"Output model file is empty: {mlmodel_output_path}")
        raise RuntimeError(f"Output model file is empty: {mlmodel_output_path}")

    logger.info("Model conversion completed successfully.")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        logger.error("Incorrect number of arguments.")
        logger.info("Usage: python ml_utils.py <model_path> <mlmodel_output_path> <input_shape>")
        sys.exit(1)

    model_path = sys.argv[1]
    mlmodel_output_path = sys.argv[2]
    input_shape = eval(sys.argv[3])  # Convert string input to tuple

    try:
        convert_model(model_path, mlmodel_output_path, input_shape)
        logger.info(f"Model converted successfully. Output saved to: {mlmodel_output_path}")
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Value error: {str(e)}")
        sys.exit(1)
    except RuntimeError as e:
        logger.error(f"Runtime error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)