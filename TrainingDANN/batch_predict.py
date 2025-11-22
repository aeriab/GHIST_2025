### --------- load modules -------------------#
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy
import tensorflow.keras.backend as K
import os

# Define a batch size for prediction to manage memory
# The model will process this many images at a time.
PRED_BATCH_SIZE = 1468

### --------- Custom Layer Definition -------------------#
@tf.custom_gradient
def grad_reverse(x):
    y = tf.identity(x)
    def custom_grad(dy):
        return -dy
    return y, custom_grad

class GradReverse(tf.keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, x):
        return grad_reverse(x)

### --------- Custom Loss/Metric Definitions -------------------#
def custom_bce(y_true, y_pred):
    y_pred = tf.boolean_mask(y_pred, tf.not_equal(y_true, -1))
    y_true = tf.boolean_mask(y_true, tf.not_equal(y_true, -1))
    return binary_crossentropy(y_true, y_pred)

def custom_categorical_ce(y_true, y_pred):
    y_pred = tf.boolean_mask(y_pred, tf.reduce_all(tf.not_equal(y_true, -1), axis=-1))
    y_true = tf.boolean_mask(y_true, tf.reduce_all(tf.not_equal(y_true, -1), axis=-1))
    return categorical_crossentropy(y_true, y_pred)

def custom_binary_accuracy(y_true, y_pred):
     y_pred = tf.boolean_mask(y_pred, tf.not_equal(y_true, -1))
     y_true = tf.boolean_mask(y_true, tf.not_equal(y_true, -1))
     return tf.keras.metrics.binary_accuracy(y_true, y_pred)

def custom_categorical_accuracy(y_true, y_pred):
     y_pred = tf.boolean_mask(y_pred, tf.reduce_all(tf.not_equal(y_true, -1), axis=-1))
     y_true =  tf.boolean_mask(y_true, tf.reduce_all(tf.not_equal(y_true, -1), axis=-1))
     return tf.keras.metrics.categorical_accuracy(y_true, y_pred)

### --------- Model Loading Function -------------------#
def load_cnn_model_weights(path_model, path_weights):
    # Register all custom objects
    custom_objects = {
        'GradReverse': GradReverse,
        'custom_bce': custom_bce,
        'custom_categorical_ce': custom_categorical_ce,
        'custom_binary_accuracy': custom_binary_accuracy,
        'custom_categorical_accuracy': custom_categorical_accuracy
    }
    
    # Load model architecture from JSON file
    if not os.path.exists(path_model):
        print(f"Error: Model JSON file not found at {path_model}")
        sys.exit(1)
    if not os.path.exists(path_weights):
        print(f"Error: Model weights file not found at {path_weights}")
        sys.exit(1)

    with open(path_model, 'r') as f:
      model = model_from_json(f.read(), custom_objects=custom_objects)
    
    # Load model weights from HDF5 file
    model.load_weights(path_weights)
    return model

### --------- Main Inference Block -------------------#
if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python batch_predict.py <model.json> <model.weights.h5> <images.npy> <results.txt>")
        sys.exit(1)

    model_json_path = sys.argv[1]
    model_weights_path = sys.argv[2]
    image_npy_path = sys.argv[3]
    output_results_path = sys.argv[4]

    # 1. Load the trained model
    print(f"Loading model from {model_json_path} and {model_weights_path}...")
    model = load_cnn_model_weights(model_json_path, model_weights_path)
    model.summary() # Print model structure
    
    # Get expected input shape from the model (e.g., (None, 99, 100, 1))
    expected_shape_hw = model.input_shape[1:3] # (H, W)
    print(f"\nModel expects input shape (H, W) of: {expected_shape_hw}")


    # 2. Load and preprocess the image batch
    print(f"Loading image batch from {image_npy_path}...")
    if not os.path.exists(image_npy_path):
        print(f"Error: Input images file not found at {image_npy_path}")
        sys.exit(1)
        
    img_batch = np.load(image_npy_path)
    
    # Input should be 3D: (N, H, W)
    if len(img_batch.shape) != 3:
        print(f"Error: Expected input file to be 3D (N, H, W), but got shape {img_batch.shape}")
        sys.exit(1)
        
    num_images = img_batch.shape[0]
    input_shape_hw = img_batch.shape[1:3]
    
    print(f"Loaded {num_images} images with shape (H, W): {input_shape_hw}")
    
    # --- Shape Validation ---
    if input_shape_hw != expected_shape_hw:
        print(f"CRITICAL ERROR: Shape mismatch!")
        print(f"Model expects images of shape {expected_shape_hw}")
        print(f"Your .npy file provided images of shape {input_shape_hw}")
        sys.exit(1)

    # Preprocess to match model input shape (N, H, W, Channels)
    # Your training script adds a channel dimension, so we do the same.
    img_processed = np.expand_dims(img_batch, axis=-1) # Shape -> (N, H, W, 1)

    print(f"Total input shape for model: {img_processed.shape}")

    # 3. Perform prediction
    # model.predict() will automatically handle the batching for us
    print(f"Running predictions with batch size {PRED_BATCH_SIZE}...")
    # Your model has two outputs: [classifier, discriminator]
    prediction = model.predict(img_processed, batch_size=PRED_BATCH_SIZE)
    
    # We only care about the first output (the classifier)
    # This will be an array of shape (N, 1)
    classifier_output = prediction[0]
    
    # 4. Interpret and save results
    print(f"Saving {num_images} results to {output_results_path}...")
    threshold = 0.5
    
    with open(output_results_path, 'w') as f:
        # Write the header
        f.write("Image_Index\tLabel\tScore\n")
        
        # Loop through each prediction
        for i in range(num_images):
            # Get the single prediction score
            score = classifier_output[i][0] 
            
            if score > threshold:
                label = "SWEEP"
            else:
                label = "NEUTRAL"
            
            # Write the result to the file
            f.write(f"{i}\t{label}\t{score:.4f}\n")

    print("Done.")