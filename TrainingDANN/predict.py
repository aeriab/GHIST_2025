### --------- load modules -------------------#
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy
import tensorflow.keras.backend as K

# Input the model json, model weights, and the shape (50,102) npy file for the given haplotype image

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
    with open(path_model, 'r') as f:
      model = model_from_json(f.read(), custom_objects=custom_objects)
    
    # Load model weights from HDF5 file
    model.load_weights(path_weights)
    return model

### --------- Main Inference Block -------------------#
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python predict.py <model.json> <model.weights.h5> <image.npy>")
        sys.exit(1)

    model_json_path = sys.argv[1]
    model_weights_path = sys.argv[2]
    image_npy_path = sys.argv[3]

    # 1. Load the trained model
    print(f"Loading model from {model_json_path} and {model_weights_path}...")
    model = load_cnn_model_weights(model_json_path, model_weights_path)
    # model.summary() # Uncomment to verify the model structure

    # 2. Load and preprocess the image
    print(f"Loading image from {image_npy_path}...")
    # Load the single image (assuming it's a 2D array [H, W])
    img = np.load(image_npy_path)

    # Preprocess to match model input shape (Batch, H, W, Channels)
    # Your training script adds a channel dimension, so we do the same.
    img_processed = np.expand_dims(img, axis=-1)   # Shape -> (H, W, 1)
    # Add a batch dimension for a single prediction
    img_processed = np.expand_dims(img_processed, axis=0) # Shape -> (1, H, W, 1)

    print(f"Input image shape for model: {img_processed.shape}")

    # 3. Perform prediction
    # Your model has two outputs: [classifier, discriminator]
    prediction = model.predict(img_processed)
    
    # We only care about the first output (the classifier)
    classifier_output = prediction[0]
    
    # The classifier output is a sigmoid (0 to 1)
    # From your data generator: 0.0 = neutral, 1.0 = sweep
    score = classifier_output[0][0] # Get the single prediction score
    
    # 4. Interpret the result
    threshold = 0.5
    if score > threshold:
        print(f"\nPrediction: SWEEP (Score: {score:.4f})")
    else:
        print(f"\nPrediction: NEUTRAL (Score: {score:.4f})")