import sys
import os
import numpy as np
# We need tensorflow to load the model, even if it's via a custom function
import tensorflow as tf
import keras
from keras.metrics import binary_accuracy


@keras.saving.register_keras_serializable()
def custom_binary_accuracy(y_true, y_pred):
    return binary_accuracy(y_true, y_pred)


# --- 1. Set up path to load your custom function ---
# This is copied directly from your script
Utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '/u/project/ngarud/Garud_lab/DANN/Utils/'))
sys.path.append(Utils_path)
from CNN_multiclass_data_mergeSims_A100 import load_cnn_model

# --- 2. Define model and image paths ---
# Use the model name from your prompt
model_name = 'DANN_multiclass' 
# This is the npy file we created in the previous step
image_path = 'GHIST_converted_2025_singlesweep.npy'

# --- 3. Load the model ---
print(f"Loading model: {model_name}...")
# This uses your custom function to load the .json and .weights.h5
model = load_cnn_model(model_name)

# You can uncomment this line to see the model's architecture
# and confirm its expected input shape
# model.summary()

# --- 4. Load and process the image ---
print(f"Loading image: {image_path}...")
# Load the original (100, 201) data
original_data = np.load(image_path)
print(f"Original data shape: {original_data.shape}") # Should be (100, 201)

# Create the target (100, 201, 2) array, initialized to zeros
target_shape = original_data.shape + (2,)
image_with_channels = np.zeros(target_shape, dtype=original_data.dtype)

# Copy the loaded data into the first channel (index 0)
image_with_channels[..., 0] = original_data
# The second channel (index 1) remains zeros

print(f"Image with channels shape: {image_with_channels.shape}") # Will be (100, 201, 2)


# --- 5. Make the prediction ---
print("Running prediction...")

# Add a batch dimension to create a shape of (1, 100, 201, 2)
# This is what model.predict expects
image_batch = np.expand_dims(image_with_channels, axis=0)
print(f"Image batch shape for prediction: {image_batch.shape}") # Will be (1, 100, 201, 2)

# This returns the probabilities for each class
# We are only predicting one image, so batch_size=32 is not needed.
prediction_probabilities = model.predict(image_batch)

# --- 6. Interpret result and save to file ---
print("Interpreting prediction...")
# prediction_probabilities shape will be (1, 3)

# Define the class labels (based on your PR script's plotting)
class_labels = ["Neutral", "Hard sweep", "Soft sweep"]
output_filename = "prediction.txt"

# Get the prediction from the first (and only) item in the batch
probabilities_for_one_image = prediction_probabilities[0]

# Get the index of the highest probability
predicted_index = np.argmax(probabilities_for_one_image)
predicted_label = class_labels[predicted_index]

# Get the confidence score (the highest probability)
confidence = np.max(probabilities_for_one_image)

print(f"Saving prediction to {output_filename}...")

# Open the output file and write the single result
with open(output_filename, 'w') as f:
    f.write("Predicted_Label,Confidence\n") # Write a header
    f.write(f"{predicted_label},{confidence:.6f}\n")

print("\n--- Prediction Complete ---")
print(f"Prediction: {predicted_label} (Confidence: {confidence:.6f})")
print(f"Result saved to {output_filename}.")

# Commiting as aeriab