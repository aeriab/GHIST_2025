import sys
import os
import numpy as np
# We need tensorflow to load the model, even if it's via a custom function
import tensorflow as tf

# --- 1. Set up path to load your custom function ---
# This is copied directly from your script
Utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '/u/project/ngarud/Garud_lab/DANN/Utils/'))
sys.path.append(Utils_path)
from CNN_multiclass_data_mergeSims_A100 import load_cnn_model

# --- 2. Define model and image paths ---
# Use the model name from your prompt
model_name = 'CNN_color_multiclass_sims_trained' 
# This is the npy file we created in the previous step
image_path = 'first_test_npy_file.npy'

# --- 3. Load the model ---
print(f"Loading model: {model_name}...")
# This uses your custom function to load the .json and .weights.h5
model = load_cnn_model(model_name)

# You can uncomment this line to see the model's architecture
# and confirm its expected input shape
# model.summary()

# --- 4. Load the entire batch of images ---
print(f"Loading image batch: {image_path}...")
# Load the entire (20773, 100, 201, 2) array
# This is already a batch, no np.expand_dims needed
image_batch = np.load(image_path)
print(f"Image batch shape for prediction: {image_batch.shape}")

# --- 5. Make the prediction ---
print("Running prediction...")
# This returns the probabilities for each class
# Run prediction on the whole batch, processing 32 images at a time for memory efficiency
prediction_probabilities = model.predict(image_batch, batch_size=32)

# --- 6. Interpret results and save to file ---
print("Interpreting all predictions...")
# prediction_probabilities shape is (20773, 3)

# Define the class labels (based on your PR script's plotting)
class_labels = ["Neutral", "Hard sweep", "Soft sweep"]
output_filename = "predictions.txt"

# Get the index of the highest probability for *each* image in the batch
# This will be an array of shape (20773,)
predicted_indices = np.argmax(prediction_probabilities, axis=1)

# Get the confidence score (the highest probability) for *each* image
confidences = np.max(prediction_probabilities, axis=1)

print(f"Saving {len(predicted_indices)} predictions to {output_filename}...")

# Open the output file and write the results
with open(output_filename, 'w') as f:
    f.write("Image_Index,Predicted_Label,Confidence\n") # Write a header
    
    for i in range(len(predicted_indices)):
        predicted_label = class_labels[predicted_indices[i]]
        confidence_score = confidences[i]
        
        # Write the line
        f.write(f"{i},{predicted_label},{confidence_score:.6f}\n")

print("\n--- Prediction Complete ---")
print(f"Results saved to {output_filename}.")