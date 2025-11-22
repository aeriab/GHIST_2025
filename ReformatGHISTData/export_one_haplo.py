import numpy as np
import sys
import os

def extract_first_slice(input_path, output_path):
    """
    Loads a .npy file, extracts the first item from the 0-th dimension,
    and saves it to a new .npy file.
    
    Args:
        input_path (str): The file path for the input .npy file.
        output_path (str): The file path to save the output .npy file.
    """
    # --- 1. Basic Input Validation ---
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at '{input_path}'")
        return

    if not input_path.endswith('.npy'):
        print(f"Warning: Input file '{input_path}' does not end with .npy")

    if not output_path.endswith('.npy'):
        print(f"Warning: Output file '{output_path}' does not end with .npy. Adding it.")
        output_path += ".npy"

    try:
        # --- 2. Load Data ---
        print(f"Loading data from '{input_path}'...")
        data = np.load(input_path)
        print(f"Original data shape: {data.shape}")

        # --- 3. Validate Shape ---
        if len(data.shape) != 3 or data.shape[1] != 50 or data.shape[2] != 102:
            print(f"Error: Expected shape (N, 50, 102), but got {data.shape}")
            return
            
        if data.shape[0] < 1:
            print(f"Error: The first dimension is empty (shape[0] is 0). Cannot extract a slice.")
            return

        # --- 4. Extract Slice ---
        # Get the first slice along the "unknown" dimension (axis 0)
        first_slice = data[0]
        print(f"Extracted slice shape: {first_slice.shape}")

        # --- 5. Save New Array ---
        np.save(output_path, first_slice)
        print(f"Successfully saved new slice to '{output_path}'")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_slice.py <input_file.npy> <output_file.npy>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    extract_first_slice(input_file, output_file)