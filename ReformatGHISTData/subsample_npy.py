import numpy as np
import argparse
import sys

# Example usage: python subsample_npy.py input.npy output.npy

def subsample_npy(input_file, output_file, num_samples=10):
    try:
        # Load the original numpy array
        # mmap_mode='r' is useful if the file is very large, so it doesn't load everything into RAM at once
        data = np.load(input_file, mmap_mode='r')
        
        total_images = data.shape[0]
        print(f"Original shape: {data.shape}")

        if total_images < num_samples:
            print(f"Error: Input file has fewer images ({total_images}) than the requested samples ({num_samples}).")
            sys.exit(1)

        # Calculate evenly spaced indices
        # np.linspace generates 'num_samples' numbers evenly spaced between 0 and total_images-1
        indices = np.linspace(0, total_images - 1, num_samples, dtype=int)
        
        print(f"Selected indices: {indices}")

        # Extract the samples using the calculated indices
        # We copy to force it into memory if we used mmap_mode
        subsampled_data = data[indices].copy()

        print(f"New shape: {subsampled_data.shape}")

        # Save the new file
        np.save(output_file, subsampled_data)
        print(f"Successfully saved to {output_file}")

    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # You can run this directly or import the function
    # Example usage: python subsample_npy.py input.npy output.npy
    
    if len(sys.argv) < 3:
        print("Usage: python subsample_npy.py <input_path> <output_path>")
        # Creating a dummy file for demonstration if run without args in an IDE
        # Remove this block in production usage
        print("\n--- Demo Mode ---")
        dummy_input = "demo_input.npy"
        dummy_output = "demo_output.npy"
        
        # Create a dummy array of shape (74211, 50, 102)
        print("Creating dummy file for demonstration...")
        dummy_data = np.zeros((74211, 50, 102), dtype=np.float16)
        np.save(dummy_input, dummy_data)
        
        subsample_npy(dummy_input, dummy_output)
    else:
        subsample_npy(sys.argv[1], sys.argv[2])