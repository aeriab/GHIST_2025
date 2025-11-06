import numpy as np
import sys
from tqdm import tqdm # A nice progress bar

# Import the functions from your other script
# This requires your script to be saved as 'haplotype_sorter.py'
try:
    import haplotype_sorter
except ImportError:
    print("Error: Could not find 'haplotype_sorter.py'.")
    print("Please save the script you provided as 'haplotype_sorter.py' in this directory.")
    sys.exit(1)

# --- Configuration ---
INPUT_NPY_FILE = 'GHIST_data.npy' # The file from our previous step
OUTPUT_NPY_FILE = 'sorted_GHIST_data.npy'
SORT_ORDERING = 'rows_freq' # Use 'rows_freq' or 'rows_dist'
# ---------------------

print(f"Loading data from {INPUT_NPY_FILE}...")
# Load the big file, e.g., shape (1468, 50, 102)
all_data = np.load(INPUT_NPY_FILE)

# Get the dimensions
num_images, num_samples, num_sites = all_data.shape
print(f"Loaded {num_images} images, each with shape ({num_samples}, {num_sites}).")

# Create a new array to store the sorted results
# We use np.int8 for memory efficiency, as your data is 0s and 1s
sorted_all_data = np.zeros_like(all_data, dtype=np.int8)

print(f"Sorting all {num_images} images by '{SORT_ORDERING}'...")

# Loop through each "image" in the big array
for i in tqdm(range(num_images)):
    # 1. Get the i-th image (shape 50, 102)
    image_data = all_data[i]
    
    # 2. Add the temporary channel (just as you proposed)
    # This creates the (50, 102, 2) array that sort_haplotypes expects.
    # The sort function uses the first channel (index 0) for clustering.
    mut_array_with_channel = np.zeros((num_samples, num_sites, 2), dtype=np.int8)
    mut_array_with_channel[..., 0] = image_data
    
    # 3. Pass it to the sorting function
    # Your 'sort_haplotypes' function modifies the array in-place
    try:
        haplotype_sorter.sort_haplotypes(mut_array_with_channel, ordering=SORT_ORDERING)
    except Exception as e:
        print(f"\nError sorting image {i}: {e}")
        # This can happen if, e.g., an image is all zeros
        print("Saving unsorted image and continuing...")
        sorted_all_data[i] = image_data
        continue

    # 4. Remove the temporary channel
    # We take the sorted data from the first channel (index 0)
    sorted_image = mut_array_with_channel[..., 0]
    
    # 5. Store the sorted image in our new big array
    sorted_all_data[i] = sorted_image

# 6. Save the final sorted data to a new file
np.save(OUTPUT_NPY_FILE, sorted_all_data)

print("---")
print(f"Success! All images have been sorted.")
print(f"Original file: {INPUT_NPY_FILE}")
print(f"Sorted file saved to: {OUTPUT_NPY_FILE}")
print(f"Final shape: {sorted_all_data.shape}")