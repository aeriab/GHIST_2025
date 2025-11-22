import numpy as np

# --- Parameters ---
input_tsv_file = 'GHIST_2025_singlesweep.21.testing.haplotype.tsv'  # The name of your input TSV file
output_npy_file = 'GHIST_data_sliding.npy' # The name of the .npy file to be saved

# These are the dimensions for your final "image"
window_height = 102     # (sites_per_image)
slide_step = 10         # The number of sites to move the window by each time

# --- Script ---

print(f"Loading data from {input_tsv_file}...")
# 1. Load the data.
# We skip the first row (header) and specify the tab delimiter.
# We use dtype=np.int8 since the values are 0 or 1.
# This is very memory-efficient.
try:
    # This loads the entire file, including the site number column
    all_data = np.loadtxt(
        input_tsv_file,
        delimiter='\t',
        skiprows=1,
        dtype=np.int8
    )
except Exception as e:
    print(f"Error loading file: {e}")
    print("Please check that your file is a valid TSV and the path is correct.")
    exit()

print(f"Raw data loaded with shape: {all_data.shape}")

# 2. Select only the chromosome columns (skip the first "site" column)
# all_data is (e.g., 149768, 51), we want (149768, 50)
chromosome_data = all_data[:, 1:]
total_sites = chromosome_data.shape[0]
num_chromosomes = chromosome_data.shape[1]
print(f"Chromosome data shape: {chromosome_data.shape} (Total Sites, Chromosomes)")

# --- New Sliding Window Logic ---

# 3. Calculate the number of windows (images)
# Formula: floor((total_items - window_size) / stride) + 1
# This calculates how many full windows of size `window_height`
# we can fit by moving `slide_step` at a time.
num_images = np.floor((total_sites - window_height) / slide_step).astype(int) + 1

# Example calculation:
# (149768 sites - 102 site window) = 149666 sites to slide over
# 149666 / 10 slide_step = 14966.6 -> floor is 14966
# 14966 steps + the first window = 14967 images
print(f"Calculated {num_images} images based on a window of {window_height} and slide of {slide_step}.")

# 4. Create the windows by iterating
print("Creating sliding windows...")
# We pre-allocate the final array for memory efficiency instead of appending
# This is much faster than using all_windows.append()
final_data = np.zeros((num_images, num_chromosomes, window_height), dtype=np.int8)

for i in range(num_images):
    # Calculate start and end index for the slice
    start_index = i * slide_step
    end_index = start_index + window_height
    
    # Extract the window (shape: [window_height, num_chromosomes])
    # e.g., (102, 50)
    window_data = chromosome_data[start_index:end_index, :]
    
    # 5. Transpose to get the final desired shape (Chromosomes, Sites)
    # (102, 50) -> (50, 102)
    window_transposed = window_data.transpose(1, 0)

    # 6. Add the transposed window to our pre-allocated array
    final_data[i] = window_transposed

# 7. Save the final .npy file
np.save(output_npy_file, final_data)

print("---")
print(f"Successfully saved data with final shape: {final_data.shape}")
print(f"Output file: {output_npy_file}")