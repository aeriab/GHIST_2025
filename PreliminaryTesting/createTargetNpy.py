import numpy as np

# --- Parameters ---
input_tsv_file = 'GHIST_2025_singlesweep.21.testing.haplotype.tsv'  # The name of your input TSV file
output_npy_file = 'GHIST_data.npy' # The name of the .npy file to be saved

# These are the dimensions for your final "image"
# (1468 images, 50 chromosomes, 102 sites per image)
num_images = 1468
sites_per_image = 102
num_chromosomes = 50

# --- Script ---

print(f"Loading data from {input_tsv_file}...")
# 1. Load the data.
# We skip the first row (header) and specify the tab delimiter.
# We use dtype=np.int8 since you said the values are 0 or 1.
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

# 2. Select only the 50 chromosome columns (skip the first "site" column)
# all_data is (149768, 51), we want (149768, 50)
chromosome_data = all_data[:, 1:]
print(f"Chromosome data shape: {chromosome_data.shape}")

# 3. Truncate the data to the exact number of sites we need.
# 1468 images * 102 sites/image = 149736 total sites
total_sites_needed = num_images * sites_per_image
# This selects the first 149,736 rows
truncated_data = chromosome_data[:total_sites_needed, :]
print(f"Truncated data shape: {truncated_data.shape}")

# 4. Reshape the data.
# The current shape is (149736, 50), which is (Total_Sites, Chromosomes).
# We want to group the sites into images.
# This gives (Num_Images, Sites_per_Image, Chromosomes)
reshaped_data = truncated_data.reshape(
    num_images,
    sites_per_image,
    num_chromosomes
)
print(f"Reshaped data: {reshaped_data.shape} (Images, Sites, Chromosomes)")

# 5. Transpose to get the final desired shape.
# We want (Num_Images, Chromosomes, Sites_per_Image)
# This swaps the last two axes.
final_data = reshaped_data.transpose(0, 2, 1)

# 6. Save the final .npy file
np.save(output_npy_file, final_data)

print("---")
print(f"Successfully saved data with final shape: {final_data.shape}")
print(f"Output file: {output_npy_file}")