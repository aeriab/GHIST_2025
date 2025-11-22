import numpy as np

# --- Parameters ---
input_tsv_file1 = 'GHIST_2025_singlesweep.growth_bg.21.testing.haplotype.tsv'  # The name of your input TSV file
input_tsv_file2 = 'GHIST_2025_multisweep.growth_bg.21.testing.haplotype.tsv'
output_npy_file = 'growth_bg.npy' # The name of the .npy file to be saved

sites_per_image = 102
num_chromosomes = 50

image_list = []

# Process the two files sequentially
for file_path in [input_tsv_file1, input_tsv_file2]:
    # Load data: skip header (row 1), grab cols 2-51 (indices 1-51)
    raw_data = np.loadtxt(file_path, delimiter='\t', skiprows=1, usecols=range(3, 53))
    
    num_snps = raw_data.shape[0]
    
    # Slide window with stride of 10
    for i in range(0, num_snps - sites_per_image + 1, 10):
        # Extract chunk: (102 rows, 50 cols)
        window = raw_data[i : i + sites_per_image, :]
        
        # Transpose to get (50 individuals, 102 SNPs) and append
        image_list.append(window.T)

# Stack all images into a single numpy array
final_data = np.array(image_list)


np.save(output_npy_file, final_data)

print("---")
print(f"Successfully saved data with final shape: {final_data.shape}")
print(f"Output file: {output_npy_file}")