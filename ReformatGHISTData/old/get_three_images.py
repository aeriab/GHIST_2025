import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

# --- Configuration ---
input_npy_path = 'sorted_growth_bg_21.npy'  # Replace with your full file path
output_npy_path = 'sorted_growth_bg_21_samples.npy'

# 1. Load the file (in read-only mode to save memory)
data = np.load(input_npy_path, mmap_mode='r')

# 2. Extract the 1st (index 0), 2nd (index 1), and last (index -1)
# passing a list of integers returns a new array with those specific rows
extracted_data = data[[0, 1, -1]]

# 3. Save the new file
np.save(output_npy_path, extracted_data)

print("--- extraction complete ---")
print(f"Original shape: {data.shape}")
print(f"New file shape: {extracted_data.shape}")






# 1. Load the data
# Shape should be (3, 50, 102)
data = np.load(output_npy_path)

# 2. Setup the Plot: 1 row, 3 columns
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Define a discrete colormap for Genotypes (0, 1, 2)
# 0 = White (Background/Major), 1 = Gray (Het), 2 = Black (Hom Alt)
# You can change these colors to whatever you prefer (e.g., ['white', 'orange', 'red'])
cmap = colors.ListedColormap(['white', 'gray', 'black'])
bounds = [-0.5, 0.5, 1.5, 2.5]
norm = colors.BoundaryNorm(bounds, cmap.N)

titles = ['First Image', 'Second Image', 'Last Image']
for i, ax in enumerate(axes):
    # Plot the image
    img = ax.imshow(data[i], cmap=cmap, norm=norm, aspect='auto')
    
    # Formatting
    ax.set_title(f"{titles[i]}\n(Index {i})")
    ax.set_xlabel("SNPs (102)")
    ax.set_ylabel("Individuals (50)")
    
    # Remove tick clutter for cleaner look
    ax.tick_params(left=False, bottom=False)

# Add a shared colorbar on the right
cbar = fig.colorbar(img, ax=axes.ravel().tolist(), ticks=[0, 1, 2], shrink=0.5)
cbar.ax.set_yticklabels(['0 (Ref)', '1 (Het)', '2 (Alt)'])

plt.suptitle(f"Genomic Image Samples (Shape: {data.shape})", fontsize=16)
plt.show()