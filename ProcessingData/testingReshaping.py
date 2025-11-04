import numpy as np
import sys

data = np.load('N_sims.npy')

new_data = data[..., 0]

# 3. Save the new array to a new file
np.save('neutrals.npy', new_data)

# Optional: Print to confirm the shapes
print(f"Original shape: {data.shape}")
print(f"New shape: {new_data.shape}")