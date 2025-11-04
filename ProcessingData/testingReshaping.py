import numpy as np
import sys

data = np.load('HS_sims.npy')

new_data = data[..., 0]

# 3. Save the new array to a new file
np.save('sweeps.npy', new_data)

# Optional: Print to confirm the shapes
print(f"Original shape: {data.shape}")
print(f"New shape: {new_data.shape}")