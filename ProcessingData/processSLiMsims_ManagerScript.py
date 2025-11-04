import numpy as np
import subprocess
import glob
import os
import sys


# --------------------- Define parameters -------------------------   
OUTPUT_FILE = sys.argv[1] # Output numpy file name ex: all_sims.npy
NUM_SAMPS = int(sys.argv[2]) # Number of samples per simulation
WINDOW_SIZE = int(sys.argv[3]) # Size of the window

input_files = sorted(glob.glob("/u/project/ngarud/Garud_lab/DANN_maya_slimulations/dann_slimulations_10246482/hard/*.txt"))  # adjust path if needed
NUM_SIMS = len(input_files)
DTYPE = np.float32
SIM_SHAPE = (NUM_SAMPS, WINDOW_SIZE, 2)
print(f"Found {NUM_SIMS} simulation files to process.")
# -------------------------------------------------------------------

# Preallocate array file
big_array = np.lib.format.open_memmap(
    OUTPUT_FILE, dtype=DTYPE, mode="w+", shape=(NUM_SIMS,) + SIM_SHAPE
)

# Loop over input files and process each one
for i, infile in enumerate(input_files):
    print(f"Processing file {i + 1}/{NUM_SIMS}: {infile}")
    # Call the external script to process the file
    subprocess.run(
        [
            "python",
            "processSLiMsims.py",
            infile,
            OUTPUT_FILE,
            str(NUM_SAMPS),
            str(WINDOW_SIZE),
            str(i)
        ],
        check=True,
    )

del big_array  # flush to disk
print(f"All results saved in {OUTPUT_FILE}")