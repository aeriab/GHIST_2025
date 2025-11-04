import numpy as np
import sys
import os

def convert_tsv_to_padded_npy(input_tsv_path, output_npy_path):
    """
    Converts a GHIST-formatted TSV file into a padded NumPy array.

    The script reads a TSV file, expecting:
    - A header row (skipped)
    - 201 data rows
    - Column 0: Site number (skipped)
    - Columns 1-49: Haplotype data (49 chromosomes)

    It loads this data, transposes it to (49, 201), pads it with 51 rows
    of zeros (ancestral) to create a (100, 201) array, and saves it
    as a .npy file.
    """
    
    # --- 1. Define expected shape ---
    # 49 real chromosomes + 51 padding chromosomes
    final_num_rows = 100
    
    # 201 sites as specified
    final_num_cols = 201 
    
    # Number of real chromosomes from the file
    real_chromosomes = 49
    
    # Calculate number of padding rows needed
    padding_rows = final_num_rows - real_chromosomes

    print(f"Loading TSV file: {input_tsv_path}")
    
    try:
        # --- 2. Load the data using np.loadtxt ---
        # We skip the first row (header)
        # We use columns 1 through 49 (Python index)
        # This corresponds to the 2nd to 50th column in the file
        data = np.loadtxt(
            input_tsv_path,
            delimiter='\t',
            skiprows=1,
            usecols=range(1, real_chromosomes + 1),
            dtype=int
        )
        
        # At this point, data shape is (num_sites, num_chromosomes), e.g., (201, 49)
        print(f"Loaded data with shape: {data.shape}")

        # Check if the number of sites matches our expectation
        if data.shape[0] != final_num_cols:
            print(f"Warning: Expected {final_num_cols} sites (rows) in the file, but found {data.shape[0]}.")
            print(f"Using the first {final_num_cols} sites.")
            # Truncate or pad sites if necessary, though truncation is most likely
            
            # Create a new array of target site shape
            new_data = np.zeros((final_num_cols, real_chromosomes), dtype=int)
            
            # Find how many sites to copy over (the smaller of the two)
            sites_to_copy = min(data.shape[0], final_num_cols)
            
            # Copy data
            new_data[:sites_to_copy, :] = data[:sites_to_copy, :]
            data = new_data
            
            print(f"Data reshaped to: {data.shape}")


        # --- 3. Transpose the data ---
        # We want (chromosomes, sites), so (49, 201)
        data_transposed = data.T
        print(f"Transposed shape: {data_transposed.shape}")

        # --- 4. Create the padding array ---
        # Create an array of all zeros for the 51 padding chromosomes
        padding_array = np.zeros((padding_rows, final_num_cols), dtype=int)
        print(f"Created padding array with shape: {padding_array.shape}")

        # --- 5. Stack the real data and padding ---
        # np.vstack stacks them vertically (adds rows)
        final_array = np.vstack((data_transposed, padding_array))
        print(f"Final array shape: {final_array.shape}")

        # --- 6. Save the final array ---
        np.save(output_npy_path, final_array)
        print(f"Successfully saved final array to: {output_npy_path}")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure the file path is correct and the file is a valid TSV.")

def main():
    if len(sys.argv) != 3:
        print("Usage: python process_tsv_to_npy.py <input_tsv_file> <output_npy_file>")
        print("Example: python process_tsv_to_npy.py chr21_data.tsv chr21_image_001.npy")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        sys.exit(1)

    # Ensure output path ends with .npy
    if not output_path.endswith('.npy'):
        output_path += '.npy'
        print(f"Adjusted output path to: {output_path}")

    convert_tsv_to_padded_npy(input_path, output_path)

if __name__ == "__main__":
    main()
