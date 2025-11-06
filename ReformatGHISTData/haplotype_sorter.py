import numpy as np
import sys

def parse_slim_sims(path_sims):
    """
    Parses a SLiM simulation output file to extract mutation types, positions, and genome sequences.
    
    Args:
        path_sims (str): Path to the SLiM simulation output file.
        
    Returns:
        tuple: A tuple containing three dictionaries:
            - mut_type_dict: Dictionary mapping mutation IDs to their types.
            - mut_pos_dict: Dictionary mapping mutation IDs to their positions.
            - genomes_dict: Dictionary mapping sample IDs to their genome sequences.
    """
    #parse file
    mut_type_dict = {}
    mut_pos_dict = {}
    genomes_dict = {}
    section = None
    # Open the file and read line by line
    with open(path_sims, 'r') as file:
        for line in file:
            line = line.strip()
            # Set section based on the line content
            if line == "Mutations:": # Start of mutations section
                section = "mutations"
                continue
            elif line == "Individuals:": # Start of individuals section
                section = None # No need to process individuals section
                continue
            elif line == "Genomes:": # Start of genomes section
                section = "genomes"
                continue

            if section == "mutations":
                # Process mutation lines
                muts_line = line.split()
                if len(muts_line) >= 2: # Ensure there are enough elements
                    # Extract mutation ID, type, and position
                    mut_id = muts_line[0]
                    #mut_type = muts_line[2] # use this for mutation type (e.g., 'm1', 'm2', etc.)
                    mut_type = muts_line[4] # selection coefficient
                    mut_pos = muts_line[3]
                    mut_type_dict[mut_id] = mut_type # Store mutation type
                    mut_pos_dict[mut_id] = mut_pos # Store mutation position
            elif section == "genomes":
                # Process genome lines
                genome_line = line.split()
                if len(genome_line) >= 2: # Ensure there are enough elements
                    sample_id = genome_line[0]
                    genome_sequence = genome_line[2:] # Get the mutations for individual
                    genomes_dict[sample_id] = genome_sequence # Store genome sequence

    sorted_positions = sorted(set(mut_pos_dict.values()))
    pos_to_idx_dict = {pos: idx for idx, pos in enumerate(sorted_positions)}

                    
    return mut_type_dict, mut_pos_dict,pos_to_idx_dict, genomes_dict    

def create_haplotype_matrix(mut_type_dict, mut_pos_dict, pos_to_idx_dict, genomes_dict):
    """
    Creates a haplotype matrix from genome sequences and mutation data.
    
    Args:
        mut_type_dict (dict): Dictionary mapping mutation IDs to their types.
        mut_pos_dict (dict): Dictionary mapping mutation IDs to their positions.
        pos_to_idx_dict (dict): Dictionary mapping mutation positions to their indices.
        genomes_dict (dict): Dictionary mapping sample IDs to their genome sequences.
        
        
        
    Returns:
        np.ndarray: A numpy array representing the haplotype matrix with dimensions
                    (n_samples, n_mutations, 2), where the last dimension indicates
                    syn/non-syn type.
    """

    # Initialize the haplotype matrix
    mut_array = np.zeros((len(genomes_dict), len(pos_to_idx_dict), 2), dtype=int)
        
    # Fill the numpy array with mutation data
    for i, sample_id in enumerate(genomes_dict):
        #print(i, sample_id)
        mutation_id_in_sample= genomes_dict[sample_id] # List of mutation IDs for the sample
        positions_idx = [pos_to_idx_dict[mut_pos_dict[id]] for id in mutation_id_in_sample] # Get the indices of the positions
      
        mut_array[i,positions_idx, 0] = 1 # Set the mutation presence to 1
        
        # Set the mutation type in the second channel of the array
        for id, pos_idx in zip(mutation_id_in_sample, positions_idx):
            if float(mut_type_dict[id]) != 0: # if selection coefficient not 0, set to non-synonymous
                mut_array[i, pos_idx, 1] = 1
            # If you want to use mutation type (e.g., 'm1', 'm2') instead of selection coefficient, uncomment below
            #if mut_type_dict[id] != 'm1':
            #    mut_array[i, pos_idx, 1] = 1


    return mut_array

def sample_haplotypes(mut_array, n_samples):
    """
    Samples a specified number of haplotypes from the haplotype matrix.
    
    Args:
        mut_array (np.ndarray): The haplotype matrix with dimensions (total num haps, n_mutations, 2).
        n_samples (int): The number of haplotypes to sample.
        
    Returns:
        np.ndarray: A numpy array containing the sampled haplotypes.
    """
    total_samples = mut_array.shape[0]
    random_indices = np.random.choice(total_samples, n_samples, replace=False) # Sample without replacement
    random_indices.sort()
    sampled_mut_array = mut_array[random_indices,:,:] # Select the sampled haplotypes
    
    return sampled_mut_array

def crop(mut_array, window_size, sparse=False):
    """
    Crops the haplotype matrix to a specified window size centered around the middle.
    
    Args:
        mut_array (np.ndarray): The haplotype matrix with dimensions (n_samples, n_mutations, 2).
        window_size (int): The desired window size in number of SNPs.
        sparse (bool): If True, randomly samples SNPs to create a sparse representation. If False, crops from the center of region. Default is False.
        
    Returns:
        np.ndarray: A cropped numpy array with dimensions (n_samples, window_size, 2).
    """
    mut_array_window_size = mut_array.shape[1]
    if mut_array_window_size <  window_size:
        raise ValueError(f"Error: window size {window_size} is larger than number of SNPs in simulated window {mut_array_window_size}.")
    elif mut_array_window_size == window_size: #no cropping needed
        return mut_array
    elif mut_array_window_size >= window_size: #crop to window size
        if sparse: 
            random_sites = np.random.choice(mut_array_window_size, window_size, replace=False) # Sample random SNPs without replacement
            random_sites.sort()
            mut_array_sub = mut_array[:, random_sites, :]
            return mut_array_sub
        else:
            start_window_idx = mut_array_window_size//2 - window_size//2
            mut_array_sub = mut_array[:, start_window_idx:start_window_idx+window_size, :]
            return mut_array_sub


def major_minor(mut_array):
    """
    Converts the haplotype matrix to major minor allele representation with major allele =0 minor allele = 1.
    
    Args:
        mut_array (np.ndarray): The haplotype matrix with dimensions (n_samples, n_mutations, 2).
        
    Returns:
        np.ndarray: A numpy array where the major with major allele =0 minor allele = 1
    """
    # Identify positions where the allele frequency of mutation 1 is greater than 0.5
    idx= np.where(np.nanmean(mut_array[:,:,0], axis=0)>0.5) #find positions allele is > 0.5 frequency (make sure that is coded as 1, otherwise normalize)
    
    mut_array[:,idx,0]=1-mut_array[:,idx,0] #flip to minor allele

    # If you want to flit to major =1 minor =0
    #mut_array[:,:,0] =1-mut_array[:,:,0] #flip to major =1 minor =0
   
    return mut_array

def clusterHaps(numSamples,samples_dict):
    """
    Clusters haplotypes based on Hamming distance, considering missing data (Ns).

    Args:
        numSamples (int): Number of samples/haplotypes.
        samples_dict (dict): Dictionary where keys are sample indices and values are numpy arrays representing haplotypes.
    Returns:
        tuple: A tuple containing two dictionaries:
            - haps_clumped: Dictionary where keys are representative haplotypes and values are lists of sample indices that belong to that haplotype cluster.
            - haps_clump_adjusted: Dictionary where keys are representative haplotypes and values are counts of samples in that cluster.    
    """
 
    haps={}
    for j in range(numSamples):
        hap_array = ','.join(map(str,samples_dict[j]))#[:,0]))
        haps.setdefault(hap_array,[])
        haps[hap_array].append(j)
    
    #now clump haplotypes
    #If a haplotype matches another haplotype at all positions except for sites where there are Ns (missing data), then the haplotypes will be combined and the 'distance' between the two haplotypes will be considered 0.
    haps_clumped ={} #all clumped haplotypes
    haps_clumped_count={} # count number of haplotypes that are clumped

    compared ={} # keep track of the haps I have compared
    for key1 in haps.keys():
        if (key1 in compared) == False: # check if I've already compared this hap
            compared[key1]=1
            haps_clumped[key1] = haps[key1]
            haps_clumped_count[key1] = 1
            
            for key2 in haps.keys(): #iterate across other haplotype to compare to
                if ((haps[key2][0] in haps_clumped[key1])== False) and ((key2 in compared) == False): #check if sample is not already included in ha[s_clumped and that we have not iterated over key2
                    [distance,s1] = hamming_distance_clump(key1,key2,0.75)
                    if distance == 0 and key1 != s1: # if I replaced 'nan' in key1, I will replace the returned key1 in haps clumped
                        haps_clumped_count[s1] = haps_clumped_count[key1]
                        haps_clumped[s1] =haps_clumped[key1]
                        del haps_clumped_count[key1]
                        del haps_clumped[key1]
                        key1=s1
                    
                    if distance <= 0:#less  or equal to distance threshold. Distance is 0 and key1 could be == s1 (i.e missing data in s2 is merged with s1)
                        haps_clumped[key1] += haps[key2] # add the array for key2 to key1 array
                        haps_clumped_count[key1] += 1
                        compared[key2] = 1 # I won't check this distance again since it has been clumped
    
    # Create a new dictionary with lengths of values
    haps_clump_adjusted = {key: len(value) for key, value in haps_clumped.items()}

    return [haps_clumped,haps_clump_adjusted]

def hamming_distance_clump(s1,s2,missing_thresh):
    """
    Calculates the Hamming distance between two haplotypes, considering missing data (Ns).
    If a haplotype has more than a specified threshold of missing data, the distance is set to the length of the haplotype.
    Args:
        s1 (str): First haplotype as a comma-separated string.
        s2 (str): Second haplotype as a comma-separated string.
        missing_thresh (float): Threshold for the proportion of missing data (Ns) allowed before setting distance to length of haplotype.
    Returns:
        tuple: A tuple containing:
            - distance (int): The Hamming distance between the two haplotypes.
            - s1 (str): The potentially modified first haplotype after merging missing data.
    """
    list_s1 = s1.split(',')
    list_s2 = s2.split(',')
    #count nan's
    numNaN_s1 = list_s1.count('nan')
    numNaN_s2 = list_s2.count('nan')
    if numNaN_s1 >= int(len(list_s1)*missing_thresh) or numNaN_s2 >= int(len(list_s2)*missing_thresh): #if hap has more than missing_thresh missing dat
        distance =len(list_s1)
    else:
        distance = 0
        for i in range(len(list_s1)):
            if list_s1[i] != list_s2[i]:
                if list_s2[i] != 'nan':
                    if list_s1[i] != 'nan':
                        distance +=1
                        if distance > 0: #distance threshold
                            return [distance, s1]
                    else:
                        s1 = ','.join(list_s1[:i] + [list_s2[i]] + list_s1[i+1:])

    return [distance,s1]

def clusterHaps_byDistance(haps_clumped_count_sort,missing_thresh):
    """
    Clusters haplotypes by their Hamming distance to the most common haplotype, considering missing data (Ns).
    Args:
        haps_clumped_count_sort (dict): Dictionary of haplotypes sorted by their frequency.
        missing_thresh (float): Threshold for the proportion of missing data (Ns) allowed before counting as differences.
    Returns:
        dict: A dictionary where keys are haplotypes and values are their distances to the most common haplotype.
    """
    haps_clumped_distance= {}
    compared = {}
    hap1 = max(haps_clumped_count_sort, key=haps_clumped_count_sort.get)
    compared[hap1]=1 # most frequent haplotype
    haps_clumped_distance[hap1]=0 # cero distance to most common hap
    for key in haps_clumped_count_sort.keys():
        if (key in compared) == False:
            # If I havent compared calc distance
            list_s1 = hap1.split(',')
            list_s2 = key.split(',')
            #count nan's
            numNaN_s1 = list_s1.count('nan')
            numNaN_s2 = list_s2.count('nan')
            if numNaN_s1 >= int(len(list_s1)*missing_thresh) or numNaN_s2 >= int(len(list_s2)*missing_thresh): #if hap has more than missing_thresh missing dat
                distance=sum(x != y and (x != 'nan' or y != 'nan') for x, y in zip(list_s1, list_s2)) # if > missing thresh nan's count as differences 
                #distance =len(list_s1)
            else:
                distance = 0
                for i in range(len(list_s1)):
                    if list_s1[i] != list_s2[i]:
                        if list_s2[i] != 'nan':
                            if list_s1[i] != 'nan':
                                distance +=1
            haps_clumped_distance[key]= distance
    return haps_clumped_distance

def sort_haplotypes(mut_array,ordering=None):
    numSamples = mut_array.shape[0]
    samples_dict = {} 
    samples_dict_synnonsyn = {} 

    for j,row in enumerate(mut_array):
        samples_dict[j] = row[:,0] #only major minor coding to use as input for clustering
        samples_dict_synnonsyn[j] = row #  full data with syn nonsyn to use sample indexes after clustering
    
    #cluster haplotypes
    [haps_clumped, haps_clumped_count] =clusterHaps(numSamples,samples_dict)
  

    if ordering is None:
        print("ERROR: need to provide ordering")
        return 0
    else:   
        if ordering == 'rows_freq':
            #order by frequency of haplotype
            haps_clumped_count_sort =dict(sorted(haps_clumped_count.items(), key=lambda item: item[1], reverse=True))
            gene_data_sorted_lst=[]
            for hap in haps_clumped_count_sort.keys():
                    for id in haps_clumped[hap]:
                        gene_data_sorted_lst.append(samples_dict_synnonsyn[id])
                        #gene_data_sorted_lst.append(samples_dict[id])
            gene_data_sorted = np.array(gene_data_sorted_lst)

            mut_array[:,:,:] = gene_data_sorted

        #sort by distance to major haplotype
        if ordering == 'rows_dist':
            haps_clumped_count_sort =dict(sorted(haps_clumped_count.items(), key=lambda item: item[1], reverse=True))
            #now order by distance to hap of highest freq
            haps_clumped_distance = clusterHaps_byDistance(haps_clumped_count_sort,0.9)
            haps_clumped_distance_sort = dict(sorted(haps_clumped_distance.items(), key=lambda item: item[1], reverse=False))
            gene_data_sorted_lst=[]
            for hap in haps_clumped_distance_sort.keys():
                # next I want to get all the samples in the full window with subhaplotype hap and order acording to frequency
                ids_hap = haps_clumped[hap] # list of sample ids with that haplotype
                data_hap = mut_array[ids_hap,:,0] # get the data for those samples
                [haps_clumped_window, haps_clumped_count_window] =clusterHaps(data_hap.shape[0],data_hap) # this is giving me line numbers based on subhap
                haps_clumped_count_window_sort =dict(sorted(haps_clumped_count_window.items(), key=lambda item: item[1], reverse=True))    
                #now sort data
                for hap_window in haps_clumped_count_window_sort.keys():
                    hap_sub_sorted_ids=[ids_hap[i] for i in haps_clumped_window[hap_window]] 
                    for id in hap_sub_sorted_ids:
                        gene_data_sorted_lst.append(samples_dict_synnonsyn[id])
                    #for id in haps_clumped[hap]:
                    #    gene_data_sorted_lst.append(samples_dict_synnonsyn[id])
            gene_data_sorted = np.array(gene_data_sorted_lst)

            mut_array[:,:,:] = gene_data_sorted

    return mut_array


# ---------------------------------------------------------------------------------------------------------------------  

# Main script execution
if __name__ == "__main__":

    # Check for correct number of command-line arguments
    if len(sys.argv) != 6:
        print("Usage: python processSLiMsims.py <input_slim_file> <output_npy_file> <n_samples> <window_size><index>")
        sys.exit(1)

    # Read command-line arguments
    PATH_SIMS = sys.argv[1] #input SLiM simulation file ex: "slim_hitchhiking_0.2.txt"
    OUTPUT_PATH = sys.argv[2] #output numpy file name ex: "mut_array_0.2_ancestralCoding.npy"
    NUM_SAMPS = int(sys.argv[3]) #number of haplotypes to sample ex: 100
    WINDOW_SIZE = int(sys.argv[4]) #window size in number of SNPs ex: 201
    INDEX= int(sys.argv[5]) #index in big array to store results


    # Parse the SLiM simulation file and generate dictionaries with mutation and genome data
    mut_type_dict, mut_pos_dict, pos_to_idx_dict, genomes_dict = parse_slim_sims(PATH_SIMS)
    
    # Create the haplotype matrix as a numpy array
    mut_array = create_haplotype_matrix(mut_type_dict, mut_pos_dict, pos_to_idx_dict, genomes_dict)


    # sample n individuals/haplotypes
    mut_array = sample_haplotypes(mut_array, n_samples=NUM_SAMPS)

    #crop to window size in number of SNPs
    mut_array = crop(mut_array, window_size=WINDOW_SIZE, sparse=False)

    #major minor 
    #mut_array=major_minor(mut_array) #use this is you want to sort by major minor otherwise 0=ancestral, 1=derived
    
    #sort haplotypes    
    mut_array_sorted= sort_haplotypes(mut_array,'rows_freq')
    #print(mut_array_sorted.shape)

    #save as npy
    #np.save(output_path, mut_array_sorted)
    # Open big file in write mode and store directly
    big_array = np.lib.format.open_memmap(
        OUTPUT_PATH, dtype=np.float32, mode="r+", shape=(2, NUM_SAMPS, WINDOW_SIZE, 2)
    )
    #print(big_array.shape)
    big_array[INDEX] = mut_array_sorted

    del big_array  # flush changes