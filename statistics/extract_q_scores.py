#!/home/mario/anaconda3/envs/project2_venv/bin python

"""
DESCRIPTION:
This script tries to gather all the steps needed to 
perform once the basecalls have been obtained.
"""

# Libraries
import os
import sys
from tombo import tombo_helper, tombo_stats, resquiggle
import h5py
import mappy
import pandas as pd
import multiprocessing as mp


# Functions
def extract_info(pair):
    read_file, reference_file = pair
    try:
        fast5_data = h5py.File(read_file, 'r')
        read_id = fast5_data['Raw']['Reads'][list(fast5_data['Raw']['Reads'].keys())[0]].attrs['read_id'].decode('UTF-8')
        # Set parameters for resquiggling
        aligner = mappy.Aligner(reference_file, preset=str('map-ont'), best_n=1)
        seq_samp_type = tombo_helper.get_seq_sample_type(fast5_data)
        std_ref = tombo_stats.TomboModel(seq_samp_type=seq_samp_type)
        # Extract data from FAST5
        mean_q_score = resquiggle.map_read(fast5_data, aligner, std_ref).mean_q_score
        failed_parsing = False
        failed_alignment = False
    except tombo_helper.TomboError:
        mean_q_score = 0
        failed_parsing = False
        failed_alignment = True
    except OSError:
        read_id = "Failed parsing"
        mean_q_score = 0
        failed_parsing = True
        failed_alignment = True
    # Return paremeters
    return read_id, mean_q_score, failed_parsing, failed_alignment

def extract_q_score(read_files, reference_file, workdir, n_processes=2):
    """
    DESCRIPTION:
    
    :param reference_file: [str] route to the reference file to align.
    :param q_score_threshold: [float] the value to filter.
    """
    data = []
    # Label to distinguish the good reads
    pairs = map(lambda x: (x, reference_file), read_files)
    with mp.Pool(n_processes) as pool:
        data = pool.map(extract_info, pairs)
    dataset_name = reference_file.split('/')[-2].split('_')[-1]
    storage_file = workdir + '/' + f'q_score_record_{dataset_name}.tsv'
    pd.DataFrame(data=data, columns=('read_id', 'mean_q_score', 'failed_parsing', 'failed_alignment')).to_csv(storage_file, sep='\t')


if __name__ == "__main__":
    
    # workdir = sys.argv[1]
    # n_processes = int(sys.argv[2])

    workdir = "/home/mario/Projects/project_2/databases/working_3xr6"
    n_processes = 4

    if workdir.endswith('ap'):
        flowcells = ['flowcell1', 'flowcell2', 'flowcell3', 'flowcell4']
        # Filter files below the q score threshold
        print('***************************************************************************************')
        print('Extract Q score')
        print('***************************************************************************************')
        single_read_files = []
        for flowcell in flowcells:
            reads_folder = workdir + '/' + 'reads' + '/' + flowcell
            single_reads_folder = reads_folder + '/' + 'single'
            q_score_threshold = 20.0
            filtered_reads = []
            single_read_files.extend([single_reads_folder + '/' + file for file in os.listdir(single_reads_folder)])
        
        reference_file = workdir + '/' + 'reference.fasta'
        
    elif workdir.endswith('3xr6'):
        flowcells = ['flowcell1', 'flowcell2', 'flowcell3']
        # Filter files below the q score threshold
        print('***************************************************************************************')
        print('Extract Q score')
        print('***************************************************************************************')
        single_read_files = []
        for flowcell in flowcells:
            reads_folder = workdir + '/' + 'reads' + '/' + flowcell
            single_reads_folder = reads_folder + '/' + 'single'
            q_score_threshold = 20.0
            filtered_reads = []
            subfolders = [single_reads_folder + '/' + subfolder for subfolder in os.listdir(single_reads_folder) if not subfolder.endswith('index') and not subfolder.endswith('txt')]
            single_read_files.extend([it for sl in [[folder + '/' + file for file in os.listdir(folder)] for folder in subfolders] for it in sl])
        
        reference_file = workdir + '/' + 'reference.fasta'

    print('Filter the selected reads')
    print('Number of reads:', len(single_read_files))
    extract_q_score(single_read_files, reference_file, workdir, n_processes)
    print('Q scores stored')            


    
    


