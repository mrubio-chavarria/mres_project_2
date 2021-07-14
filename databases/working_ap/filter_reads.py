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
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Manager
from shutil import move, rmtree


# Functions
def filter_reads(read_files, reference_file,  q_score_threshold):
    """
    DESCRIPTION:
    A function to filter the reads based on their q score. 
    :param reference_file: [str] route to the reference file to align.
    :param q_score_threshold: [float] the value to filter.
    """
    label = f'Q{int(q_score_threshold)}_'  # Label to distinguish the good reads
    for file in read_files:
        read_file = reads_folder + '/' + file
        fast5_data = h5py.File(read_file, 'r')
        # Set parameters for resquiggling
        aligner = mappy.Aligner(reference_file, preset=str('map-ont'), best_n=1)
        seq_samp_type = tombo_helper.get_seq_sample_type(fast5_data)
        std_ref = tombo_stats.TomboModel(seq_samp_type=seq_samp_type)
        # Extract data from FAST5
        try:
            map_results = resquiggle.map_read(fast5_data, aligner, std_ref)
        except tombo_helper.TomboError:
            # Avoid reads lacking alignment (very poor quality)
            continue
        # Filter reads based on q score for quality
        if map_results.mean_q_score >= q_score_threshold:
            # If it is of good quality, rename the file
            os.rename(read_file, reads_folder + '/' + label + file)


if __name__ == "__main__":
    
    workdir = sys.argv[1]
    n_jobs = int(sys.argv[2])
    flowcell = sys.argv[3]
    job_index = sys.argv[4]
    
    # workdir = f'/home/mario/Projects/project_2/databases/working_3xr6'
    # n_processes = 2
    # flowcell = 'flowcell3'

    # Filter files below the q score threshold
    print('***************************************************************************************')
    print('Filter the reads')
    print('***************************************************************************************')
    reads_folder = workdir + '/' + 'reads' + '/' + flowcell
    single_reads_folder = reads_folder + '/' + 'single'
    q_score_threshold = 20.0
    filtered_reads = []
    single_read_files = [single_reads_folder + '/' + file 
        for file in os.listdir(single_reads_folder) if file.endswith('fast5')]
    n_files_per_job = len(single_read_files) // n_jobs
    
    reads_to_filter = single_read_files[n_files_per_job*job_index:n_files_per_job*(job_index+1)]

    if len(single_read_files) % n_jobs != 0:
        extra_reads = single_read_files[n_files_per_job * n_jobs::]
        if len(extra_reads) > job_index:
            reads_to_filter.append(extra_reads[job_index])

    reference_file = workdir + '/' + 'reference.fasta'

    print('Filter the selected reads')
    filter_reads(reads_to_filter, reference_file, q_score_threshold)

    print('High-quality reads marked')
            
    

    
    


