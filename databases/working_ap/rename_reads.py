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
    for read_file in read_files:
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
            filename = read_file.split('/')[-1]
            reads_folder = '/'.join(read_file.split('/')[:-1])
            if label[:-1] in filename.split('_'):
                new_filename = reads_folder + '/' + label + filename.split('_')[-1]
            else:
                new_filename = reads_folder + '/' + label + filename
            os.rename(read_file, new_filename)


if __name__ == "__main__":
    
    workdir = sys.argv[1]
    flowcell = sys.argv[2]
    
    # workdir = f'/home/mario/Projects/project_2/databases/working_3xr6'
    # n_processes = 2
    # flowcell = 'flowcell3'

    # Filter files below the q score threshold
    print('***************************************************************************************')
    print('Rename read files')
    print('Flowcell:', flowcell)
    print('***************************************************************************************')
    single_reads_folder = workdir + '/' + 'reads' + '/' + flowcell + '/' + 'single'
    # Rename files
    for idx, file in enumerate(os.listdir(single_reads_folder)):
        if not file.endswith('fast5'):
            continue
        old_filename = single_reads_folder + '/' + file
        new_filename = single_reads_folder + '/' + f'read{str(idx)}.fast5'
        os.rename(old_filename, new_filename)
    print('Renaming changed')
        

            
    

    
    


