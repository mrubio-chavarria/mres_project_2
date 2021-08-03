#!/home/mario/anaconda3/envs/project2_venv/bin python

"""
DESCRIPTION:
This script tries to gather all the steps needed to 
perform once the basecalls have been obtained.
"""

# Libraries
import os
import sys
from numpy.core.fromnumeric import mean
from tombo import tombo_helper, tombo_stats, resquiggle
import h5py
import mappy
from torch import threshold
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Manager
from shutil import move, rmtree
from multiprocessing import Pool
import csv


# Functions
def filter_reads(pair):
    read_file, reference_file, threshold = pair
    try:
        fast5_data = h5py.File(read_file, 'r')
        read_id = fast5_data['Raw']['Reads'][list(fast5_data['Raw']['Reads'].keys())[0]].attrs['read_id'].decode('UTF-8')
        # Set parameters for resquiggling
        aligner = mappy.Aligner(reference_file, preset=str('map-ont'), best_n=1)
        seq_samp_type = tombo_helper.get_seq_sample_type(fast5_data)
        std_ref = tombo_stats.TomboModel(seq_samp_type=seq_samp_type)
        # Extract data from FAST5
        mean_q_score = resquiggle.map_read(fast5_data, aligner, std_ref).mean_q_score
        if mean_q_score >= threshold:
            filename = read_file.split('/')[-1]
            subfolder = '/'.join(read_file.split('/')[:-1])
            new_filename = subfolder + '/' + label + filename
            print('Read renamed:', new_filename)
            os.rename(read_file, new_filename)
    except tombo_helper.TomboError:
        pass
    except OSError:
        pass

    

if __name__ == "__main__":
    
    workdir = sys.argv[1]
    flowcell = sys.argv[2]
    
    # workdir = f'/home/mario/Projects/project_2/databases/working_3xr6'
    # flowcell = 'flowcell3'

    # Filter files below the q score threshold
    print('***************************************************************************************')
    print('Filter the reads')
    print('Flowcell:', flowcell)
    print('***************************************************************************************')
    label = 'Q7_'
    # # Read the IDs list
    # filtered_reads_ids_file = open(workdir + '/' + 'filtered_reads.tsv')
    # filtered_reads_ids = [file_id[0] for file_id in csv.reader(filtered_reads_ids_file, delimiter='\t') if file_id[0] != 'read_id']
    # filtered_reads_ids_file.close()
    # Read files with the signals
    reads_folder = workdir + '/' + 'reads' + '/' + flowcell
    single_reads_folder = reads_folder + '/' + 'single'
    q_score_threshold = 7.0
    filtered_reads = []
    subfolders = [single_reads_folder + '/' + subfolder for subfolder in os.listdir(single_reads_folder) if not subfolder.endswith('index') and not subfolder.endswith('txt')]
    reference_file = workdir + '/' + 'reference.fasta'
    single_read_files = tuple((it, reference_file, q_score_threshold) for sl in [[folder + '/' + file for file in os.listdir(folder)] for folder in subfolders] for it in sl)
    reference_file = workdir + '/' + 'reference.fasta'
    print('Filter the selected reads')
    print('Number of reads:', len(single_read_files))
    # Filter the files
    with Pool(4) as p:
        p.map(filter_reads, single_read_files)
    print('High-quality reads marked')            


    
    


