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
from multiprocessing import Pool
import csv


# Functions
def filter_reads(pair):
    read_file, filtered_reads_ids = pair
    try:
        fast5_data = h5py.File(read_file, 'r')
        read_id = fast5_data['Raw']['Reads'][list(fast5_data['Raw']['Reads'].keys())[0]].attrs['read_id'].decode('UTF-8')
        if read_id in filtered_reads_ids:
            filename = read_file.split('/')[-1]
            new_filename = single_reads_folder + '/' + label + filename
            os.rename(read_file, new_filename)
    except OSError:
        # Corrupted file, do not modify
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
    label = 'Q7'
    # Read the IDs list
    filtered_reads_ids_file = open(workdir + '/' + 'filtered_reads.tsv')
    filtered_reads_ids = [file_id[0] for file_id in csv.reader(filtered_reads_ids_file, delimiter='\t') if file_id[0] != 'read_id']
    filtered_reads_ids_file.close()
    # Read files with the signals
    reads_folder = workdir + '/' + 'train_reads' + '/' + flowcell
    single_reads_folder = reads_folder + '/' + 'single'
    q_score_threshold = 7.0
    filtered_reads = []
    subfolders = [single_reads_folder + '/' + subfolder for subfolder in os.listdir(single_reads_folder) if not subfolder.endswith('index') and not subfolder.endswith('txt')]
    single_read_files = [(it, filtered_reads_ids[:]) for sl in [[folder + '/' + file for file in os.listdir(folder)] for folder in subfolders] for it in sl]
    reference_file = workdir + '/' + 'reference.fasta'
    print('Filter the selected reads')
    print('Number of reads:', len(single_read_files))
    # Filter the files
    with Pool(8) as p:
        p.map(filter_reads, single_read_files)
    print('High-quality reads marked')            


    
    


