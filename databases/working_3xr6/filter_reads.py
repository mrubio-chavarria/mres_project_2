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
def filter_reads(read_files, reference_file,  q_score_threshold, workdir, new_way=False):
    """
    DESCRIPTION:
    A function to filter the reads based on their q score. 
    :param reference_file: [str] route to the reference file to align.
    :param q_score_threshold: [float] the value to filter.
    """
    label = f'Q{int(q_score_threshold)}_' 
    # Label to distinguish the good reads
    if new_way:
        # Load list
        read_ids_file = open(workdir + '/' + 'filtered_reads.tsv')
        read_ids = [read[0] for read in csv.reader(read_ids_file, delimiter='\t')][1::]
        read_ids_file.close()
        for read_file in read_files:
            try:
                fast5_data = h5py.File(read_file, 'r')
            except OSError:
                # File badly parsed or corrupted
                continue
            read_id = fast5_data['Raw']['Reads'][list(fast5_data['Raw']['Reads'].keys())[0]].attrs['read_id'].decode('UTF-8')
            if read_id in read_ids:
                reads_folder = '/'.join(read_file.split('/')[:-1])
                filename = read_file.split('/')[-1]
                new_filename = reads_folder + '/' + label + filename
                os.rename(read_file, new_filename)
    else:
        for read_file in read_files:
            try:
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
            except:
                # Corrupted file
                pass
            


if __name__ == "__main__":
    
    # workdir = sys.argv[1]
    # flowcell = sys.argv[2]
    
    workdir = f'/home/mario/Projects/project_2/databases/working_3xr6'
    flowcell = 'flowcell3'
    
    for flowcell in ['flowcell2']:
        if workdir.endswith('ap'):
            # Filter files below the q score threshold
            print('***************************************************************************************')
            print('Filter the reads')
            print('Flowcell:', flowcell)
            print('***************************************************************************************')
            reads_folder = workdir + '/' + 'reads' + '/' + flowcell
            single_reads_folder = reads_folder + '/' + 'single'
            q_score_threshold = 7.0
            filtered_reads = []
            single_read_files = [single_reads_folder + '/' + file for file in os.listdir(single_reads_folder)]
            
            reference_file = workdir + '/' + 'reference.fasta'

            print('Filter the selected reads')
            print('Number of reads:', len(single_read_files))
            print('Reads filenames:')
            [print(read.split('/')[-1]+'\n') for read in single_read_files]
            filter_reads(single_read_files, reference_file, q_score_threshold, workdir)
            print('High-quality reads marked')
        elif workdir.endswith('3xr6'):
            # Filter files below the q score threshold
            print('***************************************************************************************')
            print('Filter the reads')
            print('Flowcell:', flowcell)
            print('***************************************************************************************')
            reads_folder = workdir + '/' + 'reads' + '/' + flowcell
            single_reads_folder = reads_folder + '/' + 'single'
            q_score_threshold = 7.0
            filtered_reads = []
            subfolders = [single_reads_folder + '/' + subfolder for subfolder in os.listdir(single_reads_folder) 
                if not subfolder.endswith('index') and not subfolder.endswith('txt') and not subfolder.endswith('gz')]
            single_read_files = [it for sl in [[folder + '/' + file for file in os.listdir(folder)] for folder in subfolders] for it in sl]
            reference_file = workdir + '/' + 'reference.fasta'
            print('Filter the selected reads')
            print('Number of reads:', len(single_read_files))
            # print('Reads filenames:')
            # [print(read.split('/')[-1]+'\n') for read in single_read_files]
            filter_reads(single_read_files, reference_file, q_score_threshold, workdir)
            print('High-quality reads marked') 
    
    


