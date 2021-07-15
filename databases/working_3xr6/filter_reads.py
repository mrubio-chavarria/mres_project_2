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
def filter_reads(read_folders, reference_file,  q_score_threshold):
    """
    DESCRIPTION:
    A function to filter the reads based on their q score. 
    :param read_folders: [str] the route to folder with th reads.
    :param reference_file: [str] route to the reference file to align.
    :param q_score_threshold: [float] the value to filter.
    """
    label = f'Q{int(q_score_threshold)}_'  # Label to distinguish the good reads
    for reads_folder in read_folders:
        read_files = os.listdir(reads_folder)
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


def annotate_basecalls(*args):
    """
    """
    for pair in args:
        command = f'tombo preprocess annotate_raw_with_fastqs --fast5-basedir {pair[0]} --fastq-filenames {pair[1]} --overwrite'
        os.system(command)
        #subprocess.check_call(command, shell=True)
    

def read_code(filename):
    code1, code2 = filename.split('_')[2:4]
    return code1[0:8] + '_' + code2

if __name__ == "__main__":
    
    workdir = sys.argv[1]
    n_processes = int(sys.argv[2])
    flowcell = sys.argv[3]
    
    # workdir = f'/home/mario/Projects/project_2/databases/working_3xr6'
    # n_processes = 2
    # flowcell = 'flowcell3'

    # Filter files below the q score threshold
    print('***************************************************************************************')
    print('Filter the reads')
    print('***************************************************************************************')
    reads_folder = workdir + '/' + 'reads' + '/' + flowcell
    single_reads_folder = reads_folder + '/' + 'single'
    single_folders = [single_reads_folder + '/' + folder 
        for folder in os.listdir(single_reads_folder) 
        if not (folder.endswith('txt') or folder.endswith('index'))]
    q_score_threshold = 20.0
    filtered_reads = []
    n_folders_per_process = len(single_folders) // n_processes
    # reads_folders_lists = [single_folders[n_folders_per_read*i:n_folders_per_read*(i+1)] 
    #     if i != (n_processes - 1) else single_folders[n_folders_per_read*i:] 
    #     for i in range(n_processes)]
    reads_folders_lists = [single_folders[n_folders_per_process*i:n_folders_per_process*(i+1)] 
        for i in range(n_processes)]
    if len(single_folders) % n_processes != 0:
        extra_pairs = single_folders[n_folders_per_process * n_processes::]
        [reads_folders_lists[i].append(extra_pairs[i]) for i in range(len(extra_pairs))]
    processes = []
    # manager = Manager()
    # filtered_reads = manager.list()
    reference_file = workdir + '/' + 'reference.fasta'
    for i in range(n_processes):
        print(f'Process {i} launched')
        process = mp.Process(target=filter_reads,
                            args=(reads_folders_lists[i], reference_file, q_score_threshold))
        processes.append(process)
        process.start()
    for process in processes:
        process.join()
    # filtered_reads = list(filtered_reads)
    # filtered_reads = '\n'.join(filtered_reads)
    # file = open(reads_folder + '/' + 'filtered_reads.txt', 'w')
    # file.write(filtered_reads)
    # file.close()

    print('High-quality reads marked')
            
    

    
    


