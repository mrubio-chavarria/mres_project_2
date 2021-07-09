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


# Functions
def filter_reads(read_folders, filtered_reads, q_score_threshold):
    """
    DESCRIPTION:
    A function to filter the reads based on their q score. 
    :param read_folders: [str] the route to folder with th reads.
    :param filtered_reads: [manager.list()] list in which the names
    of the filtered reads should be appended.
    :param q_score_threshold: [float] the value to filter.
    """
    count = 0
    for reads_folder in read_folders:
        read_files = os.listdir(reads_folder)
        for file in tqdm(read_files):
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
                count += 1
                filtered_reads.append(read_file)


def annotate_basecalls(single_read_folders, basecalls_files, workdir):
    """
    """
    #for i in range(len(basecalls_files)):
    for i in range(len(basecalls_files)):
        basecall_file = workdir + '/' + 'basecalls' + '/' + basecalls_files[i]
        command = f'tombo preprocess annotate_raw_with_fastqs --fast5-basedir {single_read_folders[i]} --fastq-filenames {basecall_file} --overwrite'
        os.system(command)
    

if __name__ == "__main__":
    
    workdir = f'{sys.argv[1]}/databases/working_3xr6'
    n_processes = int(sys.argv[2])
    
    # workdir = f'/home/mario/Projects/project_2/databases/working_3xr6'
    # n_process = 4

    # Format to multiple to single read files
    print('***************************************************************************************')
    print('Format reads from multi to single files')
    print('***************************************************************************************')
    reads_folder = workdir + '/' + 'reads'
    command = f'multi_to_single_fast5 --input_path {reads_folder}/multi --save_path {reads_folder}/single'
    os.system(command)

    # Annotate the reads with the basecalls
    print('***************************************************************************************')
    print('Annotate the reads')
    print('***************************************************************************************')
    single_read_folders = [reads_folder + '/' + 'single' + '/' + folder 
        for folder in os.listdir(reads_folder + '/' + 'single') if not folder.endswith('txt') and not folder.endswith('index')]
    single_read_folders = sorted(single_read_folders, key=lambda x: int(x.split('/')[-1]))

    basecalls_folder = workdir + '/' + 'basecalls'
    basecalls_files = sorted(os.listdir(basecalls_folder), key=lambda x: int(x.split('_')[3]))
    file_pairs = list(zip(single_read_folders, basecalls_files))
    group_size = len(file_pairs) // n_processes
    group_indeces = list(range(0, len(file_pairs), group_size))
    file_groups = [file_pairs[group_size * index:group_size * (index+1)] if index != file_pairs[group_size * index::] else index 
        for index in group_indeces]
    prcesses = []
    for rank in range(n_processes):
        process = mp.Process(target=annotate_basecalls, args=(file_groups[rank][0], file_groups[rank][1], workdir))

    # Resquiggle
    print('***************************************************************************************')
    print('Resquiggle the reads...')
    print('***************************************************************************************')
    reference_file = workdir + '/' + 'reference.fasta'
    for folder in single_read_folders:
        command = f'tombo resquiggle {folder} {reference_file} --processes {n_processes} --num-most-common-errors 5 --overwrite'
        os.system(command)

    # Filter files below the q score threshold
    print('***************************************************************************************')
    print('Filter the reads')
    print('***************************************************************************************')
    q_score_threshold = 20.0
    filtered_reads = []
    n_folders_per_read = len(single_read_folders) // n_processes
    reads_folders_lists = [single_read_folders[n_folders_per_read*i:n_folders_per_read*(i+1)] 
        if i != (n_processes - 1) else single_read_folders[n_folders_per_read*i:] 
        for i in range(n_processes)]
    processes = []
    manager = Manager()
    filtered_reads = manager.list()
    for i in range(n_processes):
        process = mp.Process(target=filter_reads, args=(reads_folders_lists[i], filtered_reads, q_score_threshold))
        processes.append(process)
        process.start()
    for process in processes:
        process.join()
    filtered_reads = list(filtered_reads)
    filtered_reads = '\n'.join(filtered_reads)
    file = open(workdir + '/' + 'filtered_reads.txt', 'w')
    file.write(filtered_reads)
    file.close()
            
    

    
    


