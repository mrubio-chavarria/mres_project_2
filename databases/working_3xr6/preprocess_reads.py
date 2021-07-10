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


def annotate_basecalls(pairs, workdir):
    """
    """
    #for i in range(len(basecalls_files)):
    for pair in pairs:
        command = f'tombo preprocess annotate_raw_with_fastqs --fast5-basedir {pair[1]} --fastq-filenames {pair[0]} --overwrite'
        os.system(command)
    

def read_code(filename):
    code1, code2 = filename.split('_')[2:4]
    return code1[0:8] + '_' + code2

if __name__ == "__main__":
    
    workdir = f'{sys.argv[1]}/databases/working_3xr6'
    n_processes = int(sys.argv[2])
    
    # workdir = f'/home/mario/Projects/project_2/databases/working_3xr6'
    # n_processes = 4

    # Format to multiple to single read files
    print('***************************************************************************************')
    print('Format reads from multi to single files')
    print('***************************************************************************************')
    reads_folder = workdir + '/' + 'reads'
    basecalls_folder = workdir + '/' + 'basecalls'
    command = f'multi_to_single_fast5 --input_path {reads_folder}/multi --save_path {reads_folder}/single'
    # os.system(command)
    # Format folder name base on file codes
    single_reads_folder = workdir + '/reads/single'
    file = open(single_reads_folder + '/filename_mapping.txt')
    mapping = file.readlines()
    file.close()
    mapping = [line.split('\t') for line in mapping]
    mapping = [('_'.join(line[0].split('/')[9].split('_')[1::]).replace('.fast5', ''), line[1].replace('\n', '')) 
        for line in mapping]
    reads_mapping = {key: [] for key in set([line[0] for line in mapping])}
    [reads_mapping[line[0]].append(line[1]) for line in mapping]
    folders = os.listdir(single_reads_folder)
    for folder in folders:
        if folder.endswith('txt'): continue
        folder = single_reads_folder + '/' + folder
        reads_in_folder = os.listdir(folder)
        for key in reads_mapping.keys():
            if reads_mapping[key][0] in reads_in_folder:
                os.rename(folder, single_reads_folder + '/' + key)
                break
    file_pairs = []
    single_reads_folders = os.listdir(single_reads_folder)
    for file in os.listdir(basecalls_folder):
        [file_pairs.append((basecalls_folder + '/' + file,
                            single_reads_folder + '/' + folder))
            for folder in single_reads_folders if folder == read_code(file)]

    # Annotate the reads with the basecalls
    print('***************************************************************************************')
    print('Annotate the reads')
    print('***************************************************************************************')
    # single_read_folders = [reads_folder + '/' + 'single' + '/' + folder 
    #     for folder in os.listdir(reads_folder + '/' + 'single') if not folder.endswith('txt') and not folder.endswith('index')]
    # single_read_folders = sorted(single_read_folders, key=lambda x: int(x.split('/')[-1]))
    # basecalls_files = sorted(os.listdir(basecalls_folder), key=lambda x: int(x.split('_')[3]))
    # file_pairs = list(zip(single_read_folders, basecalls_files))
    group_size = len(file_pairs) // n_processes
    group_indeces = list(range(0, len(file_pairs), group_size))
    file_groups = [file_pairs[group_size * index:group_size * (index+1)] if index != file_pairs[group_size * index::] else index 
        for index in group_indeces]
    processes = []
    for rank in range(n_processes):
        process = mp.Process(target=annotate_basecalls, args=(file_groups[rank][0], file_groups[rank][1], workdir))
    for process in processes:
        process.join()

    # Resquiggle
    print('***************************************************************************************')
    print('Resquiggle the reads...')
    print('***************************************************************************************')
    reference_file = workdir + '/' + 'reference.fasta'
    for folder in single_reads_folders:
        folder = single_reads_folder + '/' + folder
        command = f'tombo resquiggle {folder} {reference_file} --processes {n_processes} --num-most-common-errors 5 --overwrite'
        os.system(command)

    # Filter files below the q score threshold
    print('***************************************************************************************')
    print('Filter the reads')
    print('***************************************************************************************')
    q_score_threshold = 20.0
    filtered_reads = []
    n_folders_per_read = len(single_reads_folders) // n_processes
    reads_folders_lists = [single_reads_folders[n_folders_per_read*i:n_folders_per_read*(i+1)] 
        if i != (n_processes - 1) else single_reads_folders[n_folders_per_read*i:] 
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
            
    

    
    


