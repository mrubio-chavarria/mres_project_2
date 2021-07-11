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
    selected_folder=sys.argv[4]
    
    # workdir = f'/home/mario/Projects/project_2/databases/working_3xr6'
    # selected_folder = '0'
    # n_processes = 4
    # flowcell = 'flowcell3'

    # Format to multiple to single read files
    # print('***************************************************************************************')
    # print('Format reads from multi to single files')
    # print('***************************************************************************************')
    reads_folder = workdir + '/' + 'reads' + '/' + flowcell
    basecalls_folder = workdir + '/' + 'basecalls' + '/' + flowcell
    fastq_file = basecalls_folder + '/' + 'multi.fastq'
    #command = f'multi_to_single_fast5 --input_path {reads_folder}/multi --save_path {reads_folder}/single'
    #os.system(command)
    # # Flatten directory structure
    # single_reads_folders = [reads_folder + '/' + 'single' +  '/' + folder 
    #     for folder in os.listdir(reads_folder + '/' + 'single') if not folder.endswith('txt')]
    # single_reads_folder = reads_folder + '/' + 'single'
    single_reads_folder = reads_folder + '/' + 'single' + '/' + selected_folder
    # Annotate the reads with the basecalls
    print('***************************************************************************************')
    print('Annotate the reads')
    print('***************************************************************************************')
    # Read all the possible fastqs
    command = f'tombo preprocess annotate_raw_with_fastqs --fast5-basedir {single_reads_folder} --fastq-filenames {fastq_file} --overwrite'
    code = os.system(command)
    print('Annotation completed')

    # single_folders = [folder for folder in os.listdir(single_reads_folder) if not folder.endswith('txt')]
    # single_folders = [single_reads_folder + '/' + file for file in sorted(single_folders, key=lambda x: int(x.split('/')[-1]))]
    # fastq_files = [basecalls_folder + '/' + file for file in sorted(os.listdir(basecalls_folder), key=lambda x: int(x[:-6].split('_')[-2]))]
    # file_pairs = list(zip(single_folders, fastq_files))
    # group_size = len(file_pairs) // n_processes
    # group_indeces = list(range(0, len(file_pairs), group_size))
    # file_groups = [file_pairs[group_size * i:group_size * (i+1)] if i != n_processes - 1 else file_pairs[group_size * i::] 
    #     for i in range(n_processes)]
    # if len(file_pairs) % n_processes != 0:
    #    extra_pairs = file_pairs[group_size * n_processes::]
    #    [file_groups[i].append(extra_pairs[i]) for i in range(len(extra_pairs))]
    # processes = []
    # for rank in range(n_processes):
    #     print(f'Process {rank} launched')
    #     process = mp.Process(target=annotate_basecalls, args=(file_groups[rank]))
    #     process.start()
    # for process in processes:
    #     process.join()
    # print('Annotation completed')

    # Resquiggle
    print('***************************************************************************************')
    print('Resquiggle the reads...')
    print('***************************************************************************************')    
    reference_file = workdir + '/' + 'reference.fasta'
    command = f'tombo resquiggle {single_reads_folder} {reference_file} --processes {n_processes} --num-most-common-errors 5 --overwrite'
    os.system(command)
    print('Resquiggling completed')
    
            
    

    
    


