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



if __name__ == "__main__":
    
    workdir = sys.argv[1]
    n_processes = int(sys.argv[2])
    flowcell = sys.argv[3]
    selected_folder=sys.argv[4]
    
    # Read files
    reads_folder = workdir + '/' + 'reads' + '/' + flowcell
    basecalls_folder = workdir + '/' + 'basecalls' + '/' + flowcell
    fastq_file = basecalls_folder + '/' + 'multi.fastq'
    single_reads_folder = reads_folder + '/' + 'single' + '/' + selected_folder
    
    # Annotate the reads with the basecalls
    print('***************************************************************************************')
    print('Annotate the reads')
    print('***************************************************************************************')
    # Read all the possible fastqs
    command = f'tombo preprocess annotate_raw_with_fastqs --fast5-basedir {single_reads_folder} --fastq-filenames {fastq_file} --overwrite'
    code = os.system(command)
    print('Annotation completed')

    # Resquiggle
    print('***************************************************************************************')
    print('Resquiggle the reads...')
    print('***************************************************************************************')    
    reference_file = workdir + '/' + 'reference.fasta'
    command = f'tombo resquiggle {single_reads_folder} {reference_file} --processes {n_processes} --num-most-common-errors 5 --overwrite'
    os.system(command)
    print('Resquiggling completed')
    
            
    

    
    


