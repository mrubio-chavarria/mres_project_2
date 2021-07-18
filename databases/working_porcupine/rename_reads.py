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


if __name__ == "__main__":
    
    workdir = sys.argv[1]
    flowcell = sys.argv[2]

    # Filter files below the q score threshold
    print('***************************************************************************************')
    print('Rename read files')
    print('Flowcell:', flowcell)
    print('***************************************************************************************')
    single_reads_folder = workdir + '/' + 'reads' + '/' + flowcell + '/' + 'single'
    # Rename files
    for folder in os.listdir(single_reads_folder):
        for idx, file in enumerate(os.listdir(single_reads_folder + '/' + folder)):
            if not file.endswith('fast5'):
                continue
            old_filename = single_reads_folder + '/' + folder + '/' + file
            new_filename = single_reads_folder + '/' + folder + '/' + f'read{str(idx)}.fast5'
            os.rename(old_filename, new_filename)
    print('Renaming completed')
        

            
    

    
    


