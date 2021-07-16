#!/home/mario/anaconda3/envs/project2_venv/bin python

"""
DESCRIPTION:
This script tries to gather all the steps needed to 
perform once the basecalls have been obtained.
"""

# Libraries
import os
import sys


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
    print('Renaming completed')
        

            
    

    
    


