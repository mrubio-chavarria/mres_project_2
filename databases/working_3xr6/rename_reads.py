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
    
    database_dir = sys.argv[1]
    flowcell = sys.argv[2]
    
    # database_dir = f'/home/mario/Projects/project_2/databases/working_3xr6'
    # n_processes = 2
    # flowcell = 'flowcell3'

    # Filter files below the q score threshold
    print('***************************************************************************************')
    print('Rename read files')
    print('Flowcell:', flowcell)
    print('***************************************************************************************')
    single_reads_folder = database_dir + '/' + 'reads' + '/' + flowcell + '/' + 'single'
    single_reads_subfolders = [single_reads_folder + '/' + folder 
        for folder in os.listdir(single_reads_folder) 
        if not folder.endswith('txt') and not folder.endswith('index')]
    # Rename files
    for subfolder in single_reads_subfolders:
        for idx, file in enumerate(os.listdir(subfolder)):
            if not file.endswith('fast5'):
                continue
            old_filename = subfolder + '/' + file
            new_filename = subfolder + '/' + f'read{str(idx)}.fast5'
            os.rename(old_filename, new_filename)
    print('Renaming completed')
        

            
    

    
    


