
import os
import sys
from shutil import copy
from uuid import uuid4

home = sys.argv[1]

workdir = home + "/project_2"

base_folder = workdir + "/nanopolish/working_3xr6/reads"


for flowcell in ['flowcell1', 'flowcell2', 'flowcell3']:
    print(f'Processing {flowcell}')
    folder = workdir + '/databases/working_3xr6/reads' + '/' + flowcell
    for subfolder in os.listdir(folder):
        if subfolder.endswith('txt') or subfolder.endswith('index'):
            continue
        subfolder = folder + '/' + subfolder
        for idx, file in enumerate(os.listdir(subfolder)):
            if file.endswith('fast5'):
                old_file = subfolder + '/' + file
                new_file = base_folder + '/' + str(uuid4()) + '_' + file
                try:
                    print(f'New file: {new_file}')
                    print(f'Old file: {old_file}')
                    copy(old_file, new_file)
                except FileNotFoundError as e:
                    print(e)
                    continue




