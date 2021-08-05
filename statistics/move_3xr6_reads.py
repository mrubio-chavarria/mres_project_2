
import os
import sys
from shutil import copy

home = sys.argv[1]

route = home + "/project_2/nanopolish/working_3xr6/reads"

new_route = home + "/project_2/nanopolish/working_3xr6/reads"


for flowcell in ['flowcell1', 'flowcell2', 'flowcell3']:
    folder = route + '/' + flowcell
    for subfolder in os.listdir(folder):
        if subfolder.endswith('txt') or subfolder.endswith('index'):
            continue
        subfolder = folder + '/' + subfolder
        for idx, file in enumerate(os.listdir(subfolder)):
            if file.endswith('fast5'):
                old_file = folder + '/' + file
                new_file = new_route + '/' + flowcell + '_' + subfolder + '_' + str(idx) + '_' + file
                try:
                    copy(old_file, new_file)
                except FileNotFoundError:
                    continue




