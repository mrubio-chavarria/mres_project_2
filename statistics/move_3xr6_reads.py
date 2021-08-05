
import os
import sys
from shutil import copy

home = sys.argv[1]

route = home + "/project_2/nanopolish/working_ap/reads"

new_route = home + "/project_2/nanopolish/working_ap/reads"

files = []
for flowcell in ['flowcell1', 'flowcell2', 'flowcell3', 'flowcell4']:
    folder = route + '/' + flowcell
    for idx, file in enumerate(os.listdir(folder)):
        if file.endswith('fast5'):
            old_file = folder + '/' + file
            files.append(file)
            new_file = new_route + '/' + flowcell + '_' + str(idx) + '_' + file
            try:
                copy(old_file, new_file)
            except FileNotFoundError:
                continue




