
# Libraries
import h5py
import os

# 3xr6
route = "/home/mario/Projects/project_2/databases/working_3xr6/reads"


ap_read_files = []
for flowcell in ['flowcell1','flowcell2', 'flowcell3']:
    flowcell_folder = route + '/' + flowcell + '/' + 'single'
    for folder in os.listdir(flowcell_folder):
        if folder.endswith('txt') or folder.endswith('index'):
            continue
        folder = flowcell_folder + '/' + folder
        for file in os.listdir(folder):
            if not file.endswith('fast5'):
                continue
            ap_read_files.append(folder + '/' + file)

for read_file in ap_read_files:
    file = h5py.File(read_file, 'r')
    print()
        
