#!/home/mario/anaconda3/envs/project2_venv/bin python

"""
DESCRIPTION:
This script computes several statistics from the dataset
needed to simulate the very data.
"""

# Libraries
import os

if __name__ == "__main__":

    dataset_folder = 'databases/working_3xr6'

    # Obtain the read lengths
    file = open(dataset_folder + '/' + 'reference.fasta', 'r')
    content = file.read()
    file.close()

    content = content[2:]
    read_length = dict(map(lambda x: (x.split('\n')[0], len(x.split('\n')[1])), content.split('>')))

    