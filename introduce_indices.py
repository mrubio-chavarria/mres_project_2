
"""
DESRIPTION:
The code below is just to introduce an index in the read filenames,
to establish a relationship with the fastq files obtained form the 
basecaller.
"""

# Libraries
import os

if __name__ == '__main__':
    # List files in directory
    reads_directory = '$HOME/project_2/databases/natural_flappie_r941_native_ap/reads'
    filenames = os.listdir(reads_directory)
    # Rename files with the indices
    for idx, filename in enumerate(filenames):
        os.rename(
            f'{reads_directory}/{filename}',
            f'{reads_directory}/{str(idx)}_{filename}'
        )