#!/venv/bin python

"""
DESRIPTION:
The code below is to prepare the files for basecalling.
"""

# Libraries
import os
import sys
import shutil


if __name__ == '__main__':

    # Pass as argument the prospective number of jobs
    n_jobs = int(sys.argv[1])
    # List files in directory
    reads_directory = sys.argv[2]
    filenames = os.listdir(reads_directory)
    # Rename files with the indices
    for idx, filename in enumerate(filenames):
        os.rename(
            f'{reads_directory}/{filename}',
            f'{reads_directory}/{str(idx)}_{filename}'
        )
    # Group the files in folder based on the
    # number of jobs
    filenames = list(sorted(os.listdir(reads_directory), key=lambda x: int(x.split('_')[0])))
    group_length = int(round(len(filenames) / n_jobs))
    groups = [filenames[i:i + group_length] 
        if i != (len(filenames) - group_length - 1) else filenames[i::]
        for i in range(0, len(filenames) - group_length, group_length)]
    # Create folders to store the files renamed
    [os.mkdir(reads_directory + '/' + str(i)) for i in range(n_jobs)]
    # Move the files by group
    basecalled_reads_directory = '/'.join(reads_directory.split('/')[0:-1]) + '/' + 'basecalled_reads'
    for i in range(n_jobs):
        for file in groups[i]:
            # Move reads to directories
            shutil.move(
                reads_directory + '/' + file,
                reads_directory + '/' + str(i) + '/' + file
            )
            # Create the fastq files
            with open(basecalled_reads_directory + '/' + str(i) + '.fq', 'w') as file:
                file.write('')
    
