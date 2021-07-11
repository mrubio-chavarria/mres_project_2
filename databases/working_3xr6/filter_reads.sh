#!/bin/bash
#PBS -lselect=1:ncpus=16:mem=16gb
#PBS -lwalltime=05:00:00
#PBS -J 1-2


# IMPORTANT
# - You should choose less processes than folders with single reads.
# - You need more CPUs than processes to launch.

module load anaconda3/personal

source activate project2_venv 

cd $HOME/project_2/databases/working_3xr6

# Launch the resquiggling for every folder
if [ $PBS_ARRAY_INDEX == 2 ]
then
    python3 filter_reads.py $HOME/project_2/databases/working_3xr6 12 flowcell3
else
    python3 filter_reads.py $HOME/project_2/databases/working_3xr6 12 flowcell1
fi