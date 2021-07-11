#!/bin/bash
#PBS -lselect=1:ncpus=8:mem=8gb
#PBS -lwalltime=03:00:00

# NOTE: You should choose less processes than folders with 
# single reads.

module load anaconda3/personal

source activate project2_venv 

cd $HOME/project_2/databases/working_3xr6

# Launch the resquiggling for every folder
echo $(nproc)
#python3 filter_reads.py $HOME/project_2/databases/working_3xr6 8 flowcell1