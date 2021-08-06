#!/bin/bash
#PBS -lselect=1:ncpus=4:mem=4gb
#PBS -lwalltime=10:00:00


# Load dependencies
module load anaconda3/personal

source activate nanopolish_venv

cd $HOME/project_2

# python3 statistics/move_3xr6_reads.py $HOME

cd databases/working_3xr6

rm -R *_old
