#!/bin/bash
#PBS -lselect=1:ncpus=4:mem=4gb
#PBS -lwalltime=10:00:00
#PBS -J 1-3

# Load dependencies
module load anaconda3/personal

source activate nanopolish_venv

cd $HOME/project_2

python3 statistics/move_3xr6_reads.py $HOME

