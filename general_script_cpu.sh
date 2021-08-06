#!/bin/bash
#PBS -lselect=1:ncpus=4:mem=4gb
#PBS -lwalltime=10:00:00

# Load dependencies
module load anaconda3/personal

source activate nanopolish_venv

cd $HOME/project_2

chmod -R 0777 * databases/working_3xr6/reads

rm -r nanopolish/working_3xr6/reads/*

python3 statistics/move_3xr6_reads.py $HOME

