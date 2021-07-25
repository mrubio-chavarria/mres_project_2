#!/bin/bash
#PBS -lselect=1:ncpus=4:mem=4gb
#PBS -lwalltime=01:00:00

# Load dependencies
module load anaconda3/personal

source activate project2_venv

cd $HOME/project_2

y | conda install pandas
