#!/bin/bash
#PBS -lselect=1:ncpus=32:mem=62gb
#PBS -lwalltime=06:00:00

# Load dependencies
module load anaconda3/personal

source activate project2_venv

python3 $HOME/project_2/statistics/extract_q_scores.py $HOME/project_2/databases/working_3xr6 32
