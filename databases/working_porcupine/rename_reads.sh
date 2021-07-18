#!/bin/bash
#PBS -lselect=1:ncpus=4:mem=4gb
#PBS -lwalltime=01:30:00
#PBS -J 1-4

module load anaconda3/personal

source activate project2_venv 

cd $HOME/project_2/databases/working_porcupine

python3 rename_reads.py $HOME/project_2/databases/working_porcupine flowcell$PBS_ARRAY_INDEX
