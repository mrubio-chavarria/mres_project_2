#!/bin/bash
#PBS -lselect=1:ncpus=32:mem=32gb
#PBS -lwalltime=08:00:00
#PBS -J 1-4

module load anaconda3/personal

source activate project2_venv 

cd $HOME/project_2/databases/working_ap

# Launch the resquiggling for every folder
python3 annotate_and_resquiggle_reads.py $HOME/project_2/databases/working_ap 28 flowcell$PBS_ARRAY_INDEX