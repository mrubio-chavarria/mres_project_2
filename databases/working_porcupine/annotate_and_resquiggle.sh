#!/bin/bash
#PBS -lselect=1:ncpus=8:mem=8gb
#PBS -lwalltime=03:00:00
#PBS -J 0-24

module load anaconda3/personal

source activate project2_venv 

cd $HOME/project_2/databases/working_porcupine

# Launch the resquiggling for every folder
python3 annotate_and_resquiggle_reads.py $HOME/project_2/databases/working_porcupine 8 flowcell1 $PBS_ARRAY_INDEX