#!/bin/bash
#PBS -lselect=1:ncpus=12:mem=12gb
#PBS -lwalltime=03:00:00

# Load dependencies
module load anaconda3/personal

source activate project2_venv

# Launch script
# python3 $HOME/project_2/assembly_tutorial_self_multiprocessing_cpu.py
python3 $HOME/project_2/databases/working_3xr6/preprocess_reads.py $HOME/project_2 32 flowcell3
