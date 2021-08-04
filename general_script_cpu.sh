#!/bin/bash
#PBS -lselect=1:ncpus=4:mem=4gb
#PBS -lwalltime=01:00:00
#PBS -J 1-3

# Load dependencies
module load anaconda3/personal

source activate project2_venv

cd $HOME/project_2/databases/working_3xr6/reads

tar -xf flowcell$PBS_ARRAY_INDEX.tar.gz