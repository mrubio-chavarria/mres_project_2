#!/bin/bash
#PBS -lselect=1:ncpus=8:mem=8gb
#PBS -lwalltime=01:00:00
#PBS -J 1-3

# Load dependencies
module load anaconda3/personal

source activate project2_venv

cd $HOME/project_2/databases

rm -r working_3xr6/reads/flowcell$PBS_ARRAY_INDEX/single_security
cp -R working_3xr6/reads/flowcell$PBS_ARRAY_INDEX/single working_3xr6/reads/flowcell$PBS_ARRAY_INDEX/single_security


