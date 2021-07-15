#!/bin/bash
#PBS -lselect=1:ncpus=16:mem=16gb
#PBS -lwalltime=03:00:00

# Load dependencies
module load anaconda3/personal

source activate project2_venv

# Launch script
cd $HOME/project_2/databases
rm -r working_3xr6
cp -R working_3xr6_security working_3xr6
