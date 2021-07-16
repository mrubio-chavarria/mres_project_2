#!/bin/bash
#PBS -lselect=1:ncpus=16:mem=16gb
#PBS -lwalltime=03:00:00

# Load dependencies
module load anaconda3/personal

source activate project2_venv

# Launch script
for flowcell in flowcell1 flowcell2 flowcell3
do
    cd $HOME/project_2/databases/working_3xr6/reads/$flowcell
    cp -R single_security single
done
