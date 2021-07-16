#!/bin/bash
#PBS -lselect=1:ncpus=32:mem=124gb
#PBS -lwalltime=01:00:00

# Load dependencies
module load anaconda3/personal

source activate project2_venv

cd $HOME/project_2/databases


for flowcell in flowcell1 flowcell2 flowcell3 flowcell4
do
    cp -R working_ap/reads/$flowcell/single working_ap/reads/$flowcell/single_security
done


