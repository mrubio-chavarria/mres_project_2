#!/bin/bash
#PBS -lselect=1:ncpus=4:mem=4gb
#PBS -lwalltime=10:00:00


# Load dependencies
module load anaconda3/personal

source activate project2_venv

cd $HOME/project_2/databases/working_3xr6/reads

rm -r flowcell3
tar -xf flowcell3.tar.gz