#!/bin/bash
#PBS -lselect=1:ncpus=32:mem=32gb
#PBS -lwalltime=12:00:00
#PBS -J 1-4


# IMPORTANT
# - You should choose less processes than folders with single reads.
# - You need more CPUs than processes to launch.

module load anaconda3/personal

source activate project2_venv 

cd $HOME/project_2/databases/working_ap

# Launch the resquiggling for every folder
echo "Job index: $PBS_ARRAY_INDEX"
python3 filter_reads.py $HOME/project_2/databases/working_ap 28 flowcell$PBS_ARRAY_INDEX