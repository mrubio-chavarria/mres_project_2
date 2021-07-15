#!/bin/bash
#PBS -lselect=1:ncpus=4:mem=4gb
#PBS -lwalltime=12:00:00
#PBS -J 0-44

# IMPORTANT
# - You should choose less processes than folders with single reads.
# - You need more CPUs than processes to launch.

module load anaconda3/personal

source activate project2_venv 

cd $HOME/project_2/databases/working_ap

# Launch the resquiggling for every folder
echo "Job index: $PBS_ARRAY_INDEX"
for flowcell in flowcell1 flowcell2 flowcell3 flowcell4
do
    python3 filter_reads.py $HOME/project_2/databases/working_ap 45 $flowcell $PBS_ARRAY_INDEX
done