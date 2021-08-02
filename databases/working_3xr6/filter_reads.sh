#!/bin/bash
#PBS -lselect=1:ncpus=4:mem=4gb
#PBS -lwalltime=12:00:00

module load anaconda3/personal

source activate project2_venv 

cd $HOME/project_2/databases/working_3xr6

flowcell=flowcell1

# Launch the resquiggling for every folder
echo "Job index: 1"
echo "Flocell: $flowcell"
python3 filter_reads.py $HOME/project_2/databases/working_3xr6 $flowcell
