#!/bin/bash
#PBS -lselect=1:ncpus=4:mem=4gb
#PBS -lwalltime=12:00:00
#PBS -J 1-3

module load anaconda3/personal

source activate project2_venv 

cd $HOME/project_2/databases/working_3xr6

flowcell=flowcell$PBS_ARRAY_INDEX

# Launch the resquiggling for every folder
echo "Job index: $PBS_ARRAY_INDEX"
echo "Flocell: $flowcell"
python3 filter_reads.py $HOME/project_2/databases/working_3xr6 $flowcell
