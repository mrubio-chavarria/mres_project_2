#!/bin/bash
#PBS -lselect=1:ncpus=8:mem=48gb:ngpus=2
#PBS -lwalltime=24:00:00

# Load dependencies
module load anaconda3/personal
source activate project2_venv

# Check GPUs
echo "Available GPUs: $CUDA_VISIBLE_DEVICES"

# Launch script
echo "Launch script"
id=1
python3 $HOME/project_2/experiment3_3xr6.py $CUDA_VISIBLE_DEVICES $HOME/project_2/databases/working_3xr6 $HOME/project_2/3xr6_exp3_epoch_1_$id.tsv
