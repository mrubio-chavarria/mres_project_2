#!/bin/bash
#PBS -lselect=1:ncpus=4:mem=24gb:ngpus=2
#PBS -lwalltime=10:00:00

# Load dependencies
module load anaconda3/personal
source activate project2_venv

# Check GPUs
echo "Available GPUs: $CUDA_VISIBLE_DEVICES"

# Launch script
echo "Launch script"
python3 $HOME/project_2/main.py $CUDA_VISIBLE_DEVICES $HOME/project_2/databases/toy_working_3xr6
