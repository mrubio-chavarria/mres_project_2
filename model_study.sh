#!/bin/bash
#PBS -lselect=1:ncpus=4:mem=24gb:ngpus=1
#PBS -lwalltime=06:00:00
PBS_ARRAY_INDEX="MODEL STUDY"

# Load dependencies
module load anaconda3/personal
source activate project2_venv

# Check GPUs
echo "Available GPUs: $CUDA_VISIBLE_DEVICES"

# Check array index
echo "Array job ID: $PBS_ARRAY_INDEX"

# Launch script
echo "Launch script"
python3 $HOME/project_2/model_study.py $CUDA_VISIBLE_DEVICES

