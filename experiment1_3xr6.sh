#!/bin/bash
#PBS -lselect=1:ncpus=4:mem=24gb:ngpus=1
#PBS -lwalltime=24:00:00
#PBS -J 1-11


# Load dependencies
module load anaconda3/personal
source activate project2_venv

# Check GPUs
echo "Available GPUs: $CUDA_VISIBLE_DEVICES"

# Check array index
echo "Array job ID: $PBS_ARRAY_INDEX"

# Launch script
echo "Launch script"
python3 $HOME/project_2/experiment1_3xr6.py $CUDA_VISIBLE_DEVICES $HOME/project_2/databases/working_3xr6 $HOME/project_2/3xr6_exp1_gamma_$PBS_ARRAY_INDEX.tsv $PBS_ARRAY_INDEX
