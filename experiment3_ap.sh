#!/bin/bash
#PBS -lselect=1:ncpus=8:mem=48gb:ngpus=2
#PBS -lwalltime=24:00:00
#PBS -J 1-6

# Load dependencies
module load anaconda3/personal
source activate project2_venv

# Check GPUs
echo "Available GPUs: $CUDA_VISIBLE_DEVICES"

# Check array index
echo "Array job ID: $PBS_ARRAY_INDEX"

# Launch script
echo "Launch script"
python3 $HOME/project_2/experiment3_ap.py $CUDA_VISIBLE_DEVICES $HOME/project_2/databases/working_ap $HOME/project_2/ap_exp3_epoch_$PBS_ARRAY_INDEX.tsv $PBS_ARRAY_INDEX
