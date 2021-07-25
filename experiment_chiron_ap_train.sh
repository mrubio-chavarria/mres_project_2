#!/bin/bash
#PBS -lselect=1:ncpus=8:mem=48gb:ngpus=2
#PBS -lwalltime=01:15:00

# Load dependencies
module load anaconda3/personal
source activate project2_venv

# Check GPUs
echo "Available GPUs: $CUDA_VISIBLE_DEVICES"

# Launch script
echo "Launch script"
python3 $HOME/project_2/experiment_chiron_ap_train.py $CUDA_VISIBLE_DEVICES $HOME/project_2/databases/working_ap $HOME/project_2/ap_Adam_storage_3.tsv
