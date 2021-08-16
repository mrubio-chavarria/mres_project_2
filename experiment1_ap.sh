#!/bin/bash
#PBS -lselect=1:ncpus=32:mem=192gb:ngpus=8
#PBS -lwalltime=24:00:00
PBS_ARRAY_INDEX=3

# Load dependencies
module load anaconda3/personal
source activate project2_venv

# Check GPUs
echo "Available GPUs: $CUDA_VISIBLE_DEVICES"

# Check array index
echo "Array job ID: $PBS_ARRAY_INDEX"

# Launch script
echo "Launch script"
python3 $HOME/project_2/experiment1_ap.py $CUDA_VISIBLE_DEVICES $HOME/project_2/databases/working_ap $HOME/project_2/complete_ap_exp1_gamma_$PBS_ARRAY_INDEX.tsv $PBS_ARRAY_INDEX $HOME/project_2/databases/working_ap/saved_models/1_model_2021-08-14_14:15:34.497265_3323e367-2fa9-4c62-8256-5a67d38ba2fc.pt
