#!/bin/bash
#PBS -lselect=1:ncpus=16:mem=96gb:ngpus=4
#PBS -lwalltime=24:00:00
PBS_ARRAY_INDEX=7

# Load dependencies
module load anaconda3/personal
source activate project2_venv

# Check GPUs
echo "Available GPUs: $CUDA_VISIBLE_DEVICES"

# Check array index
echo "Array job ID: $PBS_ARRAY_INDEX"

# Launch script
echo "Launch script"
python3 $HOME/project_2/experiment1_3xr6.py $CUDA_VISIBLE_DEVICES $HOME/project_2/databases/working_3xr6 $HOME/project_2/complete_3xr6_exp1_gamma_$PBS_ARRAY_INDEX.tsv $PBS_ARRAY_INDEX /rds/general/user/mr820/home/project_2/databases/working_3xr6/saved_models/5_model_2021-08-14_12:00:01.979539_58dee358-5112-41e0-9023-fbedc6140f0a.pt
