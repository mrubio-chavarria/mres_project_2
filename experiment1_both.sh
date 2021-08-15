#!/bin/bash
#PBS -lselect=1:ncpus=16:mem=96gb:ngpus=4
#PBS -lwalltime=24:00:00
PBS_ARRAY_INDEX=4

# Load dependencies
module load anaconda3/personal
source activate project2_venv

# Check GPUs
echo "Available GPUs: $CUDA_VISIBLE_DEVICES"

# Check array index
echo "Array job ID: $PBS_ARRAY_INDEX"

# Launch script
echo "Launch script"
python3 $HOME/project_2/experiment1_both.py $CUDA_VISIBLE_DEVICES $HOME/project_2/databases/working_ap $HOME/project_2/databases/working_3xr6 $HOME/project_2/complete_both_exp1_gamma_$PBS_ARRAY_INDEX.tsv $PBS_ARRAY_INDEX /rds/general/user/mr820/home/project_2/databases/working_both/saved_models/3_model_2021-08-14_12:34:37.446454_ce5cabf5-093a-47e3-940c-ae417fed30b8.pt
