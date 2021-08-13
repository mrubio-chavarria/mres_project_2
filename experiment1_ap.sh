#!/bin/bash
#PBS -lselect=1:ncpus=16:mem=96gb:ngpus=4
#PBS -lwalltime=24:00:00
PBS_ARRAY_INDEX=1

# Load dependencies
module load anaconda3/personal
source activate project2_venv

# Check GPUs
echo "Available GPUs: $CUDA_VISIBLE_DEVICES"

# Check array index
echo "Array job ID: $PBS_ARRAY_INDEX"

# Launch script
echo "Launch script"
python3 $HOME/project_2/experiment1_ap.py $CUDA_VISIBLE_DEVICES $HOME/project_2/databases/working_ap $HOME/project_2/complete_ap_exp1_gamma_$PBS_ARRAY_INDEX.tsv $PBS_ARRAY_INDEX $HOME/project_2/databases/working_ap/saved_models/0_model_2021-08-13_10:18:03.395735_df985faf-7f76-4c06-94c7-9408b5a076cb.pt
