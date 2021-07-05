#!/bin/bash
#PBS -lselect=1:ncpus=32:mem=124gb
#PBS -lwalltime=01:00:00

# Load dependencies
module load anaconda3/personal

source activate project2_venv

echo "Available GPUs: $CUDA_VISIBLE_DEVICES"

# Launch script
python3 $HOME/project_2/assembly_tutorial_self_multiprocessing_cpu.py $CUDA_VISIBLE_DEVICES
