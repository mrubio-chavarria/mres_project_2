#!/bin/bash
#PBS -lselect=1:ncpus=32:mem=124gb
#PBS -lwalltime=01:00:00

# Load dependencies
module load anaconda3/personal

source activate project2_venv

# Launch script
# python3 $HOME/project_2/assembly_tutorial_self_multiprocessing_cpu.py
echo "AQUI CUDA DEVICES:"
echo $CUDA_VISIBLE_DEVICES
echo "FIN"
echo "AQUI CUDA DEVICES OTRA VEZ:"
echo "$CUDA_VISIBLE_DEVICES"
echo "FIN"
