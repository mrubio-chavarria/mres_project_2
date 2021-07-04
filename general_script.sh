#!/bin/bash
#PBS -lselect=1:ncpus=16:mem=96gb:ngpus=4
#PBS -lwalltime=01:00:00

# Load dependencies
module load anaconda3/personal

# Check the availability of CUDA
if [ -z "$CUDA_VISIBLE_DEVICES" ]
then
    echo "\$CUDA_VISIBLE_DEVICES is empty"
    cuda_available="NO"
else
    echo "\$CUDA_VISIBLE_DEVICES is NOT empty"
    echo "\$CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    cuda_available="YES"
fi

# Launch script
python3 $HOME/project_2/assembly_tutorial_self_multiprocessing.py cuda_available $CUDA_VISIBLE_DEVICES