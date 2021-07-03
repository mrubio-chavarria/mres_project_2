#!/bin/bash
#PBS -l select=1:ncpus=4:mem=24gb:ngpus=1
#PBS -l walltime=00:30:00

module load anaconda3/personal

source activate project2_venv

python3 $HOME/project_2/test_dataparallel.py $CUDA_VISIBLE_DEVICES