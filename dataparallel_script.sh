#!/bin/bash
#PBS -l select=1:ncpus=8:mem=48gb:ngpus=2
#PBS -l walltime=00:30:00

module load anaconda3/personal

source activate project2_venv

python3 $HOME/project_2/assembly_tutorial_self.py $CUDA_VISIBLE_DEVICES