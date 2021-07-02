#!/bin/bash
#PBS -l select=1:ncpus=32:mem=16gb:ngpus=8
#PBS -l walltime=01:00:00

module load anaconda3/personal

python3 $HOME/project_2/test_dataparallel.py