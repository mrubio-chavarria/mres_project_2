#!/bin/bash
#PBS -l select=1:ncpus=16:mem=96gb:ngpus=4
#PBS -l walltime=00:15:00

module load anaconda3/personal

python3 $HOME/project_2/test_dataparallel.py