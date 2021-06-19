#!/bin/bash
#PBS -l select=1:ncpus=2:mem=4gb
#PBS -l walltime=04:00:00
#PBS -J 0-13

# Load modules
module load parallel/default
module load anaconda3/personal

# Set the current working directory
cd $HOME/project_2

python3 main.py