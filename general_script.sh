#!/bin/bash
#PBS -l select=1:ncpus=8:mem=12gb
#PBS -l walltime=01:00:00

module load anaconda3/personal

python3 main_parallel.py