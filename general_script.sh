#!/bin/bash
#PBS -l select=1:ncpus=8:mem=16gb
#PBS -l walltime=01:00:00

module load anaconda3/personal

python3 assembly_tutorial_self.py