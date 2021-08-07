#!/bin/bash
#PBS -l select=1:ncpus=8:mem=24gb
#PBS -l walltime=12:00:00

module load anaconda3/personal

source activate nanopolish_venv

cd $HOME/project_2/nanopolish/working_3xr6

flye --nano-raw reads.fasta --out-dir assembly --threads 4  --iterations 0