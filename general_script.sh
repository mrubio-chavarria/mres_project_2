#!/bin/bash
#PBS -l select=1:ncpus=1:mem=12gb
#PBS -l walltime=01:00:00

module load anaconda3/personal

conda create --name project2_venv

source activate project2_venv

conda config --add channels conda-forge bioconda

conda install pytorch=1.8 --yes 

yes | pip3 install ont-fast5-api