#!/bin/bash
#PBS -lselect=1:ncpus=32:mem=124gb
#PBS -lwalltime=01:00:00

# Load dependencies
module load anaconda3/personal

source activate project2_venv

cd $HOME/databases

mv working_ap.tar.gz reference_ap.tar.gz 

tar -xf reference_ap.tar.gz
