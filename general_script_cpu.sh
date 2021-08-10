#!/bin/bash
#PBS -lselect=1:ncpus=8:mem=32gb
#PBS -lwalltime=10:00:00

# Load dependencies
module load anaconda3/personal

source activate nanopolish_venv

cd $HOME/project_2/nanopolish/working_3xr6

canu -p 3xr6 -d canu_assembly genomeSize=5m -nanopore multi.fastq minInputCoverage=0 stopOnLowCoverage=0 useGrid=false