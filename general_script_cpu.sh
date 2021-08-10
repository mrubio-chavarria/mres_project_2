#!/bin/bash
#PBS -lselect=1:ncpus=32:mem=64gb
#PBS -lwalltime=24:00:00

# Load dependencies
module load anaconda3/personal

source activate nanopolish_venv

cd $HOME/project_2/nanopolish/working_ap

canu -p ap -d canu_assembly genomeSize=5m -nanopore multi.fastq minInputCoverage=0 stopOnLowCoverage=0 useGrid=false