#!/bin/bash
#PBS -lselect=1:ncpus=1:mem=4gb
#PBS -lwalltime=03:00:00

# Load dependencies
module load anaconda3/personal

source activate project2_venv

# Convert from multi to single format
flowcell=flowcell2
reads_folder=$PWD/reads/$flowcell

echo "Convert from multi to single format"
multi_to_single_fast5 --input_path $reads_folder/multi --save_path $reads_folder/single
echo "Conversion completed"