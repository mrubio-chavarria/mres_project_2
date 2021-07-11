#!/bin/bash
#PBS -lselect=1:ncpus=1:mem=4gb
#PBS -lwalltime=03:00:00

# Load dependencies
module load anaconda3/personal

source activate project2_venv

# Convert from multi to single format
for flowcell in "flowcell1 flowcell2 flowcell3"
do
    reads_folder=$HOME/project_2/databases/working_3xr6/reads/$flowcell
    echo "*****************************************"
    echo "Convert fast5 from multi to single format"
    echo "*****************************************"
    echo "Input folder : $reads_folder/multi"
    echo "Output folder : $reads_folder/single"
    multi_to_single_fast5 --input_path $reads_folder/multi --save_path $reads_folder/single
    echo "Conversion completed"

    echo "******************************************"
    echo "Convert fastq from single to multi format"
    echo "******************************************"
    cd $HOME/project_2/databases/working_3xr6/basecalls/$flowcell
    rm multi.fastq
    cat *.fastq > multi.fastq
    echo "Merged file: multi.fastq" 
    echo "Conversion completed"
done