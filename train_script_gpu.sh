#!/bin/bash
#PBS -lselect=1:ncpus=16:mem=96gb:ngpus=4
#PBS -lwalltime=03:00:00

# Load dependencies
module load anaconda3/personal
source activate project2_venv

# Compress dataset
echo "Compressing dataset"
cd $HOME/project2/databases
tar -zcvf single.tar.gz working_3xr6/reads/single
tar -zcvf reference.fasta.tar.gz working_3xr6/reference.fasta
echo "Compression completed"

# Send dataset
echo "Transfering dataset to node"
mkdir $TMPDIR/project_2/databases
mv single.tar.gz $TMPDIR/project_2/databases/single.tar.gz
mv reference.fasta.tar.gz $TMPDIR/project_2/databases/reference.fasta.tar.gz
echo "Transference completed"

# Extract dataset
echo "Extracting dataset"
cd $TMPDIR/project_2/databases
tar -xzvf single.tar.gz
tar -xzvf reference.fasta.tar.gz

# Format the file structure
mkdir working_3xr6/reads/single
mv single working_3xr6/reads/single
mv reference.fasta working_3xr6/reference.fasta
echo "Extraction completed"

# Check GPUs
echo "Available GPUs: $CUDA_VISIBLE_DEVICES"

# Launch script
echo "Launch script"
python3 $HOME/project_2/main.py $CUDA_VISIBLE_DEVICES $TMPDIR/project_2/databases/working_3xr6
