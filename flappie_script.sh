#!/bin/bash
#PBS -l select=1:ncpus=32:mem=12gb
#PBS -l walltime=04:00:00
#PBS -J 0-13

# Load modules
module load parallel

# Set the current working directory
cd "$(dirname "$0")"

# Run OpenBLAS in single threaded mode
export OPENBLAS_NUM_THREADS=1

# Launch the model
find $HOME/project_2/databases/natural_flappie_r941_native_ap/reads/$PBS_ARRAY_INDEX -name \*.fast5 | parallel -P $(nproc) -X $HOME/flappie/flappie > $HOME/project_2/databases/natural_flappie_r941_native_ap/basecalled_sequences/$PBS_ARRAY_INDEX.fq