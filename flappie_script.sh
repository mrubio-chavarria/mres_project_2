#!/bin/bash
#PBS -l select=1:ncpus=12:mem=2gb
#PBS -l walltime=03:00:00
#PBS -J 0-13


# Set the current working directory
cd "$(dirname "$0")"

# Run OpenBLAS in single threaded mode
export OPENBLAS_NUM_THREADS=1

# Launch the model
find $HOME/project_2/databases/natural_flappie_r941_native_ap/reads/$PBS_ARRAY_INDEX -name \*.fast5 | parallel -P $(nproc) -X $HOME/project2/flappie/flappie > $HOME/project_2/databases/natural_flappie_r941_native_ap/basecalled_reads/$PBS_ARRAY_INDEX.fq