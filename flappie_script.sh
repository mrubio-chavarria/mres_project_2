#!/bin/bash

# Name Flappie
shopt -s expand_aliases
alias flappie='/home/mario/Projects/flappie/flappie'

# Set the current working directory
cd "$(dirname "$0")"

# Run OpenBLAS in single threaded mode
export OPENBLAS_NUM_THREADS=1

flappie --model help

# Launch the model
# flappie --model r941_native --uuid acinetobacter_data/selected_reads > acinetobacter_data/basecalled_reads/basecalled_acinetobacter.fq
# find reads -name \*.fast5 | parallel -P $(nproc) -X flappie > basecalls.fq
find /home/mario/Projects/project_2/databases/natural_flappie_r941_native_ap/reads -name \*.fast5 | parallel -P 3 -X /home/mario/Projects/flappie/flappie > basecalls.fq