#!/bin/bash
#PBS -lselect=1:ncpus=8:mem=48gb:ngpus=2
#PBS -lwalltime=03:00:00

# Load dependencies
module load anaconda3/personal

source activate taiyaki_venv

yes | conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

