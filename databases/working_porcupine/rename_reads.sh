#!/bin/bash
#PBS -lselect=1:ncpus=4:mem=4gb
#PBS -lwalltime=01:30:00


module load anaconda3/personal

source activate project2_venv 

cd $HOME/project_2/databases/working_porcupine

for flowcell in flowcell1 flowcell2 flowcell3 flowcell4
do
  python3 rename_reads.py $HOME/project_2/databases/working_porcupine $flowcell
done