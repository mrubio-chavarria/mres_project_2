#!/bin/bash
#PBS -lselect=1:ncpus=4:mem=4gb
#PBS -lwalltime=00:30:00


module load anaconda3/personal

source activate project2_venv 

cd $HOME/project_2/databases/working_3xr6


for flowcell in flowcell1 flowcell2 flowcell3
do
  # Flatten folder structure
  cd reads/$flowcell/single
  for folder in $(ls -d */)
  do
    cp $folder/* .
    rm -r $folder
  done
  # Rename files
  python3 rename_reads.py $HOME/project_2/databases/working_3xr6 $flowcell
done