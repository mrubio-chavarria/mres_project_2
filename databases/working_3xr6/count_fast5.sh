#!/bin/bash
#PBS -lselect=1:ncpus=4:mem=4gb
#PBS -lwalltime=05:00:00

cd $HOME/project_2/databases/working_3xr6/reads

# Count reads per flowcell
for flowcell in flowcell1 flowcell2 flowcell3
do
    cd $flowcell/single
    total_reads=0
    hq_total_reads=0
    for folder in $(ls -d */)
    do  
        cd $folder
        local_files=$(ls -d *.fast5 | wc -l)
        total_reads=$(($total_reads + $local_files))
        hq_local_files=$(ls -d kHQk_*.fast5 | wc -l)
        hq_total_reads=$(($hq_total_reads + $hq_local_files))
        cd ..
    done
    echo "Flowcell: $flowcell"
    echo "Total reads: $total_reads"
    echo "Total high-quality reads: $hq_total_reads"
    cd ../..
done


