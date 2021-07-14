#!/bin/bash
#PBS -lselect=1:ncpus=4:mem=4gb
#PBS -lwalltime=05:00:00

cd $HOME/project_2/databases/working_ap/reads

# Count reads per flowcell
for flowcell in $(ls -d flowcell*)
do
    cd $flowcell/single
    hq_total_reads=$(ls -d Q20_*.fast5 | wc -l)
    total_reads=$(ls -d *.fast5 | wc -l)
    echo "Flowcell: $flowcell"
    echo "Total reads: $total_reads"
    echo "Total high-quality reads: $hq_total_reads"
    cd ../..
done


