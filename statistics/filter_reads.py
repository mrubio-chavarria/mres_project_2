
import os
from Bio import SeqIO

if __name__ == "__main__":

    # Reference route
    reads = "/home/mario/Documentos/Imperial/Project_2/output_experiments/assemblies/3xr6/multi.fastq"
    filtered_reads = "/home/mario/Documentos/Imperial/Project_2/output_experiments/assemblies/3xr6/filtered_reads.fastq"

    records = []
    for record in SeqIO.parse(reads, "fastq"):
        if len(record.seq) > 120:
            continue
        records.append(record)
    
    print()
    SeqIO.write(records, filtered_reads, "fastq")
    print()

