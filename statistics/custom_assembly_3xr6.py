
# Libraries
import os
from Bio import SeqIO
import mappy as mp
import pandas as pd
import csv


# Functions
def align(reference, reads):
    # Load reference to align
    aligner = mp.Aligner(reference)
    for name, seq, _ in mp.fastx_read(reads):
        for hit in aligner.map(seq): # traverse alignments
            yield name, hit.q_st,  hit.q_en,  hit.strand,  hit.ctg,  hit.ctg_len,  hit.r_st,  hit.r_en,  hit.mlen,  hit.blen,  hit.mapq, hit.NM, seq


def label_sequence(seq, start, end):
    return [(i, seq[i-start]) for i in range(start, end+1)]
    

if __name__ == "__main__":
    
    # Read Guppy reads
    workdir = "/home/mario/Documentos/Imperial/Project_2/output_experiments/assemblies"
    reference = workdir + '/' + '3xr6' + '/' + 'reference.fasta'
    reads = workdir + '/' + '3xr6' + '/' + 'multi.fastq'
    
    # Load/compute alignments
    load_alignment = True
    if load_alignment:
        file = open(workdir + '/' + '3xr6_alignment_analisys.tsv')
        data = csv.reader(file, delimiter='\t')
    else:
        alignment_data = list(align(reference, reads))
        labels = ['name', 'q_st', 'q_end', 'strand', 'ctg', 'ctg_len', 'r_st', 'r_end', 'mlen', 'blen', 'mapq', 'NM', 'seq']
        pd.DataFrame(data=alignment_data, columns=labels).to_csv(workdir + '/' + '3xr6_alignment_analisys.tsv', sep='\t', index=False)   
    
    # Record length data
    data = []
    for record in SeqIO.parse(reads, "fastq"):
        data.append((record.name, len(record.seq)))
    labels = ['id', 'seq_length']
    pd.DataFrame(data=data, columns=labels).to_csv(workdir + '/' + '3xr6_alignment_length.tsv', sep='\t', index=False)   
    print()


    