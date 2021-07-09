
******************************************************************************
README
******************************************************************************

This folder contains all the data preprocessed to work train the basecallers.
The content of every folder is as follows:

- reads: the fast5 files containing the sequencing raw signal.

- basecalls: this folder contains the basecalled reads stored in the directory
reads. The basecalling has been performed with Guppy 4.5.3 and the model:
dna_r9.4.1_450bps_hac. The complete description of the basecalling process is 
in reference_3xr6.

Besides, there are some files:

- preprocess_reads.py: a Python 3 script that encodes the process of formatting 
the reads from multi to single format, copying the files to the adjacent folder,
annotation of the raw signal, resquiggling and filtering. The output is the very
single resquiggled fast5 files in reads/single and filtered_reads.txt in the 
database folder. 

- filtered_reads.txt: file containing the routes to the single fast5 files 
containing the reads with a q score at of the value of the threshold.

- reference.fasta: the file with the reference sequence against the basecalls 
are resquiggled in the reads stored in the single fast5 files.


