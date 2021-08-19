
library(DECIPHER)


data.alignment <- read.csv2("/home/mario/Documentos/Imperial/Project_2/output_experiments/assemblies/3xr6_alignment_analisys.tsv", sep='\t')

chromosomes <- unique(data.alignment$ctg)

seqs <- list()

threshold <- 0.05

consensus <- c()
for (chr in chromosomes) {
  seqs <- DNAStringSet(data.alignment[data.alignment$ctg == chr,]$seq)
  
  consensus <- c(consensus, ConsensusSequence(seqs,
                    threshold = threshold,
                    ambiguity = FALSE,
                    noConsensusChar = "N",
                    minInformation = 1 - threshold,
                    ignoreNonBases = FALSE,
                    includeTerminalGaps = FALSE))
  
}
