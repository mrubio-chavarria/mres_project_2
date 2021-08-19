
# Libraries
library(rjson)
library(tidyverse)
library(ggplot2)
library(ShortRead)
library(RColorBrewer)
library(graphics)
library(scico)


# Directory
setwd('/home/mario/Projects/project_2')

# AP
# Load summaries
sum1 <- read.csv2('statistics/sequencing_summaries/ap/sequencing_summary_1.txt', sep='\t')
sum2 <- read.csv2('statistics/sequencing_summaries/ap/sequencing_summary_2.txt', sep='\t')
sum3 <- read.csv2('statistics/sequencing_summaries/ap/sequencing_summary_3.txt', sep='\t')
sum4 <- read.csv2('statistics/sequencing_summaries/ap/sequencing_summary_4.txt', sep='\t')
summary <- rbind(sum1, sum2, sum3, sum4)
summary$mean_q_score <- as.numeric(summary$mean_qscore_template)
summary$mean_qscore_template <- NULL
summary$sequence_length_template <- as.numeric(summary$sequence_length_template)
# Load fast5 metadata
meta <- read.csv2('statistics/metadata_ap.tsv', sep='\t')
meta$signal_matching_score <- as.numeric(meta$signal_matching_score)
meta$aligned_section_length <- meta$aligned_section_length / meta$raw_signal_length
meta$read_accuracy <- meta$num_matches / (meta$num_matches + meta$num_mismatches + meta$num_deletions + meta$num_insertions)
meta$at_content <- as.numeric(meta$at_content)
meta$gc_content <- as.numeric(meta$gc_content)
meta$failed_alignment <- as.logical(meta$failed_alignment)
meta$failed_parsing <- as.logical(meta$failed_parsing)
meta$mean_q_score <- NULL
# Load q-scores
qscore.ap <- read.csv2('statistics/qs_record_ap/q_score_record_ap_4162206.tsv', sep='\t')
qscore.ap$X <- NULL
qscore.ap$mean_q_score_identity <- as.numeric(qscore.ap$mean_q_score)
qscore.ap$mean_q_score <- NULL
# Combine data
data_ap <- merge(x = meta, y = summary[, c('read_id', 'mean_q_score')], by = 'read_id')

# Obtain list of reads beyond the threshold
threshold <- 7
filtered.reads <- data_ap[, c('read_id', 'mean_q_score')] %>% filter(mean_q_score >= threshold)
# write.table(filtered.reads, file='statistics/filtered_reads_ap.tsv', sep="\t", row.names=F)

# 3xr6
# Load summaries
sum1 <- read.csv2('statistics/sequencing_summaries/3xr6/sequencing_summary_1.txt', sep='\t')
sum2 <- read.csv2('statistics/sequencing_summaries/3xr6/sequencing_summary_2.txt', sep='\t')
sum3 <- read.csv2('statistics/sequencing_summaries/3xr6/sequencing_summary_3.txt', sep='\t')
summary <- rbind(sum1, sum2, sum3, sum4)
summary$mean_q_score <- as.numeric(summary$mean_qscore_template)
summary$mean_qscore_template <- NULL
summary$sequence_length_template <- as.numeric(summary$sequence_length_template)
# Load fast5 metadata
meta <- read.csv2('statistics/metadata_3xr6.tsv', sep='\t')
meta$signal_matching_score <- as.numeric(meta$signal_matching_score)
meta$aligned_section_length <- meta$aligned_section_length / meta$raw_signal_length
meta$read_accuracy <- meta$num_matches / (meta$num_matches + meta$num_mismatches + meta$num_deletions + meta$num_insertions)
meta$at_content <- as.numeric(meta$at_content)
meta$gc_content <- as.numeric(meta$gc_content)
meta$failed_alignment <- as.logical(meta$failed_alignment)
meta$failed_parsing <- as.logical(meta$failed_parsing)
meta$mean_q_score <- NULL
# Load q-scores
qscore.3xr6 <- read.csv2('statistics/qs_record_3xr6/q_score_record_3xr6.tsv', sep='\t')
qscore.3xr6$X <- NULL
qscore.3xr6$mean_q_score <- as.numeric(qscore.3xr6$mean_q_score)
# Combine data
data_3xr6 <- merge(x = meta, y = summary[, c('read_id', 'mean_q_score')], by = 'read_id')

# 3xr6 alignment data
data.alignment <- read.csv2("/home/mario/Documentos/Imperial/Project_2/output_experiments/assemblies/3xr6_alignment_length.tsv", sep='\t')
data.alignment$seq <- NULL
filtered.ids <- data.alignment[data.alignment$seq_length <= 120, ]$id
total.ids <- data_3xr6[data_3xr6$aligned_section_length > 0.7, ]$read_id
mean(total.ids %in% filtered.ids)


# Obtain list of reads beyond the threshold
threshold <- 7
filtered.reads <- data_3xr6[, c('read_id', 'mean_q_score')] %>% filter(mean_q_score >= threshold)
# write.table(filtered.reads, file='statistics/filtered_reads_3xr6.tsv', sep="\t", row.names=F)

text_size=30
line_size=2
# ------------------------------------------------------------------------------
# 2D histogram of read length vs q  score
# ------------------------------------------------------------------------------
# AP
data_ap %>%
  ggplot(aes(x = sequence_length, y = mean_q_score)) + 
  geom_hex() +
  theme_bw() +
  theme(text = element_text(size=text_size)) +
  labs(x = 'Sequence length (bases)', y = 'Mean base Q score', fill='Counts') +
  scale_fill_gradientn(colours=hcl.colors(10, palette='YlGnBu')) +
  geom_hline(yintercept=30, linetype="dashed", color = "red", size=line_size) +
  geom_hline(yintercept=20, linetype="dashed", color = "red", size=line_size) +
  geom_hline(yintercept=10, linetype="dashed", color = "red", size=line_size) +
  geom_hline(yintercept=7, linetype="dashed", color = "red", size=line_size)
# 3xr6
data_3xr6 %>%
  ggplot(aes(x = sequence_length, y = mean_q_score)) + 
  geom_hex() +
  theme_bw() +
  theme(text = element_text(size=text_size)) +
  labs(x = 'Sequence length (bases)', y = 'Mean base Q score', fill='Counts') +
  scale_fill_gradientn(colours=hcl.colors(10, palette='YlGnBu')) +
  geom_hline(yintercept=20, linetype="dashed",color = "red", size=line_size) +
  geom_hline(yintercept=10, linetype="dashed",color = "red", size=line_size) +
  geom_hline(yintercept=7, linetype="dashed",color = "red", size=line_size)


# ------------------------------------------------------------------------------
# Histogram q score with density
# ------------------------------------------------------------------------------
selected.color.ap.1 <- "steelblue"
selected.color.ap.2 <- "lightblue"
selected.color.3xr6.1 <- "firebrick1"
selected.color.3xr6.2 <- "lightsalmon"
alpha <- 0.7
binwidth <- 1.5
ggplot() +
  geom_histogram(data=data_ap[!data_ap$failed_parsing, ], 
                 binwidth=binwidth, 
                 aes(x=mean_q_score, y=100 * ..count../sum(..count..), alpha=alpha), 
                 color=selected.color.ap.1, fill=selected.color.ap.2, ) +
  theme_bw() +
  theme(text = element_text(size=text_size)) +
  geom_histogram(data=data_3xr6[!data_3xr6$failed_parsing, ],
                 binwidth=binwidth,
                 aes(x=mean_q_score, y=100 * ..count../sum(..count..)),
                 color=selected.color.3xr6.1, fill=selected.color.3xr6.2, alpha=alpha) +
  geom_vline(
    linetype = "dotdash", color = "black", size = line_size,
    xintercept = median(data_ap[!data_ap$failed_parsing, ]$mean_q_score)
  ) + 
  geom_vline(
    linetype = "dotdash", color = "black", size = line_size,
    xintercept = median(data_3xr6[!data_3xr6$failed_parsing, ]$mean_q_score)
  ) +
  # Introduce accuracies
  geom_vline(
    linetype = "dashed", color = "olivedrab4", size = line_size,
    xintercept = 7
  ) +
  labs(x = 'Mean base Q-score', y = 'Density (%)') +
  theme(legend.position = "none")


# ------------------------------------------------------------------------------
# Normalised histogram signal matching score
# ------------------------------------------------------------------------------
selected.color.ap.1 <- "steelblue"
selected.color.ap.2 <- "lightblue"
selected.color.3xr6.1 <- "firebrick1"
selected.color.3xr6.2 <- "lightsalmon"
alpha <- 0.7
binwidth <- 0.3
# These are unaligned reads, the -1 is artificial
selected.data_3xr6 <- data_3xr6 %>% filter(signal_matching_score != -1)
ggplot() +
  geom_histogram(data=data_ap[!data_ap$failed_parsing, ], 
                 binwidth=binwidth, 
                 aes(x=signal_matching_score, y=100 * ..count../sum(..count..), alpha=alpha), 
                 color=selected.color.ap.1, fill=selected.color.ap.2, ) +
  theme_bw() +
  theme(text = element_text(size=text_size)) +
  geom_histogram(data=selected.data_3xr6[!selected.data_3xr6$failed_parsing, ],
                 binwidth=binwidth,
                 aes(x=signal_matching_score, y=100 * ..count../sum(..count..)),
                 color=selected.color.3xr6.1, fill=selected.color.3xr6.2, alpha=alpha) +
  geom_vline(
    linetype = "dotdash", color = "black", size = 2,
    xintercept = median(data_ap[!data_ap$failed_parsing, ]$signal_matching_score)
  ) + 
  geom_vline(
    linetype = "dotdash", color = "black", size = 2,
    xintercept = median(selected.data_3xr6[!selected.data_3xr6$failed_alignment, ]$signal_matching_score)
  ) +
  labs(x = 'Signal matching score', y = 'Density (%)') +
  theme(legend.position = "none")


# ------------------------------------------------------------------------------
# Normalised histogram aligned section length
# ------------------------------------------------------------------------------
selected.color.ap.1 <- "steelblue"
selected.color.ap.2 <- "lightblue"
selected.color.3xr6.1 <- "firebrick1"
selected.color.3xr6.2 <- "lightsalmon"
binwidth = 0.1
ggplot() +
  geom_histogram(data=data_ap[!data_ap$failed_parsing, ],
                 binwidth=binwidth,
                 aes(x=aligned_section_length, y=100 * ..count../sum(..count..)),
                 color=selected.color.ap.1, fill=selected.color.ap.2, alpha=0.6) +
  theme_bw() +
  theme(text = element_text(size=text_size)) +
  geom_histogram(data=data_3xr6[!data_3xr6$failed_parsing, ],
                 binwidth=binwidth,
                 aes(x=aligned_section_length, y=100 * ..count../sum(..count..)), 
                 color=selected.color.3xr6.1, fill=selected.color.3xr6.2, alpha=0.6) +
  geom_vline(
    linetype = "dotdash", color = "black", size = 2,
    xintercept = median(data_ap[!data_ap$failed_parsing, ]$aligned_section_length)
  ) + 
  geom_vline(
    linetype = "dotdash", color = "black", size = 2,
    xintercept = median(data_3xr6[!data_3xr6$failed_parsing, ]$aligned_section_length)
  ) +
  labs(x = 'Signal relative aligned region', y = 'Density (%)')


# ------------------------------------------------------------------------------
# Normalised histogram read accuracy
# ------------------------------------------------------------------------------
selected.color.ap.1 <- "steelblue"
selected.color.ap.2 <- "lightblue"
selected.color.3xr6.1 <- "firebrick1"
selected.color.3xr6.2 <- "lightsalmon"
binwidth = 0.05
# These are unaligned reads, the -1 is artificial
selected.data_3xr6 <- data_3xr6 %>% filter(signal_matching_score != -1)
ggplot() +
  geom_histogram(data=data_ap[!data_ap$failed_parsing, ], 
                 binwidth=binwidth, 
                 aes(x=read_accuracy, y=100 * ..count../sum(..count..)), 
                 color=selected.color.ap.1, fill=selected.color.ap.2, alpha=0.6) +
  theme_bw() +
  theme(text = element_text(size=text_size)) +
  geom_histogram(data=selected.data_3xr6[!selected.data_3xr6$failed_parsing, ], 
                 binwidth=binwidth, 
                 aes(x=read_accuracy, y=100 * ..count../sum(..count..)), 
                 color=selected.color.3xr6.1, fill=selected.color.3xr6.2, alpha=0.6) +
  geom_vline(
    linetype = "dotdash", color = "black", size = 2,
    xintercept = median(data_ap[!data_ap$failed_parsing, ]$read_accuracy)
  ) + 
  geom_vline(
    linetype = "dotdash", color = "black", size = 2,
    xintercept = median(selected.data_3xr6[!selected.data_3xr6$failed_parsing, ]$read_accuracy)
  ) +
  labs(x = 'Read accuracy', y = 'Density (%)')


# ------------------------------------------------------------------------------
# Normalised histogram AT & GC content
# ------------------------------------------------------------------------------
# AP
binwidth <- 100 * 0.01
selected.color.at.1 <- "darkgoldenrod"
selected.color.at.2 <- "burlywood1"
selected.color.gc.1 <- "olivedrab"
selected.color.gc.2 <- "olivedrab1"
alpha <- 0.6
data_ap[!data_ap$failed_parsing, ] %>%
  ggplot() +
  geom_histogram(binwidth=binwidth, 
                 aes(x=100 * at_content, y=100 * ..count../sum(..count..)), 
                 color=selected.color.at.1, fill=selected.color.at.2, alpha=alpha) +
  theme_bw() +
  theme(text = element_text(size=text_size)) +
  geom_histogram(binwidth=binwidth,
                 aes(x=100 * gc_content, y=100 * ..count../sum(..count..)), 
                 color=selected.color.gc.1, fill=selected.color.gc.2, alpha=alpha / 3) +
  labs(x = 'Sequence content (%)', y = 'Density (%)') +
  geom_vline(
    linetype = "dotdash", color = "black", size = 2,
    xintercept = median(100 * data_ap[!data_ap$failed_parsing, ]$at_content)
  ) +
  geom_vline(
    linetype = "dotdash", color = "black", size = 2,
    xintercept = median(100 * data_ap[!data_ap$failed_parsing, ]$gc_content)
  )
# 3xr6
binwidth <-100 * 0.02  # Data is in %
selected.color.at.1 <- "darkgoldenrod"
selected.color.at.2 <- "burlywood1"
selected.color.gc.1 <- "olivedrab"
selected.color.gc.2 <- "olivedrab1"
alpha <- 0.6
data_3xr6[!data_3xr6$failed_parsing, ] %>%
  ggplot() +
  geom_histogram(binwidth=binwidth,
                 aes(x=100 * at_content, y=100 * ..count../sum(..count..)), 
                 color=selected.color.at.1, fill=selected.color.at.2, alpha=alpha) +
  theme_bw() +
  theme(text = element_text(size=text_size)) +
  geom_histogram(binwidth=binwidth,
                 aes(x=100 * gc_content, y=100 * ..count../sum(..count..)), 
                 color=selected.color.gc.1, fill=selected.color.gc.2, alpha=alpha / 3) +
  labs(x = 'Sequence content (%)', y = 'Density (%)') +
  geom_vline(
    linetype = "dotdash", color = "black", size = 2,
    xintercept = median(100 * data_3xr6[!data_3xr6$failed_parsing, ]$at_content)
  ) +
  geom_vline(
    linetype = "dotdash", color = "black", size = 2,
    xintercept = median(100 * data_3xr6[!data_3xr6$failed_parsing, ]$gc_content)
  )
  

