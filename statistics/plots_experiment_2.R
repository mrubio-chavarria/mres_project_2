
# Libraries
library(tidyverse)

# Load data
# AP
data.1 <- read.csv2("/home/mario/Documentos/Imperial/Project_2/output_experiments/experiment2/ap/round1/complete_ap_exp1_gamma_1.tsv", sep="\t", header=F)
data.1$dataset <-  "ap"
colnames(data.1) <- c("step", "loss", "cer", "train", "dataset")
data.1$step <- as.numeric(data.1$step)
data.1$epoch <- trunc(data.1$step / 10000)
data.2 <- read.csv2("/home/mario/Documentos/Imperial/Project_2/output_experiments/experiment2/ap/round2/complete_ap_exp1_gamma_2.tsv", sep="\t", header=F)
data.2$dataset <- "ap"
colnames(data.2) <- c("step", "loss", "cer", "train", "dataset")
data.2$step <- 20000:40001
data.2$epoch <- trunc(data.2$step / 10000) 
data.ap <- rbind(data.1, data.2)
data.ap$loss <- as.numeric(data.ap$loss)
data.ap$cer <- as.numeric(data.ap$cer)
data.ap$dataset <- factor(data.ap$dataset, levels=c("ap", "3xr6", "both"))
data.ap$train <- as.logical(data.ap$train)
# Validation
data.ap.validation <- rbind(data.ap[1, ], data.ap[!data.ap$train, ])
data.ap.validation$epoch <- c(0, 0, 1, 2, 3)
data.ap.validation$step <- c(0, 9999, 19999, 29999, 39999)


# 3xr6
data.1 <- read.csv2("/home/mario/Documentos/Imperial/Project_2/output_experiments/experiment2/3xr6/round1/complete_3xr6_exp1_gamma_1.tsv", sep="\t", header=F)
data.1$dataset <-  "3xr6"
colnames(data.1) <- c("step", "loss", "cer", "train", "dataset")
data.1$epoch <- trunc(data.1$step / 10000)
data.1$step <- as.numeric(data.1$step)
data.2 <- read.csv2("/home/mario/Documentos/Imperial/Project_2/output_experiments/experiment2/3xr6/round2/complete_3xr6_exp1_gamma_2.tsv", sep="\t", header=F)
data.2$dataset <-  "3xr6"
colnames(data.2) <- c("step", "loss", "cer", "train", "dataset")
data.2$step <- 20000:40001
data.2$epoch <- trunc(data.2$step / 10000)
data.3xr6 <- rbind(data.1, data.2)
data.3xr6$loss <- as.numeric(data.3xr6$loss)
data.3xr6$cer <- as.numeric(data.3xr6$cer)
data.3xr6$dataset <- factor(data.3xr6$dataset, levels=c("ap", "3xr6", "both"))
data.3xr6$train <- as.logical(data.3xr6$train)
# Validation
data.3xr6.validation <- rbind(data.3xr6[1, ], data.3xr6[!data.3xr6$train, ])
data.3xr6.validation$epoch <- c(0, 0, 1, 2, 3)
data.3xr6.validation$step <- c(0, 9999, 19999, 29999, 39999)


# Both
data.1 <- read.csv2("/home/mario/Documentos/Imperial/Project_2/output_experiments/experiment2/both/round1/complete_both_exp1_gamma_1.tsv", sep="\t", header=F)
data.1$dataset <-  "both"
colnames(data.1) <- c("step", "loss", "cer", "train", "dataset")
data.1$epoch <- trunc(data.1$step / 10000)
data.1$step <- as.numeric(data.1$step)
data.2 <- read.csv2("/home/mario/Documentos/Imperial/Project_2/output_experiments/experiment2/both/round2/complete_both_exp1_gamma_2.tsv", sep="\t", header=F)
data.2$dataset <-  "both"
colnames(data.2) <- c("step", "loss", "cer", "train", "dataset")
data.2$step <- 20000:40001
data.2$epoch <- trunc(data.2$step / 10000)
data.both <- rbind(data.1, data.2)
data.both$loss <- as.numeric(data.both$loss)
data.both$cer <- as.numeric(data.both$cer)
data.both$dataset <- factor(data.both$dataset, levels=c("ap", "3xr6", "both"))
data.both$train <- as.logical(data.both$train)
# Validation
data.both.validation <- rbind(data.both[1, ], data.both[!data.both$train, ])
data.both.validation$epoch <- c(0, 0, 1, 2, 3)
data.both.validation$step <- c(0, 9999, 19999, 29999, 39999)


# Combine
data <- rbind(data.ap, data.3xr6, data.both)

# PLOTS
text_size=30
validation_line_size=1.25
# Loss
ggplot() +
  geom_line(data = data.3xr6[data.3xr6$train, ], aes(x = step, y = log(loss, 10), alpha=0.5), color = "firebrick1") +
  geom_line(data = data.both[data.both$train, ], aes(x = step, y = log(loss, 10), alpha=0.5), color = "forestgreen") +
  geom_line(data = data.ap[data.ap$train, ], aes(x = step, y = log(loss, 10), alpha=0.5), color = "steelblue") +
  geom_point(data = data.3xr6.validation, aes(x = step, y = log(loss, 10)), color = "red", shape=0, size=5) +
  geom_point(data = data.both.validation, aes(x = step, y = log(loss, 10)), color = "green", shape=1, size=5) +
  geom_point(data = data.ap.validation, aes(x = step, y = log(loss, 10)), color = "blue", shape=2, size=5) +
  geom_line(data = data.3xr6.validation, aes(x = step, y = log(loss, 10)), color = "red") +
  geom_line(data = data.both.validation, aes(x = step, y = log(loss, 10)), color = "green") +
  geom_line(data = data.ap.validation, aes(x = step, y = log(loss, 10)), color = "blue") +
  theme_bw() +
  theme(text = element_text(size=text_size)) +
  labs(x = 'Batch', y = bquote(log[10]*'(CTC Loss)')) +
  theme(legend.position = 'none')

  
# CER
ggplot() +
  geom_line(data = data.3xr6[data.3xr6$train, ], aes(x = step, y = cer), color = "firebrick1", alpha=0.5) +
  geom_line(data = data.ap[data.ap$train, ], aes(x = step, y = cer), color = "steelblue", alpha=0.7) +
  geom_line(data = data.both[data.both$train, ], aes(x = step, y = cer), color = "forestgreen", alpha=0.5) +
  geom_point(data = data.3xr6.validation, aes(x = step, y = cer), color = "red", shape=0, size=5) +
  geom_point(data = data.both.validation, aes(x = step, y = cer), color = "green", shape=1, size=5) +
  geom_point(data = data.ap.validation, aes(x = step, y = cer), color = "blue", shape=2, size=5) +
  geom_line(data = data.3xr6.validation, aes(x = step, y = cer), color = "red") +
  geom_line(data = data.both.validation, aes(x = step, y = cer), color = "green") +
  geom_line(data = data.ap.validation, aes(x = step, y = cer), color = "blue") +
  theme_bw() +
  theme(text = element_text(size=text_size)) +
  labs(x = 'Batch', y = 'Average CER')

# Testing data
test.data <- read.csv2("/home/mario/Documentos/Imperial/Project_2/output_experiments/test_data.tsv", sep="\t", header=T)
test.data$step <- NULL
test.data$loss <- as.numeric(test.data$loss)
test.data$error <- as.numeric(test.data$error)
test.data$dataset <- factor(test.data$dataset, levels=c("AP", "3xr6", "Both"))
test.data$model <- factor(test.data$model, levels=c("AP", "3xr6", "Both"))
# Dataset: AP
test.data.ap <- test.data[test.data$dataset == "AP", ]
# Loss
loss.ap.ap <- mean(test.data.ap[test.data.ap$model == "AP", ]$loss)
loss.ap.3xr6 <- mean(test.data.ap[test.data.ap$model == "3xr6", ]$loss)
loss.ap.both <- mean(test.data.ap[test.data.ap$model == "Both", ]$loss)
# Error
avg.error.ap.ap <- mean(test.data.ap[test.data.ap$model == "AP", ]$error)
avg.error.ap.3xr6 <- mean(test.data.ap[test.data.ap$model == "3xr6", ]$error)
avg.error.ap.both <- mean(test.data.ap[test.data.ap$model == "Both", ]$error)
c(avg.error.ap.ap, avg.error.ap.3xr6, avg.error.ap.both)
# Dataset: 3xr6
test.data.3xr6 <- test.data[test.data$dataset == "3xr6", ]
# Loss
loss.3xr6.ap <- mean(test.data.3xr6[test.data.3xr6$model == "AP", ]$loss)
loss.3xr6.3xr6 <- mean(test.data.3xr6[test.data.3xr6$model == "3xr6", ]$loss)
loss.3xr6.both <- mean(test.data.3xr6[test.data.3xr6$model == "Both", ]$loss)
# Error
avg.error.3xr6.ap <- mean(test.data.3xr6[test.data.3xr6$model == "AP", ]$error)
avg.error.3xr6.3xr6 <- mean(test.data.3xr6[test.data.3xr6$model == "3xr6", ]$error)
avg.error.3xr6.both <- mean(test.data.3xr6[test.data.3xr6$model == "Both", ]$error)
c(avg.error.3xr6.ap, avg.error.3xr6.3xr6, avg.error.3xr6.both)
# Dataset: Both
test.data.both <- test.data[test.data$dataset == "Both", ]
# Loss
loss.both.ap <- mean(test.data.both[test.data.both$model == "AP", ]$loss)
loss.both.3xr6 <- mean(test.data.both[test.data.both$model == "3xr6", ]$loss)
loss.both.both <- mean(test.data.both[test.data.both$model == "Both", ]$loss)
# Error
avg.error.both.ap <- mean(test.data.both[test.data.both$model == "AP", ]$error)
avg.error.both.3xr6 <- mean(test.data.both[test.data.both$model == "3xr6", ]$error)
avg.error.both.both <- mean(test.data.both[test.data.both$model == "Both", ]$error)
c(avg.error.both.ap, avg.error.both.3xr6, avg.error.both.both)



