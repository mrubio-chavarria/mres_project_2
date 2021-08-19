
# Libraries
library(rjson)
library(tidyverse)
library(ggplot2)
library(ShortRead)
library(RColorBrewer)
library(graphics)
library(scico)


# Read experiment data

setwd("/home/mario/Documentos/Imperial/Project_2/output_experiments/experiment_3")

# Read data
# AP
data.0 <- read.csv2("DecoderChiron/ap/ap_exp3_epoch_0.tsv", sep='\t')
data.0$epochs <- 0
data.1 <- read.csv2("DecoderChiron/ap/ap_exp3_epoch_1.tsv", sep='\t')
data.1$epochs <- 1
data.2 <- read.csv2("DecoderChiron/ap/ap_exp3_epoch_2.tsv", sep='\t')
data.2$epochs <- 2
data.3 <- read.csv2("DecoderChiron/ap/ap_exp3_epoch_3.tsv", sep='\t')
data.3$epochs <- 3
data.4 <- read.csv2("DecoderChiron/ap/ap_exp3_epoch_4.tsv", sep='\t')
data.4$epochs <- 4
data.5 <- read.csv2("DecoderChiron/ap/ap_exp3_epoch_5.tsv", sep='\t')
data.5$epochs <- 5
data <- rbind(data.0, data.1, data.2, data.3, data.4, data.5)
data$loss <- as.numeric(data$loss)
data$steps <- as.numeric(data$X)
data$X <- NULL
data$epochs <- factor(data$epochs, levels = c(0, 1, 2, 3, 4, 5))
data.ap <- data

# 3xr6
data.0 <- read.csv2("DecoderChiron/3xr6/3xr6_exp3_epoch_0.tsv", sep='\t')
data.0$epochs <- 0
data.1 <- read.csv2("DecoderChiron/3xr6/3xr6_exp3_epoch_1.tsv", sep='\t')
data.1$epochs <- 1
data.2 <- read.csv2("DecoderChiron/3xr6/3xr6_exp3_epoch_2.tsv", sep='\t')
data.2$epochs <- 2
data.3 <- read.csv2("DecoderChiron/3xr6/3xr6_exp3_epoch_3.tsv", sep='\t')
data.3$epochs <- 3
data.4 <- read.csv2("DecoderChiron/3xr6/3xr6_exp3_epoch_4.tsv", sep='\t')
data.4$epochs <- 4
data.5 <- read.csv2("DecoderChiron/3xr6/3xr6_exp3_epoch_5.tsv", sep='\t')
data.5$epochs <- 5
data <- rbind(data.0, data.1, data.2, data.3, data.4, data.5)
data$loss <- as.numeric(data$loss)
data$steps <- as.numeric(data$X)
data$X <- NULL
data$epochs <- factor(data$epochs, levels = c(0, 1, 2, 3, 4, 5))
data.3xr6 <- data

# Combine
data.3xr6$dataset <- "3xr6"
data.ap$dataset <- "ap"
data <- rbind(data.ap, data.3xr6)
data$dataset <- factor(data$dataset, levels = c("ap", "3xr6"))

# Plot data
text_size=30
# AP
selected.dataset <- 'ap'
# General chart
data %>% filter(dataset == selected.dataset) %>%
  ggplot(aes(x = steps, y = loss)) + 
  theme_bw() +
  geom_line(aes(color = epochs)) +
  labs(x = 'Batch', y = 'Loss') +
  theme(text = element_text(size=text_size)) +
  theme(legend.position = "none")   
# Cross entropy only
n.epoch <- 5
epoch.data <- data %>% filter(epochs == n.epoch, dataset == selected.dataset)
medians <- c(epoch.data[1, ]$loss,
             median(epoch.data[1:500, ]$loss), 
             median(epoch.data[501:1000, ]$loss), 
             median(epoch.data[1001:1500, ]$loss),
             median(epoch.data[1501:2000, ]$loss),
             median(epoch.data[2001:2500, ]$loss))
medians <- data.frame(medians)
medians$steps <- c(0, 500, 1000, 1500, 2000, 2500)
data %>% filter(epochs == n.epoch, dataset == selected.dataset) %>%
  ggplot(aes(x = steps, y = loss)) +
  theme_bw() +
  geom_line(color = "gray") +
  geom_point(data = medians, aes(x = steps, y = medians), color = "brown") +
  geom_line(data = medians, aes(x = steps, y = medians), color = "brown") +
  labs(x = 'Batch', y = 'Loss') +
  theme(text = element_text(size=text_size)) +
  theme(legend.position = "none")   


colour <- 'steelblue'
selected.dataset <- 'ap'
# 0 epochs
n.epoch <- 0
data %>% filter(epochs == n.epoch, dataset == selected.dataset) %>%
  ggplot(aes(x = steps, y = loss)) +
  theme_bw() +
  geom_line(color = colour) +
  labs(x = 'Batch', y = 'Loss') +
  theme(text = element_text(size=text_size))

# 1 epochs
n.epoch <- 1
data %>% filter(epochs == n.epoch, dataset == selected.dataset) %>%
  ggplot(aes(x = steps, y = loss)) +
  theme_bw() +
  geom_line(color = colour) +
  labs(x = 'Batch', y = 'Loss') +
  theme(text = element_text(size=text_size))

# 2 epochs
n.epoch <- 2
data %>% filter(epochs == n.epoch, dataset == selected.dataset) %>%
  ggplot(aes(x = steps, y = loss)) +
  theme_bw() +
  geom_line(color = colour) +
  labs(x = 'Batch', y = 'Loss') +
  theme(text = element_text(size=text_size))

# 3 epochs
n.epoch <- 3
data %>% filter(epochs == n.epoch, dataset == selected.dataset) %>%
  ggplot(aes(x = steps, y = loss)) +
  theme_bw() +
  geom_line(color = colour) +
  labs(x = 'Batch', y = 'Loss') +
  theme(text = element_text(size=text_size))

# 4 epochs
n.epoch <- 4
data %>% filter(epochs == n.epoch, dataset == selected.dataset) %>%
  ggplot(aes(x = steps, y = loss)) +
  theme_bw() +
  geom_line(color = colour) +
  labs(x = 'Batch', y = 'Loss') +
  theme(text = element_text(size=text_size))

# 5 epochs
n.epoch <- 5
data %>% filter(epochs == n.epoch, dataset == selected.dataset) %>%
  ggplot(aes(x = steps, y = loss)) +
  theme_bw() +
  geom_line(color = colour) +
  labs(x = 'Batch', y = 'Loss') +
  theme(text = element_text(size=text_size))

# 3xr6
selected.dataset <- '3xr6'
# General chart
data %>% filter(dataset == selected.dataset) %>%
  ggplot(aes(x = steps, y = loss)) + 
  theme_bw() +
  geom_line(aes(color = epochs)) +
  labs(x = 'Batch', y = 'Loss') +
  theme(text = element_text(size=text_size)) +
  theme(legend.position = "none")   
# Cross entropy only
n.epoch <- 5
epoch.data <- data %>% filter(epochs == n.epoch, dataset == selected.dataset)
medians <- c(epoch.data[1, ]$loss,
             median(epoch.data[1:500, ]$loss), 
             median(epoch.data[501:1000, ]$loss), 
             median(epoch.data[1001:1500, ]$loss),
             median(epoch.data[1501:2000, ]$loss),
             median(epoch.data[2001:2500, ]$loss))
medians <- data.frame(medians)
medians$steps <- c(0, 500, 1000, 1500, 2000, 2500)
data %>% filter(epochs == n.epoch, dataset == selected.dataset) %>%
  ggplot(aes(x = steps, y = loss)) +
  theme_bw() +
  geom_line(color = "gray") +
  geom_point(data = medians, aes(x = steps, y = medians), color = "brown") +
  geom_line(data = medians, aes(x = steps, y = medians), color = "brown") +
  labs(x = 'Batch', y = 'Loss') +
  theme(text = element_text(size=text_size)) +
  theme(legend.position = "none")   


colour <- 'salmon'
selected.dataset <- '3xr6'
# 0 epochs
n.epoch <- 0
data %>% filter(epochs == n.epoch, dataset == selected.dataset) %>%
  ggplot(aes(x = steps, y = loss)) +
  theme_bw() +
  geom_line(color = colour) +
  labs(x = 'Batch', y = 'Loss') +
  theme(text = element_text(size=text_size))

# 1 epochs
n.epoch <- 1
data %>% filter(epochs == n.epoch, dataset == selected.dataset) %>%
  ggplot(aes(x = steps, y = loss)) +
  theme_bw() +
  geom_line(color = colour) +
  labs(x = 'Batch', y = 'Loss') +
  theme(text = element_text(size=text_size))

# 2 epochs
n.epoch <- 2
data %>% filter(epochs == n.epoch, dataset == selected.dataset) %>%
  ggplot(aes(x = steps, y = loss)) +
  theme_bw() +
  geom_line(color = colour) +
  labs(x = 'Batch', y = 'Loss') +
  theme(text = element_text(size=text_size))

# 3 epochs
n.epoch <- 3
data %>% filter(epochs == n.epoch, dataset == selected.dataset) %>%
  ggplot(aes(x = steps, y = loss)) +
  theme_bw() +
  geom_line(color = colour) +
  labs(x = 'Batch', y = 'Loss') +
  theme(text = element_text(size=text_size))

# 4 epochs
n.epoch <- 4
data %>% filter(epochs == n.epoch, dataset == selected.dataset) %>%
  ggplot(aes(x = steps, y = loss)) +
  theme_bw() +
  geom_line(color = colour) +
  labs(x = 'Batch', y = 'Loss') +
  theme(text = element_text(size=text_size))

# 5 epochs
n.epoch <- 5
data %>% filter(epochs == n.epoch, dataset == selected.dataset) %>%
  ggplot(aes(x = steps, y = loss)) +
  theme_bw() +
  geom_line(color = colour) +
  labs(x = 'Batch', y = 'Loss') +
  theme(text = element_text(size=text_size))



