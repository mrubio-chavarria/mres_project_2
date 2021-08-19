
# Libraries
library(tidyverse)
library(ggplot2)
library(stringr)
library(jcolors)
library(RColorBrewer)
library(plot3D)
library(jcolors)


# Random model
# Load data weights
route <- "/home/mario/Projects/project_2/statistics/models/random_weights.tsv"
data <- read.csv2(route, sep='\t')
data <- sapply(data, as.numeric)
data <- data[, 2:dim(data)[2]]
data.0 <- data.frame(value = data[, 'X0'], pos=0)
data.1 <- data.frame(value = data[, 'X1'], pos=1)
data.2 <- data.frame(value = data[, 'X2'], pos=2)
data.3 <- data.frame(value = data[, 'X3'], pos=3)
data.4 <- data.frame(value = data[, 'X4'], pos=4)
data <- rbind(data.0, data.1, data.2, data.3, data.4) 

pos = 4
data %>%
ggplot() +
  geom_histogram(bins = 30, aes(x = value, y = ..count../sum(..count..)))

route <- "/home/mario/Projects/project_2/statistics/models/random_bias.tsv"
data <- read.csv2(route, sep='\t')
data <- sapply(data, as.numeric)
data <- data.frame(pos=1:(length(data)-1), value=data[2:length(data)])

data %>% 
  ggplot + 
  geom_point(aes(x = pos, y = value))


# AP
route <- "/home/mario/Projects/project_2/statistics/models/ap"
all.weights <- data.frame(model=0, value=0, pos=0)
for (file in list.files(path = route, full.names = TRUE, recursive = TRUE)) {
  if (grepl('weights_model', file, fixed = TRUE)) {
    data <- read.csv2(file, sep='\t')
    data <- sapply(data, as.numeric)
    data <- data[, 2:dim(data)[2]]
    data.0 <- data.frame(value = data[, 'X0'], pos=0)
    data.1 <- data.frame(value = data[, 'X1'], pos=1)
    data.2 <- data.frame(value = data[, 'X2'], pos=2)
    data.3 <- data.frame(value = data[, 'X3'], pos=3)
    data.4 <- data.frame(value = data[, 'X4'], pos=4)
    data <- rbind(data.0, data.1, data.2, data.3, data.4)
    data$model <- rep(str_split(str_replace(file, 'weights_', ''), '/')[[1]][9], each=dim(data)[1])
    all.weights <- rbind(all.weights, data)
  }
}
all.weights <- all.weights[2:dim(all.weights)[1], ]

all.biases <- data.frame(model=0, value=0, pos=0)
for (file in list.files(path = route, full.names = TRUE, recursive = TRUE)) {
  if (grepl('bias_model', file, fixed = TRUE)) {
    data <- read.csv2(file, sep='\t')
    data <- sapply(data, as.numeric)
    data <- data[2:length(data)]
    data.0 <- data.frame(value = data['X0'], pos=0)
    data.1 <- data.frame(value = data['X1'], pos=1)
    data.2 <- data.frame(value = data['X2'], pos=2)
    data.3 <- data.frame(value = data['X3'], pos=3)
    data.4 <- data.frame(value = data['X4'], pos=4)
    data <- rbind(data.0, data.1, data.2, data.3, data.4)
    data$model <- rep(str_split(str_replace(file, 'bias_', ''), '/')[[1]][9], each=dim(data)[1])
    all.biases <- rbind(all.biases, data)
  }
}
all.biases <- all.biases[2:dim(all.biases)[1], ]




models <- unique(all.weights$model)
# Chart weights
selected.model <- models[3]
selected.color.1 <- "cornsilk"
selected.color.2 <- "darkorange"
all.weights %>%
  ggplot() +
  geom_histogram(bins = 20, aes(x = value, y = ..count../sum(..count..), alpha=0.7), fill=selected.color.1, color=selected.color.2)  +
  theme_bw() +
  labs(x = "Weight value", y = "Density") +
  theme(legend.position = "none") +
  theme(text = element_text(size=15))


# Chart biases
relationship <- c('A', 'T', 'G', 'C', 'Blank')
all.biases$bases <- factor(sapply(all.biases$pos, function (x) {relationship[(x+1)]}), levels=relationship)
all.biases %>%
  ggplot() +
  geom_point(aes(x = bases, y = value, color=model, stroke=1.5)) +
  theme_bw() +
  labs(x = "Base", y = "Bias value") +
  theme(legend.position = "none") +
  scale_color_jcolors(palette = "pal7") +
  theme(text = element_text(size=15))

