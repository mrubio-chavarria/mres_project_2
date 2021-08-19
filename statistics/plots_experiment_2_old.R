
# Libraries
library(rjson)
library(tidyverse)
library(ggplot2)
library(stringr)
library(jcolors)

# Directory
setwd('/home/mario/Documentos/Imperial/Project_2/output_experiments')


# AP
# Adam
# Alias Experiment LR      WD    Momemtum Optimiser Scheduler
# 1     4107318    0.001   0     0        Adam      None
# 2     4112089    0.001   0     0        Adam      OneCycleLR
# 3     4146186    0.0001  0     0        Adam      None
# 4     4146203    0.0001  0     0        Adam      OneCycleLR
# 5     4146848    0.001   0.01  0        Adam      None
# Loss
result <- fromJSON(file = "experiment_2/ap/Adam/4107318/loss_ap_adam.json")
data <- data.frame(step=result[[1]]$x, loss=result[[1]]$y, optimiser='Adam',  scheduler='None')
data$experiment <- "4107318"
data$learning_rate <- 0.001
data_ap_adam <- data
result <- fromJSON(file = "experiment_2/ap/Adam/4112089/loss_ap_adam_scheduler.json")
data <- data.frame(step=result[[1]]$x, loss=result[[1]]$y, optimiser='Adam',  scheduler='None')
data$experiment <- "4112089"
data$learning_rate <- 0.001
data_ap_adam <- rbind(data_ap_adam, data)
result <- read.csv2(file = "experiment_2/ap/Adam/4146186/ap_Adam_storage.tsv", sep="\t")
result$step <- 0:(dim(result)[1]-1)
result$avgcer <- NULL
result$optimiser <- "Adam"
result$scheduler <- "None"
result$experiment <- "4146186"
result$learning_rate <- 0.0001
data_ap_adam <- rbind(data_ap_adam, result)
result <- read.csv2(file = "experiment_2/ap/Adam/4146203/ap_Adam_storage_2.tsv", sep="\t")
result$step <- 0:(dim(result)[1]-1)
result$avgcer <- NULL
result$optimiser <- "Adam"
result$scheduler <- "OneCycleLR"
result$experiment <- "4146203"
result$learning_rate <- 0.0001
data_ap_adam <- rbind(data_ap_adam, result)
result <- read.csv2(file = "experiment_2/ap/Adam/4146848/ap_Adam_storage_4.tsv", sep="\t")
result$step <- 0:(dim(result)[1]-1)
result$avgcer <- NULL
result$optimiser <- "Adam"
result$scheduler <- "None"
result$experiment <- "4146848"
result$learning_rate <- 0.001
result$X <- NULL
data_ap_adam <- rbind(data_ap_adam, result)
data_ap_adam$loss <- as.numeric(data_ap_adam$loss)
# Plot
relationship <- data.frame(experiment=c("4107318", "4112089", "4146186", "4146203", "4146848"), alias=as.factor(c(
  'Learning rate: 1E-3', 'Learning rate: 1E-3 + OneCycleLR', 'Learning rate: 1E-4', 'Learning rate: 1E-4 + OneCycleLR', 'Learning rate: 1E-3 + Weight Decay'
)))
data_ap_adam %>% merge(relationship, by="experiment") %>%
  ggplot() +
  geom_line(aes(x = step, y = loss, color=alias)) +
  theme_bw() +
  labs(x = "Batch", y = "Loss", color="Experiment") +
  theme(text=element_text(size=20)) + 
  scale_color_brewer(palette="Set1")
  

# RMSprop 
# Alias  Experiment LR       WD       Momemtum   Optimiser Scheduler
# 6      4108437    0.001    0        0          RMSprop      None
# 7      4112608    0.001    0        0          RMSprop      OneCycleLR
# 8      4146953    0.0001   0.01     0.9        RMSprop      None
# 9      4146966    0.0001   0        0.9        RMSprop      None
# 10     4146972    0.001    0.01     0.9        RMSprop      None
# Loss
result <- fromJSON(file = "experiment_2/ap/RMSprop/4108437/loss_ap_rmsprop.json")
data <- data.frame(step=result[[1]]$x, loss=result[[1]]$y, optimiser='RMSprop',  scheduler='None')
data$experiment <- "4108437"
data$learning_rate <- 0.001
data_ap_rmsprop <- data
result <- fromJSON(file = "experiment_2/ap/RMSprop/4112608/loss_ap_rmsprop_scheduler.json")
data <- data.frame(step=result[[1]]$x, loss=result[[1]]$y, optimiser='RMSprop',  scheduler='OneCycleLR')
data$experiment <- "4112608"
data$learning_rate <- 0.001
data_ap_rmsprop <- rbind(data_ap_rmsprop, data)
result <- read.csv2(file = "experiment_2/ap/RMSprop/4146953/ap_RMSprop_storage_1.tsv", sep="\t")
result$step <- 0:(dim(result)[1]-1)
result$avgcer <- NULL
result$optimiser <- "RMSprop"
result$scheduler <- "None"
result$experiment <- "4146953"
result$learning_rate <- 0.0001
result <- result[, 2:dim(result)[2]]
data_ap_rmsprop <- rbind(data_ap_rmsprop, result)
result <- read.csv2(file = "experiment_2/ap/RMSprop/4146966/ap_RMSprop_storage_2.tsv", sep="\t")
result$step <- 0:(dim(result)[1]-1)
result$avgcer <- NULL
result$optimiser <- "RMSprop"
result$scheduler <- "None"
result$experiment <- "4146966"
result <- result[, 2:dim(result)[2]]
result$learning_rate <- 0.0001
data_ap_rmsprop <- rbind(data_ap_rmsprop, result)
result <- read.csv2(file = "experiment_2/ap/RMSprop/4146972/ap_RMSprop_storage_3.tsv", sep="\t")
result$step <- 0:(dim(result)[1]-1)
result$avgcer <- NULL
result$optimiser <- "RMSprop"
result$scheduler <- "None"
result$experiment <- "4146972"
result$learning_rate <- 0.001
result <- result[, 2:dim(result)[2]]
data_ap_rmsprop <- rbind(data_ap_rmsprop, result)
# Plot
data_ap_rmsprop$loss <- as.numeric(data_ap_rmsprop$loss)
relationship <- data.frame(experiment=c("4108437", "4112608", "4146953", "4146966", "4146972"), alias=as.factor(c(
  'Learning rate: 1E-3', 'Learning rate: 1E-3 + OneCycleLR', 'Learning rate: 1E-4 + Weight Decay + Momentum', 'Learning rate: 1E-4 + Momentum', 'Learning rate: 1E-3 + Weight Decay + Momentum'
)))
data_ap_rmsprop %>% merge(relationship, by="experiment") %>%
  ggplot() +
  geom_line(aes(x = step, y = loss, color=alias)) +
  theme_bw() +
  labs(x = "Batch", y = "Loss", color="Experiment") +
  theme(text=element_text(size=20)) + 
  scale_color_brewer(palette="Set1")

# SGD 
# Alias Experiment LR       WD       Momemtum   Optimiser Scheduler
# 11    4109506    0.001    0        0          SGD      None
# 12    4112614    0.001    0        0          SGD      OneCycleLR
# 13    4147215    0.0001   0        0.9        SGD      None
# Loss
result <- fromJSON(file = "experiment_2/ap/SGD/4109506/loss_ap_sgd.json")
data <- data.frame(step=result[[1]]$x, loss=result[[1]]$y, optimiser='SGD',  scheduler='None')
data$experiment <- "4109506"
data$learning_rate <- 0.001
data_ap_sgd <- data
result <- fromJSON(file = "experiment_2/ap/SGD/4112614/loss_ap_sgd_scheduler.json")
data <- data.frame(step=result[[1]]$x, loss=result[[1]]$y, optimiser='SGD',  scheduler='OneCycleLR')
data$experiment <- "4112614"
data$learning_rate <- 0.001
data_ap_sgd <- rbind(data_ap_sgd, data)
result <- read.csv2(file = "experiment_2/ap/SGD/4147215/ap_SGD_storage_2.tsv", sep="\t")
result$step <- 0:(dim(result)[1]-1)
result$avgcer <- NULL
result$optimiser <- "SGD"
result$scheduler <- "None"
result$experiment <- "4147215"
result$learning_rate <- 0.0001
result <- result[, 2:dim(result)[2]]
data_ap_sgd <- rbind(data_ap_sgd, result)
# Plot
data_ap_sgd$loss <- as.numeric(data_ap_sgd$loss)
relationship <- data.frame(experiment=c("4109506", "4112614", "4147215"), alias=as.factor(c(
  'Learning rate: 1E-3', 'Learning rate: 1E-3 + OneCycleLR', 'Learning rate: 1E-4 + Momentum'
)))
data_ap_sgd %>% merge(relationship, by="experiment") %>%
  ggplot() +
  geom_line(aes(x = step, y = loss, color=alias)) +
  theme_bw() +
  labs(x = "Batch", y = "Loss", color="Experiment") +
  theme(text=element_text(size=20)) + 
  scale_color_brewer(palette="Set1")


# 3xr6
# Adam
# Alias Experiment LR      WD    Momemtum Optimiser Scheduler
# 1     4112576    0.001   0     0        Adam      None
# 2     4113357    0.001   0     0        Adam      OneCycleLR
# 3     4147521    0.0001  0.01  0        Adam      None
# 4     4147563    0.0001  0     0        Adam      OneCycleLR
result <- fromJSON(file = "experiment_2/3xr6/Adam/4112576/loss_3xr6_adam.json")
data <- data.frame(step=result[[1]]$x, loss=result[[1]]$y, optimiser='Adam',  scheduler='None')
data$experiment <- "4112576"
data$learning_rate <- 0.001
data_3xr6_adam <- data
result <- fromJSON(file = "experiment_2/3xr6/Adam/4113357/loss_3xr6_adam_scheduler.json")
data <- data.frame(step=result[[1]]$x, loss=result[[1]]$y, optimiser='Adam',  scheduler='OneCycleLR')
data$experiment <- "4113357"
data$learning_rate <- 0.001
data_3xr6_adam <- rbind(data_3xr6_adam, data)
result <- read.csv2(file = "experiment_2/3xr6/Adam/4147521/3xr6_Adam_storage_1.tsv", sep="\t")
result$step <- 0:(dim(result)[1]-1)
result$avgcer <- NULL
result$optimiser <- "Adam"
result$scheduler <- "None"
result$experiment <- "4147521"
result$learning_rate <- 0.0001
result$X <- NULL
data_3xr6_adam <- rbind(data_3xr6_adam, result)
result <- read.csv2(file = "experiment_2/3xr6/Adam/4147563/3xr6_Adam_storage_2.tsv", sep="\t")
result$step <- 0:(dim(result)[1]-1)
result$avgcer <- NULL
result$optimiser <- "Adam"
result$scheduler <- "OneCycleLR"
result$experiment <- "4147563"
result$learning_rate <- 0.0001
result$X <- NULL
data_3xr6_adam <- rbind(data_3xr6_adam, result)
# Plot
data_3xr6_adam$loss <- as.numeric(data_3xr6_adam$loss)
relationship <- data.frame(experiment=c("4112576", "4113357", "4147521", "4147563"), alias=as.factor(c(
  'Learning rate: 1E-3', 'Learning rate: 1E-3 + OneCycleLR', 'Learning rate: 1E-4 + Weight Decay', 'Learning rate: 1E-4 + OneCycleLR'
)))
data_3xr6_adam %>% merge(relationship, by="experiment") %>%
  ggplot() +
  geom_line(aes(x = step, y = loss, color=alias)) +
  theme_bw() +
  labs(x = "Batch", y = "Loss", color="Experiment") +
  theme(text=element_text(size=20)) + 
  scale_color_brewer(palette="Set1")

# RMSprop
# Alias Experiment LR       WD       Momemtum Optimiser Scheduler
# 5     4113024    0.001    0        0        RMSprop   None
# 6     4134997    0.001    0        0        RMSprop   OneCycleLR
# 7     4147392    0.0001   0.01     0.9      RMSprop   None
# 8     4147415    0.0001   0        0.9      RMSprop   None
result <- fromJSON(file = "experiment_2/3xr6/RMSprop/4113024/loss_3xr6_rmsprop.json")
data <- data.frame(step=result[[1]]$x, loss=result[[1]]$y, optimiser='RMSprop',  scheduler='None')
data$experiment <- "4113024"
data$learning_rate <- 0.001
data_3xr6_rmsprop <- data
result <- fromJSON(file = "experiment_2/3xr6/RMSprop/4134997/loss_3xr6_rmsprop_scheduler.json")
data <- data.frame(step=result[[1]]$x, loss=result[[1]]$y, optimiser='RMSprop',  scheduler='OneCycleLR')
data$experiment <- "4134997"
data$learning_rate <- 0.001
data_3xr6_rmsprop <- rbind(data_3xr6_rmsprop, data)
result <- read.csv2(file = "experiment_2/3xr6/RMSprop/4147392/3xr6_RMSprop_storage_1.tsv", sep="\t")
result$step <- 0:(dim(result)[1]-1)
result$avgcer <- NULL
result$optimiser <- "RMSprop"
result$scheduler <- "None"
result$experiment <- "4147392"
result$learning_rate <- 0.0001
result$X <- NULL
data_3xr6_rmsprop <- rbind(data_3xr6_rmsprop, result)
result <- read.csv2(file = "experiment_2/3xr6/RMSprop/4147415/3xr6_RMSprop_storage_2.tsv", sep="\t")
result$step <- 0:(dim(result)[1]-1)
result$avgcer <- NULL
result$optimiser <- "RMSprop"
result$scheduler <- "None"
result$experiment <- "4147415"
result$learning_rate <- 0.0001
result$X <- NULL
data_3xr6_rmsprop <- rbind(data_3xr6_rmsprop, result)
# Plot
data_3xr6_rmsprop$loss <- as.numeric(data_3xr6_rmsprop$loss)
relationship <- data.frame(experiment=c("4113024", "4134997", "4147392", "4147415"), alias=as.factor(c(
  'Learning rate: 1E-3', 'Learning rate: 1E-3 + OneCycleLR', 'Learning rate: 1E-4 + Weight Decay + Momentum', 'Learning rate: 1E-4 + Momentum'
)))
data_3xr6_rmsprop %>% merge(relationship, by="experiment") %>%
  ggplot() +
  geom_line(aes(x = step, y = loss, color=alias)) +
  theme_bw() +
  labs(x = "Batch", y = "Loss", color="Experiment") +
  theme_bw() +
  labs(x = "Batch", y = "Loss", color="Experiment") +
  theme(text=element_text(size=20)) + 
  scale_color_brewer(palette="Set1")

# SGD
# Alias Experiment LR       WD       Momentum Optimiser Scheduler
# 9     4113028    0.001    0        0        SGD       None
# 10    4135668    0.001    0        0        SGD       OneCycleLR
# 11    4147242    0.0001   0        0.9      SGD       None
result <- fromJSON(file = "experiment_2/3xr6/SGD/4113028/loss_3xr6_sgd.json")
data <- data.frame(step=result[[1]]$x, loss=result[[1]]$y, optimiser='SGD',  scheduler='None')
data$experiment <- "4113028"
data$learning_rate <- 0.001
data_3xr6_sgd <- data
result <- fromJSON(file = "experiment_2/3xr6/SGD/4135668/loss_3xr6_sgd_scheduler.json")
data <- data.frame(step=result[[1]]$x, loss=result[[1]]$y, optimiser='SGD',  scheduler='OneCycleLR')
data$experiment <- "4135668"
data$learning_rate <- 0.001
data_3xr6_sgd <- rbind(data_3xr6_sgd, data)
result <- read.csv2(file = "experiment_2/3xr6/SGD/4147242/3xr6_SGD_storage_1.tsv", sep="\t")
result$step <- 0:(dim(result)[1]-1)
result$avgcer <- NULL
result$optimiser <- "SGD"
result$scheduler <- "None"
result$experiment <- "4147242"
result$learning_rate <- 0.0001
result$X <- NULL
data_3xr6_sgd <- rbind(data_3xr6_sgd, result)
# Plot
data_3xr6_sgd$loss <- as.numeric(data_3xr6_sgd$loss)
relationship <- data.frame(experiment=c("4113028", "4135668", "4147242"), alias=as.factor(c(
  'Learning rate: 1E-3', 'Learning rate: 1E-3 + OneCycleLR', 'Learning rate: 1E-4 + Momentum'
)))
data_3xr6_sgd %>% merge(relationship, by="experiment") %>%
  ggplot() +
  geom_line(aes(x = step, y = loss, color=alias)) +
  theme_bw() +
  labs(x = "Batch", y = "Loss", color="Experiment") +
  theme(text=element_text(size=20))

# # Avg CER
# result <- fromJSON(file = "ap/4107318/avgcer_ap_adam.json")
# data <- data.frame(step=result[[1]]$x, avgcer=result[[1]]$y, optimiser='Adam',  scheduler='None')
# data_ap <- data
# result <- fromJSON(file = "ap/4108437/avgcer_ap_rmsprop.json")
# data <- data.frame(step=result[[1]]$x, avgcer=result[[1]]$y, optimiser='RMSprop',  scheduler='None')
# data_ap <- rbind(data_ap, data)
# result <- fromJSON(file = "ap/4109506/avgcer_ap_sgd.json")
# data <- data.frame(step=result[[1]]$x, avgcer=result[[1]]$y, optimiser='SGD',  scheduler='None')
# data_ap <- rbind(data_ap, data)
# result <- fromJSON(file = "ap/4112089/avgcer_ap_adam_scheduler.json")
# data <- data.frame(step=result[[1]]$x, avgcer=result[[1]]$y, optimiser='Adam',  scheduler='OneCycleLR')
# data_ap <- rbind(data_ap, data)
# result <- fromJSON(file = "ap/4112608/avgcer_ap_rmsprop_scheduler.json")
# data <- data.frame(step=result[[1]]$x, avgcer=result[[1]]$y, optimiser='RMSprop',  scheduler='OneCycleLR')
# data_ap <- rbind(data_ap, data)
# result <- fromJSON(file = "ap/4112614/avgcer_ap_sgd_scheduler.json")
# data <- data.frame(step=result[[1]]$x, avgcer=result[[1]]$y, optimiser='SGD',  scheduler='OneCycleLR')
# data_ap <- rbind(data_ap, data)
# data_ap$group <- str_replace(paste(data_ap$optimiser, data_ap$scheduler, sep=' & '), ' & None', '')
# data_ap_avgcer <- data_ap
# 
# # 3xr6
# result <- fromJSON(file = "3xr6/4112576/loss_3xr6_adam.json")
# data <- data.frame(step=result[[1]]$x, loss=result[[1]]$y, optimiser='Adam',  scheduler='None')
# data_3xr6 <- data
# result <- fromJSON(file = "3xr6/4113024/loss_3xr6_rmsprop.json")
# data <- data.frame(step=result[[1]]$x, loss=result[[1]]$y, optimiser='RMSprop',  scheduler='None')
# data_3xr6 <- rbind(data_3xr6, data)
# result <- fromJSON(file = "3xr6/4113028/loss_3xr6_sgd.json")
# data <- data.frame(step=result[[1]]$x, loss=result[[1]]$y, optimiser='SGD',  scheduler='None')
# data_3xr6 <- rbind(data_3xr6, data)
# 
# 
# # Plot the data
# # Loss
# data_ap_loss %>%
#   ggplot +
#   geom_line(mapping = aes(x = step, y = loss, color = group), size=1.15) +
#   labs(x = "Batch", y = "Loss", color = "Optimiser") +
#   scale_color_jcolors(palette = "pal8") + 
#   theme_bw() +
#   theme(text = element_text(size=20))
# 
# data_ap_avgcer %>%
#   ggplot +
#   geom_line(mapping = aes(x = step, y = avgcer, color = group), size=1.15) +
#   labs(x = "Batch", y = "Average CER", color = "Optimiser") +
#   scale_color_jcolors(palette = "pal8") + 
#   theme_bw() +
#   theme(text = element_text(size=20)) 
# 
# data_3xr6[data_3xr6$scheduler == 'None', ] %>%
#   ggplot +
#   geom_line(mapping = aes(x = step, y = loss, color = optimiser)) +
#   labs(x = "Batch", y = "Loss", color = "Optimiser") +
#   scale_color_brewer(palette="Set1") + 
#   theme_bw()
# 
# data_ap[data_ap$scheduler == 'OneCycleLR', ]$loss[1:5]
# data_ap[data_ap$scheduler == 'None', ]$loss[1:5]
# 
