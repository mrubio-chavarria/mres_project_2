
# Libraries
library(tidyverse)

# Read data
base.route <- '/home/mario/Documentos/Imperial/Project_2/output_experiments/experiment1'

# AP
dataset <- 'ap/ap'
# LSTM (gamma 0)
gamma <- 1  # The index is one value up
data.route <- paste(base.route, '/', dataset, '_exp1_gamma_', gamma, '.tsv', sep='')
data.1 <- read.csv2(data.route, header = F, sep='\t')
data.1 <- data.1[, 2:4]
colnames(data.1) <- c('loss', 'avgcer', 'training')
data.1$training <- as.logical(data.1$training)
data.1$gamma <- 'LSTM'
data.1$step <- 0
data.1[data.1$training, ]$step <- 0:(dim(data.1[data.1$training, ])[1]-1)
data.1[!data.1$training, ]$step <- 0:(dim(data.1[!data.1$training, ])[1]-1)
data.1$loss <- as.numeric(data.1$loss)
data.1$avgcer <- as.numeric(data.1$avgcer)
# BNLSTM (gamma 1)
gamma <- 2  # The index is one value up
data.route <- paste(base.route, '/', dataset, '_exp1_gamma_', gamma, '.tsv', sep='')
data.2 <- read.csv2(data.route, header = F, sep='\t')
data.2 <- data.2[, 2:4]
colnames(data.2) <- c('loss', 'avgcer', 'training')
data.2$training <- as.logical(data.2$training)
data.2$gamma <- '0.1'
data.2$step <- 0
data.2[data.2$training, ]$step <- 0:(dim(data.2[data.2$training, ])[1]-1)
data.2[!data.2$training, ]$step <- 0:(dim(data.2[!data.2$training, ])[1]-1)
data.2$loss <- as.numeric(data.2$loss)
data.2$avgcer <- as.numeric(data.2$avgcer)
# BNLSTM (gamma 2)
gamma <- 3  # The index is one value up
data.route <- paste(base.route, '/', dataset, '_exp1_gamma_', gamma, '.tsv', sep='')
data.3 <- read.csv2(data.route, header = F, sep='\t')
data.3 <- data.3[, 2:4]
colnames(data.3) <- c('loss', 'avgcer', 'training')
data.3$training <- as.logical(data.3$training)
data.3$gamma <- '0.2'
data.3$step <- 0
data.3[data.3$training, ]$step <- 0:(dim(data.3[data.3$training, ])[1]-1)
data.3[!data.3$training, ]$step <- 0:(dim(data.3[!data.3$training, ])[1]-1)
data.3$loss <- as.numeric(data.3$loss)
data.3$avgcer <- as.numeric(data.3$avgcer)
# BNLSTM (gamma 3)
gamma <- 4  # The index is one value up
data.route <- paste(base.route, '/', dataset, '_exp1_gamma_', gamma, '.tsv', sep='')
data.4 <- read.csv2(data.route, header = F, sep='\t')
data.4 <- data.4[, 2:4]
colnames(data.4) <- c('loss', 'avgcer', 'training')
data.4$training <- as.logical(data.4$training)
data.4$gamma <- '0.3'
data.4$step <- 0
data.4[data.4$training, ]$step <- 0:(dim(data.4[data.4$training, ])[1]-1)
data.4[!data.4$training, ]$step <- 0:(dim(data.4[!data.4$training, ])[1]-1)
data.4$loss <- as.numeric(data.4$loss)
data.4$avgcer <- as.numeric(data.4$avgcer)
# BNLSTM (gamma 5)
gamma <- 6  # The index is one value up
data.route <- paste(base.route, '/', dataset, '_exp1_gamma_', gamma, '.tsv', sep='')
data.6 <- read.csv2(data.route, header = F, sep='\t')
data.6 <- data.6[, 2:4]
colnames(data.6) <- c('loss', 'avgcer', 'training')
data.6$training <- as.logical(data.6$training)
data.6$gamma <- '0.5'
data.6$step <- 0
data.6[data.6$training, ]$step <- 0:(dim(data.6[data.6$training, ])[1]-1)
data.6[!data.6$training, ]$step <- 0:(dim(data.6[!data.6$training, ])[1]-1)
data.6$loss <- as.numeric(data.6$loss)
data.6$avgcer <- as.numeric(data.6$avgcer)
# BNLSTM (gamma 6)
gamma <- 7  # The index is one value up
data.route <- paste(base.route, '/', dataset, '_exp1_gamma_', gamma, '.tsv', sep='')
data.7 <- read.csv2(data.route, header = F, sep='\t')
data.7 <- data.7[, 2:4]
colnames(data.7) <- c('loss', 'avgcer', 'training')
data.7$training <- as.logical(data.7$training)
data.7$gamma <- '0.6'
data.7$step <- 0
data.7[data.7$training, ]$step <- 0:(dim(data.7[data.7$training, ])[1]-1)
data.7[!data.7$training, ]$step <- 0:(dim(data.7[!data.7$training, ])[1]-1)
data.7$loss <- as.numeric(data.7$loss)
data.7$avgcer <- as.numeric(data.7$avgcer)
# BNLSTM (gamma 7)
gamma <- 8  # The index is one value up
data.route <- paste(base.route, '/', dataset, '_exp1_gamma_', gamma, '.tsv', sep='')
data.8 <- read.csv2(data.route, header = F, sep='\t')
data.8 <- data.8[, 2:4]
colnames(data.8) <- c('loss', 'avgcer', 'training')
data.8$training <- as.logical(data.8$training)
data.8$gamma <- '0.7'
data.8$step <- 0
data.8[data.8$training, ]$step <- 0:(dim(data.8[data.8$training, ])[1]-1)
data.8[!data.8$training, ]$step <- 0:(dim(data.8[!data.8$training, ])[1]-1)
data.8$loss <- as.numeric(data.8$loss)
data.8$avgcer <- as.numeric(data.8$avgcer)
# BNLSTM (gamma 8)
gamma <- 9 # The index is one value up
data.route <- paste(base.route, '/', dataset, '_exp1_gamma_', gamma, '.tsv', sep='')
data.9 <- read.csv2(data.route, header = F, sep='\t')
data.9 <- data.9[, 2:4]
colnames(data.9) <- c('loss', 'avgcer', 'training')
data.9$training <- as.logical(data.9$training)
data.9$gamma <- '0.8'
data.9$step <- 0
data.9[data.9$training, ]$step <- 0:(dim(data.9[data.9$training, ])[1]-1)
data.9[!data.9$training, ]$step <- 0:(dim(data.9[!data.9$training, ])[1]-1)
data.9$loss <- as.numeric(data.9$loss)
data.9$avgcer <- as.numeric(data.9$avgcer)
# BNLSTM (gamma 9)
gamma <- 10 # The index is one value up
data.route <- paste(base.route, '/', dataset, '_exp1_gamma_', gamma, '.tsv', sep='')
data.10 <- read.csv2(data.route, header = F, sep='\t')
data.10 <- data.10[, 2:4]
colnames(data.10) <- c('loss', 'avgcer', 'training')
data.10$training <- as.logical(data.10$training)
data.10$gamma <- '0.9'
data.10$step <- 0
data.10[data.10$training, ]$step <- 0:(dim(data.10[data.10$training, ])[1]-1)
data.10[!data.10$training, ]$step <- 0:(dim(data.10[!data.10$training, ])[1]-1)
data.10$loss <- as.numeric(data.10$loss)
data.10$avgcer <- as.numeric(data.10$avgcer)

# Combine the data
data.ap <- rbind(data.1, data.2, data.3, data.4, data.6, data.7, data.8, data.9, data.10)
data.ap$gamma <- factor(data.ap$gamma, levels=c('LSTM', '0.1', '0.2', '0.3', '0.5', '0.6', '0.7', '0.8', '0.9'))

# PLOTS
text_size=30
line_size=2
# AP
# Loss against training
data.ap[data.ap$training, ] %>%
  ggplot() +
  geom_line(aes(x = step, y = log(loss, 10), color = gamma)) +
  theme_bw() +
  theme(text = element_text(size=text_size)) +
  labs(x = 'Batch', y = bquote(log[10]*'(CTC Loss)'), color = 'Gamma') +
  scale_fill_gradientn(colours=hcl.colors(10, palette='YlGnBu'))
# Error against training  
data.ap[data.ap$training, ] %>%
  ggplot() +
  geom_line(aes(x = step, y = avgcer, color = gamma)) +
  theme_bw() +
  theme(text = element_text(size=text_size)) +
  labs(x = 'Batch', y = 'Average CER', color='Gamma') +
  scale_fill_gradientn(colours=hcl.colors(10, palette='YlGnBu'))


# 3xr6
dataset <- '3xr6/3xr6'
# LSTM (gamma 0)
gamma <- 1  # The index is one value up
data.route <- paste(base.route, '/', dataset, '_exp1_gamma_', gamma, '.tsv', sep='')
data.1 <- read.csv2(data.route, header = F, sep='\t')
data.1 <- data.1[, 2:4]
colnames(data.1) <- c('loss', 'avgcer', 'training')
data.1$training <- as.logical(data.1$training)
data.1$gamma <- 'LSTM'
data.1$step <- 0
data.1[data.1$training, ]$step <- 0:(dim(data.1[data.1$training, ])[1]-1)
data.1[!data.1$training, ]$step <- 0:(dim(data.1[!data.1$training, ])[1]-1)
data.1$loss <- as.numeric(data.1$loss)
data.1$avgcer <- as.numeric(data.1$avgcer)
# BNLSTM (gamma 1)
gamma <- 2  # The index is one value up
data.route <- paste(base.route, '/', dataset, '_exp1_gamma_', gamma, '.tsv', sep='')
data.2 <- read.csv2(data.route, header = F, sep='\t')
data.2 <- data.2[, 2:4]
colnames(data.2) <- c('loss', 'avgcer', 'training')
data.2$training <- as.logical(data.2$training)
data.2$gamma <- '0.1'
data.2$step <- 0
data.2[data.2$training, ]$step <- 0:(dim(data.2[data.2$training, ])[1]-1)
data.2[!data.2$training, ]$step <- 0:(dim(data.2[!data.2$training, ])[1]-1)
data.2$loss <- as.numeric(data.2$loss)
data.2$avgcer <- as.numeric(data.2$avgcer)
# BNLSTM (gamma 2)
gamma <- 3  # The index is one value up
data.route <- paste(base.route, '/', dataset, '_exp1_gamma_', gamma, '.tsv', sep='')
data.3 <- read.csv2(data.route, header = F, sep='\t')
data.3 <- data.3[, 2:4]
colnames(data.3) <- c('loss', 'avgcer', 'training')
data.3$training <- as.logical(data.3$training)
data.3$gamma <- '0.2'
data.3$step <- 0
data.3[data.3$training, ]$step <- 0:(dim(data.3[data.3$training, ])[1]-1)
data.3[!data.3$training, ]$step <- 0:(dim(data.3[!data.3$training, ])[1]-1)
data.3$loss <- as.numeric(data.3$loss)
data.3$avgcer <- as.numeric(data.3$avgcer)
# BNLSTM (gamma 3)
gamma <- 4  # The index is one value up
data.route <- paste(base.route, '/', dataset, '_exp1_gamma_', gamma, '.tsv', sep='')
data.4 <- read.csv2(data.route, header = F, sep='\t')
data.4 <- data.4[, 2:4]
colnames(data.4) <- c('loss', 'avgcer', 'training')
data.4$training <- as.logical(data.4$training)
data.4$gamma <- '0.3'
data.4$step <- 0
data.4[data.4$training, ]$step <- 0:(dim(data.4[data.4$training, ])[1]-1)
data.4[!data.4$training, ]$step <- 0:(dim(data.4[!data.4$training, ])[1]-1)
data.4$loss <- as.numeric(data.4$loss)
data.4$avgcer <- as.numeric(data.4$avgcer)
# BNLSTM (gamma 5)
gamma <- 6  # The index is one value up
data.route <- paste(base.route, '/', dataset, '_exp1_gamma_', gamma, '.tsv', sep='')
data.6 <- read.csv2(data.route, header = F, sep='\t')
data.6 <- data.6[, 2:4]
colnames(data.6) <- c('loss', 'avgcer', 'training')
data.6$training <- as.logical(data.6$training)
data.6$gamma <- '0.5'
data.6$step <- 0
data.6[data.6$training, ]$step <- 0:(dim(data.6[data.6$training, ])[1]-1)
data.6[!data.6$training, ]$step <- 0:(dim(data.6[!data.6$training, ])[1]-1)
data.6$loss <- as.numeric(data.6$loss)
data.6$avgcer <- as.numeric(data.6$avgcer)
# BNLSTM (gamma 7)
gamma <- 8  # The index is one value up
data.route <- paste(base.route, '/', dataset, '_exp1_gamma_', gamma, '.tsv', sep='')
data.8 <- read.csv2(data.route, header = F, sep='\t')
data.8 <- data.8[, 2:4]
colnames(data.8) <- c('loss', 'avgcer', 'training')
data.8$training <- as.logical(data.8$training)
data.8$gamma <- '0.7'
data.8$step <- 0
data.8[data.8$training, ]$step <- 0:(dim(data.8[data.8$training, ])[1]-1)
data.8[!data.8$training, ]$step <- 0:(dim(data.8[!data.8$training, ])[1]-1)
data.8$loss <- as.numeric(data.8$loss)
data.8$avgcer <- as.numeric(data.8$avgcer)
# BNLSTM (gamma 8)
gamma <- 9  # The index is one value up
data.route <- paste(base.route, '/', dataset, '_exp1_gamma_', gamma, '.tsv', sep='')
data.9 <- read.csv2(data.route, header = F, sep='\t')
data.9 <- data.9[, 2:4]
colnames(data.9) <- c('loss', 'avgcer', 'training')
data.9$training <- as.logical(data.9$training)
data.9$gamma <- '0.8'
data.9$step <- 0
data.9[data.9$training, ]$step <- 0:(dim(data.9[data.9$training, ])[1]-1)
data.9[!data.9$training, ]$step <- 0:(dim(data.9[!data.9$training, ])[1]-1)
data.9$loss <- as.numeric(data.9$loss)
data.9$avgcer <- as.numeric(data.9$avgcer)
# BNLSTM (gamma 9)
gamma <- 10  # The index is one value up
data.route <- paste(base.route, '/', dataset, '_exp1_gamma_', gamma, '.tsv', sep='')
data.10 <- read.csv2(data.route, header = F, sep='\t')
data.10 <- data.10[, 2:4]
colnames(data.10) <- c('loss', 'avgcer', 'training')
data.10$training <- as.logical(data.10$training)
data.10$gamma <- '0.9'
data.10$step <- 0
data.10[data.10$training, ]$step <- 0:(dim(data.10[data.10$training, ])[1]-1)
data.10[!data.10$training, ]$step <- 0:(dim(data.10[!data.10$training, ])[1]-1)
data.10$loss <- as.numeric(data.10$loss)
data.10$avgcer <- as.numeric(data.10$avgcer)
# BNLSTM (gamma 10)
gamma <- 11  # The index is one value up
data.route <- paste(base.route, '/', dataset, '_exp1_gamma_', gamma, '.tsv', sep='')
data.11 <- read.csv2(data.route, header = F, sep='\t')
data.11 <- data.11[, 2:4]
colnames(data.11) <- c('loss', 'avgcer', 'training')
data.11$training <- as.logical(data.11$training)
data.11$gamma <- '1'
data.11$step <- 0
data.11[data.11$training, ]$step <- 0:(dim(data.11[data.11$training, ])[1]-1)
data.11[!data.11$training, ]$step <- 0:(dim(data.11[!data.11$training, ])[1]-1)
data.11$loss <- as.numeric(data.11$loss)
data.11$avgcer <- as.numeric(data.11$avgcer)

# Combine the data
data.3xr6 <- rbind(data.1, data.2, data.3, data.4, data.6, data.8, data.9, data.10, data.11)
data.3xr6$gamma <- factor(data.3xr6$gamma, levels=c('LSTM', '0.1', '0.2', '0.3', '0.5', '0.7', '0.8', '0.9', '1'))

# PLOTS
text_size=30
line_size=2
# AP
# Loss against training
data.3xr6[data.3xr6$training, ] %>%
  ggplot() +
  geom_line(aes(x = step, y = log(loss, 10), color = gamma)) +
  theme_bw() +
  theme(text = element_text(size=text_size)) +
  labs(x = 'Batch', y = 'CTC Loss', color='Gamma') +
  scale_fill_gradientn(colours=hcl.colors(10, palette='YlGnBu'))
# Error against training  
data.3xr6[data.3xr6$training, ] %>%
  ggplot() +
  geom_line(aes(x = step, y = avgcer, color = gamma)) +
  theme_bw() +
  theme(text = element_text(size=text_size)) +
  labs(x = 'Batch', y = 'Average CER', color='Gamma') +
  scale_fill_gradientn(colours=hcl.colors(10, palette='YlGnBu'))

