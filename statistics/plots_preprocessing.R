
# Libraries
library(rhdf5)
library(tidyverse)
library(ggalt)


# Example Fast5 file
route <- "/home/mario/Projects/project_2/databases/working_ap/reads/flowcell3/single/Q20_read13.fast5"

h5ls(route)

signal <- h5read(route,"/Raw")$Reads$Read_153571$Signal

n_points = 50
n_start = 75
n_end = n_start + n_points
data <- data.frame(step=n_start:n_end, signal = signal[n_start:n_end])

spline_int <- as.data.frame(spline(data$step, data$signal))

# Figure without the dashed lines
data %>%
  ggplot(aes(x=step, y=signal)) +
  geom_line(data = spline_int, aes(x = x, y = y), size = 20, color='dodgerblue3') +
  theme_void() +
  theme(text = element_text(size=20))  + 
  labs(x = 'Samples', y = 'Raw signal')

# Figure with the dashed lines
spline_int$norm <- (spline_int$y - mean(spline_int$y)) / sd(spline_int$y) 
data %>%
  ggplot(aes(x=step, y=norm)) +
  geom_line(data = spline_int, aes(x = x, y = norm), size = 1, color='dodgerblue3') +
  theme_classic() +
  theme(text = element_text(size=20))  + 
  labs(x = 'Samples', y = 'Normalised signal') +
  geom_vline(xintercept=106, linetype="dashed",color = "black", size=1, alpha=0.4) +
  geom_vline(xintercept=113, linetype="dashed",color = "black", size=1, alpha=0.4) +
  geom_vline(xintercept=121.75, linetype="dashed",color = "black", size=1, alpha=0.4) +
  geom_vline(xintercept=127, linetype="dashed",color = "black", size=1, alpha=0.4) +
  geom_vline(xintercept=131.25, linetype="dashed",color = "black", size=1, alpha=0.4) +
  geom_vline(xintercept=140, linetype="dashed",color = "black", size=1, alpha=0.4) +
  geom_vline(xintercept=144, linetype="dashed",color = "black", size=1, alpha=0.4) +
  geom_vline(xintercept=147.25, linetype="dashed",color = "black", size=1, alpha=0.4) +
  geom_vline(xintercept=150, linetype="dashed",color = "black", size=1, alpha=0.4) +
  geom_vline(xintercept=154.25, linetype="dashed",color = "black", size=1, alpha=0.4)
  
