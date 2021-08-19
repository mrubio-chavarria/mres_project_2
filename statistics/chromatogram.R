
# Library
library(dplyr)
library(ggplot2)
library(pracma)

lim <- 500
# Adenina 1
bias <- 1
data.a1 <- data.frame(step = 0:lim)
data.a1$base <- "adenine"
data.a1$intensity <- 0
a.intensity <- dnorm(linspace(-0.5, 0.5, 100), sd=0.1) + bias
data.a1$intensity[0:100] <- a.intensity

# Timina 1
data.t1 <- data.frame(step = 0:lim)
data.t1$base <- "thymine"
data.t1$intensity <- 0
t.intensity <- dnorm(linspace(-0.5, 0.5, 100), sd=0.1)
data.t1$intensity[41:140] <- a.intensity

# Adenina 2
data.a2 <- data.frame(step = 0:lim)
data.a2$base <- "adenine"
data.a2$intensity <- 0
a.intensity <- dnorm(linspace(-0.5, 0.5, 100), sd=0.1)
data.a2$intensity[82:181] <- a.intensity

# Cytosine 1
data.c1 <- data.frame(step = 0:lim)
data.c1$base <- "cytosine"
data.c1$intensity <- 0
c.intensity <- dnorm(linspace(-0.5, 0.5, 100), sd=0.1)
data.c1$intensity[123:222] <- c.intensity

# Cytosine 2
data.c2 <- data.frame(step = 0:lim)
data.c2$base <- "guanine"
data.c2$intensity <- 0
c.intensity <- dnorm(linspace(-0.5, 0.5, 100), sd=0.1)
data.c2$intensity[164:263] <- c.intensity

# Guanine 1
data.g1 <- data.frame(step = 0:lim)
data.g1$base <- "guanine"
data.g1$intensity <- 0
g.intensity <- dnorm(linspace(-0.5, 0.5, 100), sd=0.1)
data.g1$intensity[205:304] <- g.intensity

# Guanine 2
data.g2 <- data.frame(step = 0:lim)
data.g2$base <- "guanine"
data.g2$intensity <- 0
g.intensity <- dnorm(linspace(-0.5, 0.5, 100), sd=0.1)
data.g2$intensity[246:345] <- g.intensity

# Adenina 3
data.a3 <- data.frame(step = 0:lim)
data.a3$base <- "adenine"
data.a3$intensity <- 0
a.intensity <- dnorm(linspace(-0.5, 0.5, 100), sd=0.1)
data.a3$intensity[287:386] <- a.intensity

# Timina 1
data.t2 <- data.frame(step = 0:lim)
data.t2$base <- "thymine"
data.t2$intensity <- 0
t.intensity <- dnorm(linspace(-0.5, 0.5, 100), sd=0.1)
data.t2$intensity[328:427] <- a.intensity

# Base zero
data.b <- data.frame(step = 0:lim)
data.b$base <- "none"
data.b$intensity <- 0

# Plot chromatogram
a.color <- "steelblue4"
t.color <- "gold2"
g.color <- "chartreuse3"
c.color <- "red3"
size <- 1.2
ggplot() +
  geom_line(data=data.a1, aes(x = step, y = intensity), color=a.color, size=size) +
  geom_line(data=data.t1, aes(x = step, y = intensity), color=t.color, size=size) +
  geom_line(data=data.a2, aes(x = step, y = intensity), color=a.color, size=size) +
  geom_line(data=data.c1, aes(x = step, y = intensity), color=c.color, size=size) +
  geom_line(data=data.c2, aes(x = step, y = intensity), color=c.color, size=size) +
  geom_line(data=data.g1, aes(x = step, y = intensity), color=g.color, size=size) +
  geom_line(data=data.g2, aes(x = step, y = intensity), color=g.color, size=size) +
  geom_line(data=data.a3, aes(x = step, y = intensity), color=a.color, size=size) +
  geom_line(data=data.t2, aes(x = step, y = intensity), color=t.color, size=size) +
  geom_line(data=data.b, aes(x = step, y = intensity), color="black", size=size) +
  theme_linedraw() +
  labs(y = "Intensity")
  # theme(axis.title.x=element_blank(),
  #       axis.text.x=element_blank(),
  #       axis.ticks.x=element_blank())

