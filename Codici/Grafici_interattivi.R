setwd("C:\\Users\\Beatrice\\Documents\\CLAMSES\\Statistical Learning\\Project")


dati<-read.csv("Datini3.csv", sep=";",header = T, dec=",")
head(dati)
dati=dati[,-16]

AUC_best<-c()
for (i in 1: nrow(dati)){
  AUC_best[i]<-max(dati[i,9:14])
}

dati$AUC_best<-AUC_best
set.seed(417)
library(plotly)
colors <- c('#4AC6B7', '#1972A4', '#965F8A', '#FF7070', '#C61951',"#E8A3A2")

p <- plot_ly(dati, x = ~n, y = ~p, color = ~Class, size = ~AUC_best, colors = colors,
             type = 'scatter', mode = 'markers',
             marker = list(symbol = 'circle', sizemode = 'diameter',
                           line = list(width = 2, color = '#FFFFFF')),
             text = ~paste('Method:', Class, '<br>AUC:', AUC_best)) %>%
  layout(title = 'AUC per metodi',
         plot_bgcolor = "white",
         paper_bgcolor="white",
         legend = list(orientation = 'h'),
         xaxis = list(title = 'n',
                      gridcolor = 'rgb(255, 255, 255)',
                      range = c(1.2334, 7),
                      type = 'log',
                      zerolinewidth = 1,
                      ticklen = 5,
                      gridwidth = 2),
         yaxis = list(title = 'Life Expectancy (years)',
                      gridcolor = 'rgb(255, 255, 255)',
                      range = c(-100.911111, 421.72921793264332),
                      zerolinewidth = 1,
                      ticklen = 5,
                      gridwith = 2),
         paper_bgcolor = 'rgb(243, 243, 243)',
         plot_bgcolor = 'rgb(243, 243, 243)')
p


library(cluster)
#install.packages("fpc")
library(fpc)
set.seed(123)
km <- kmeans(dati[,c(2:4,6,8)],5)
clusterini <- km$cluster
dati_c <- cbind(dati, clusterini)


g1 <- dati_c[which(clusterini==1),]
summary(g1)
#129, 1831
g2 <- dati_c[which(clusterini==2),]
summary(g2)
#286048, 567479 
g3 <- dati_c[which(clusterini==3),]
summary(g3)
#3062, 5803  
g4 <- dati_c[which(clusterini==4),]
summary(g4)
#49097, 95156
g5 <- dati_c[which(clusterini==5),]
summary(g5)
#6435 11183

AUC_best<-c()
for (i in 1: nrow(dati)){
  AUC_best[i]<-max(dati[i,9:14])
}
dati$AUC_best<-AUC_best
set.seed(417)
library(plotly)
colors <- c('#4AC6B7', '#1972A4', '#965F8A', '#FF7070', '#C61951',"#E8A3A2")

#Primo cluster----
plot_ly(g1, x = ~n, y = ~p, color = ~Class, size = ~AUC_best, colors = colors,
             type = 'scatter', mode = 'markers',
             marker = list(symbol = 'circle', sizemode = 'diameter',
                           line = list(width = 2, color = '#FFFFFF')),
             text = ~paste('Method:', Class, '<br>AUC:', AUC_best,'<br>Normalized:',Scaled,
                           '<br>Dataset:',Dataset)) %>%
  layout(showlegend = FALSE,
         plot_bgcolor = "white",
         paper_bgcolor="white",
      title = 'AUC per metodi',
         xaxis = list(title = 'n',
                      gridcolor = 'gray50',
                      range = c(0.0001335, 1900.987374),
                      zerolinewidth = 1,
                      ticklen = 5,
                      gridwidth = 2),
         yaxis = list(title = 'p',
                      gridcolor = 'gray50',
                      range = c(-100.911111, 421.72921793264332),
                      zerolinewidth = 1,
                      ticklen = 5,
                      gridwith = 2))

#Secondo cluster-------
AUC_best<-c()
for (i in 1: nrow(dati)){
  AUC_best[i]<-max(dati[i,9:14])
}
dati$AUC_best<-AUC_best
set.seed(417)
library(plotly)
colors <- c('#4AC6B7', '#1972A4', '#965F8A', '#FF7070', '#C61951',"#E8A3A2")

plot_ly(g2, x = ~n, y = ~p, color = ~Class, size = ~AUC_best, colors = colors,
        type = 'scatter', mode = 'markers',
        marker = list(symbol = 'circle', sizemode = 'diameter',
                      line = list(width = 2, color = '#FFFFFF')),
        text = ~paste('Method:', Class, '<br>AUC:', AUC_best,'<br>Normalized:',Scaled,
                      '<br>Dataset:',Dataset)) %>%
  layout(showlegend = FALSE,
         plot_bgcolor = "white",
         paper_bgcolor="white",
         title = 'AUC per metodi',
         xaxis = list(title = 'n',
                      gridcolor = 'gray50',
                      range = c(200000.7293, 600000.987374),
                      zerolinewidth = 1,
                      ticklen = 5,
                      gridwidth = 2),
         yaxis = list(title = 'p',
                      gridcolor = 'gray50',
                      range = c(-100.911111, 421.72921793264332),
                      zerolinewidth = 1,
                      ticklen = 5,
                      gridwith = 2))

#Terzo cluster----
plot_ly(g3, x = ~n, y = ~p, color = ~Class, size = ~AUC_best, colors = colors,
        type = 'scatter', mode = 'markers',
        marker = list(symbol = 'circle', sizemode = 'diameter',
                      line = list(width = 2, color = '#FFFFFF')),
        text = ~paste('Method:', Class, '<br>AUC:', AUC_best,'<br>Normalized:',Scaled,
                      '<br>Dataset:',Dataset)) %>%
  layout(showlegend = FALSE,
         plot_bgcolor = "white",
         paper_bgcolor="white",
         title = 'AUC per metodi',
         xaxis = list(title = 'n',
                      gridcolor = 'gray50',
                      range = c(2000.7293, 6000.987374),
                      zerolinewidth = 1,
                      ticklen = 5,
                      gridwidth = 2),
         yaxis = list(title = 'p',
                      gridcolor = 'gray50',
                      range = c(-100.911111, 421.72921793264332),
                      zerolinewidth = 1,
                      ticklen = 5,
                      gridwith = 2))

#Quarto cluster-------
plot_ly(g4, x = ~n, y = ~p, color = ~Class, size = ~AUC_best, colors = colors,
        type = 'scatter', mode = 'markers',
        marker = list(symbol = 'circle', sizemode = 'diameter',
                      line = list(width = 2, color = '#FFFFFF')),
        text = ~paste('Method:', Class, '<br>AUC:', AUC_best,'<br>Normalized:',Scaled,
                      '<br>Dataset:',Dataset)) %>%
  layout(showlegend = FALSE,
         plot_bgcolor = "white",
         paper_bgcolor="white",
         title = 'AUC per metodi',
         xaxis = list(title = 'n',
                      gridcolor = 'gray50',
                      range = c(40097,100000),
                      zerolinewidth = 1,
                      ticklen = 5,
                      gridwidth = 2),
         yaxis = list(title = 'p',
                      gridcolor = 'gray50',
                      range = c(-100.911111, 421.72921793264332),
                      zerolinewidth = 1,
                      ticklen = 5,
                      gridwith = 2))

#Quinto cluster-------
plot_ly(g5, x = ~n, y = ~p, color = ~Class, size = ~AUC_best, colors = colors,
        type = 'scatter', mode = 'markers',
        marker = list(symbol = 'circle', sizemode = 'diameter',
                      line = list(width = 2, color = '#FFFFFF')),
        text = ~paste('Method:', Class, '<br>AUC:', AUC_best,'<br>Normalized:',Scaled,
                      '<br>Dataset:',Dataset)) %>%
  layout(showlegend = FALSE,
         plot_bgcolor = "white",
         paper_bgcolor="white",
         title = 'AUC per metodi',
         xaxis = list(title = 'n',
                      gridcolor = 'gray50',
                      range = c(5000 ,12000),
                      zerolinewidth = 1,
                      ticklen = 5,
                      gridwidth = 2),
         yaxis = list(title = 'p',
                      gridcolor = 'gray50',
                      range = c(-100.911111, 421.72921793264332),
                      zerolinewidth = 1,
                      ticklen = 5,
                      gridwith = 2))


#-----------------------------------------------------------------------------------------
#Scatterplot 3D per ogni metodo
#-----------------------------------------------------------------------------------------

set.seed(417)
library(plotly)
plot_ly(x=dati$n, y=dati$p, z=dati$IF, type="scatter3d", mode="markers", color=dati$IF)
library("colorspace")
pal <- choose_palette()

#DBSCAN------
plot_ly(dati, x = ~n, y = ~p, z = ~DBSCAN,mode="markers", color=dati$DBSCAN,
        text = ~paste('Dataset:',Dataset)) %>%
  add_markers() %>%
  layout(title = 'AUC DBSCAN',
         scene = list(xaxis = list(title = 'n'),
                      yaxis = list(title = 'p'),
                      zaxis = list(title = 'AUC')),
         annotations = list(
           x = 1.13,
           y = 1.05,
           text = 'AUC',
           xref = 'paper',
           yref = 'paper',
           showarrow = FALSE
         ))
dati$Scaled<-as.factor(dati$Scaled)
plot_ly(dati, x = ~n, y = ~p, z = ~DBSCAN,mode="markers", color = ~Scaled, colors = c('#BF382A', '#0C4B8E'),
        text = ~paste('Dataset:',Dataset)) %>%
  add_markers() %>%
  layout(title = 'AUC DBSCAN',
         scene = list(xaxis = list(title = 'n'),
                      yaxis = list(title = 'p'),
                      zaxis = list(title = 'AUC')),
         annotations = list(
           x = 1.13,
           y = 1.05,
           text = 'Dati normalizzati',
           xref = 'paper',
           yref = 'paper',
           showarrow = FALSE
         ))


#Optics------
plot_ly(dati, x = ~n, y = ~p, z = ~Optics,mode="markers", color=dati$Optics,
        text = ~paste('Dataset:',Dataset)) %>%
  add_markers() %>%
  layout(title = 'AUC Optics',
         scene = list(xaxis = list(title = 'n'),
                      yaxis = list(title = 'p'),
                      zaxis = list(title = 'AUC')),
         annotations = list(
           x = 1.13,
           y = 1.05,
           text = 'AUC',
           xref = 'paper',
           yref = 'paper',
           showarrow = FALSE
         ))

plot_ly(dati, x = ~n, y = ~p, z = ~Optics,mode="markers", color = ~Scaled, colors = c('#BF382A', '#0C4B8E'),
        text = ~paste('Dataset:',Dataset)) %>%
  add_markers() %>%
  layout(title = 'AUC Optics',
         scene = list(xaxis = list(title = 'n'),
                      yaxis = list(title = 'p'),
                      zaxis = list(title = 'AUC')),
         annotations = list(
           x = 1.13,
           y = 1.05,
           text = 'Dati normalizzati',
           xref = 'paper',
           yref = 'paper',
           showarrow = FALSE
         ))

#Lof------
plot_ly(dati, x = ~n, y = ~p, z = ~Lof,mode="markers", color=dati$Lof,
        text = ~paste('Dataset:',Dataset)) %>%
  add_markers() %>%
  layout(title = 'AUC Lof',
         scene = list(xaxis = list(title = 'n'),
                      yaxis = list(title = 'p'),
                      zaxis = list(title = 'AUC')),
         annotations = list(
           x = 1.13,
           y = 1.05,
           text = 'AUC',
           xref = 'paper',
           yref = 'paper',
           showarrow = FALSE
         ))
plot_ly(dati, x = ~n, y = ~p, z = ~Lof,mode="markers", color = ~Scaled, colors = c('#BF382A', '#0C4B8E'),
        text = ~paste('Dataset:',Dataset)) %>%
  add_markers() %>%
  layout(title = 'AUC Lof',
         scene = list(xaxis = list(title = 'n'),
                      yaxis = list(title = 'p'),
                      zaxis = list(title = 'AUC')),
         annotations = list(
           x = 1.13,
           y = 1.05,
           text = 'Dati normalizzati',
           xref = 'paper',
           yref = 'paper',
           showarrow = FALSE
         ))

#kNN------
plot_ly(dati, x = ~n, y = ~p, z = ~kNN,mode="markers", color=dati$kNN,
        text = ~paste('Dataset:',Dataset)) %>%
  add_markers() %>%
  layout(title = 'AUC kNN',
         scene = list(xaxis = list(title = 'n'),
                      yaxis = list(title = 'p'),
                      zaxis = list(title = 'AUC')),
         annotations = list(
           x = 1.13,
           y = 1.05,
           text = 'AUC',
           xref = 'paper',
           yref = 'paper',
           showarrow = FALSE
         ))
plot_ly(dati, x = ~n, y = ~p, z = ~kNN,mode="markers", color = ~Scaled, colors = c('#BF382A', '#0C4B8E'),
        text = ~paste('Dataset:',Dataset)) %>%
  add_markers() %>%
  layout(title = 'AUC kNN',
         scene = list(xaxis = list(title = 'n'),
                      yaxis = list(title = 'p'),
                      zaxis = list(title = 'AUC')),
         annotations = list(
           x = 1.13,
           y = 1.05,
           text = 'Dati normalizzati',
           xref = 'paper',
           yref = 'paper',
           showarrow = FALSE
         ))
#SVM------
plot_ly(dati, x = ~n, y = ~p, z = ~SVM,mode="markers", color=dati$SVM,
        text = ~paste('Dataset:',Dataset)) %>%
  add_markers() %>%
  layout(title = 'AUC SVM',
         scene = list(xaxis = list(title = 'n'),
                      yaxis = list(title = 'p'),
                      zaxis = list(title = 'AUC')),
         annotations = list(
           x = 1.13,
           y = 1.05,
           text = 'AUC',
           xref = 'paper',
           yref = 'paper',
           showarrow = FALSE
         ))
plot_ly(dati, x = ~n, y = ~p, z = ~SVM,mode="markers", color = ~Scaled, colors = c('#BF382A', '#0C4B8E'),
        text = ~paste('Dataset:',Dataset)) %>%
  add_markers() %>%
  layout(title = 'AUC SVM',
         scene = list(xaxis = list(title = 'n'),
                      yaxis = list(title = 'p'),
                      zaxis = list(title = 'AUC')),
         annotations = list(
           x = 1.13,
           y = 1.05,
           text = 'Dati normalizzati',
           xref = 'paper',
           yref = 'paper',
           showarrow = FALSE
         ))


#Isolation Forest------
plot_ly(dati, x = ~n, y = ~p, z = ~IF,mode="markers", color=dati$IF,
              text = ~paste('Dataset:',Dataset)) %>%
  add_markers() %>%
  layout(title = 'AUC Isolation Forest',
         scene = list(xaxis = list(title = 'n'),
                      yaxis = list(title = 'p'),
                      zaxis = list(title = 'AUC')),
         annotations = list(
           x = 1.13,
           y = 1.05,
           text = 'AUC',
           xref = 'paper',
           yref = 'paper',
           showarrow = FALSE
         ))
dati$Scaled<-as.factor(dati$Scaled)
plot_ly(dati, x = ~n, y = ~p, z = ~IF,mode="markers", color = ~Scaled, colors = c('#BF382A', '#0C4B8E'),
        text = ~paste('Dataset:',Dataset)) %>%
  add_markers() %>%
  layout(title = 'AUC Isolation Forest',
         scene = list(xaxis = list(title = 'n'),
                      yaxis = list(title = 'p'),
                      zaxis = list(title = 'AUC')),
         annotations = list(
           x = 1.13,
           y = 1.05,
           text = 'Dati normalizzati',
           xref = 'paper',
           yref = 'paper',
           showarrow = FALSE
         ))
