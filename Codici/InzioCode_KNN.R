#----------------------------------------------------------------------------------------------------
# LIBRARIES -----
#----------------------------------------------------------------------------------------------------
#Import packages e librerie utilizzate:
library(R.matlab)
library(dbscan)
library(e1071)
library(caret)
library(NLP)
library(tm)
library(DMwR)
library(Rlof)
#Per l'istallazione del pacchetto isofor:
#install.packages("devtools")
#library(devtools)
#devtools::install_github("Zelazny7/isofor")
library(isofor)
#Per leggere i dati .mat
#install.packages("rmatio")
library(rmatio)
#Per il calcolo dell'AUC:
#install.packages("pROC")
library(pROC)
#Per il KNN-kdtree:
#install.packages("RANN")
#install.packages("adamethods")
#install.packages("DDoutlier")
library(RANN)
library(adamethods)
library(DDoutlier)
library(kknn)
library(partykit)
library(rpart)
library(rpart.plot)
#--------------------------------------------------------------------------------------------------------
# INITIAL CODE & FUNCTION -----
#--------------------------------------------------------------------------------------------------------
#Import data:

rm(list=ls())
setwd("C:/Users/Beatrice/Documents/CLAMSES/Statistical Learning/Project")

path="dati.mat"
#inserire path
data=readMat(path)
X=data$X
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))

p=dim(X)[2]; n=dim(X)[1]

#imputare a factor le variabili che sono categoriche:

fac<-c()#indice della variabile)
for (i in 1:length(fac)){
 dati[,i]<-as.factor(dati[,i])
}
#One hot encoding:
dati<-model.matrix(~.-1,dati[,-(p+1)])

#Dati Scalati:
X_scale<-scale(X)


#Split Automatizzato:
#Di seguito vengono proposte due funzioni per l'automatizzazione dello split

#Utilizziamo l'rpart:
rpart_split_finder <- function(outlier.scores, n){
  x <- outlier.scores
  y <- 1:n
  library(rpart)
  library(rpart.plot)
  splits=c()
  for(i in 1:3){
    set.seed(123)
    tree <-rpart(y~x,
                 control=rpart.control(maxdepth=i, # at most 1 split
                                       cp=0, # any positive improvement will do
                                       minsplit=1,
                                       minbucket=1, # even leaves with 1 point are accepted
                                       xval=0))
    
    tagli <- rpart.rules(tree)
    if( i!=1){
      splits<-c(splits, tagli[,5], tagli[,7])
    }
    else{
      splits<-c(splits, tagli[,5])
    }
  }
  togliere<-c("")
  if (togliere %in% splits){splits<-splits[which(splits!=togliere)]}
  splits<-as.numeric(splits)
  return(unique(splits))
}

ctree_split_finder <- function(outlier.scores, n){
  library(partykit)
  
  x <- outlier.scores
  y <- 1:n
  taglio=c()
  for (i in 1:3){
    albero <- ctree(y~x, maxdepth=i)
    splittini <- partykit:::.list.rules.party(albero)
    splittini<-unlist(strsplit(splittini, " "))
    togliere<-c("x","<",">","<=",">=","&","")
    for (el in togliere){
      if(el %in% splittini){splittini<-splittini[which(splittini!=el)]}}
    taglio <- c(taglio,round(as.numeric(splittini), 5))
  }
  
  return(unique(taglio))
}

# DATASET -----------------------------------------------------------------------------------------------------------------------

# Lympho - 6 outliers --------------------------
data=readMat("lympho.mat")
X=as.data.frame(data$X)
Y=data$y
p=dim(X)[2]; n=dim(X)[1]
summary(X)
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))

fac=c(1:18)
for (i in 1:length(fac)){
  dati[,i]<-as.factor(dati[,i])}
#One hot encoding:
dati<-model.matrix(~.-1,dati[,-(p+1)])
X=dati[,1:(ncol(dati)-1)]
p=dim(X)[2]; n=dim(X)[1]

#(1)
#Usare un knn aggregato con la seguente scelta di parametri:

knn.agg<-KNN_AGG(X, k_min = min(log(n),p+1), k_max = max(log(n),p+1))
plot(knn.agg)

a <- rpart_split_finder(knn.agg, n)
plot(sort(knn.agg), col=is_out[order(knn.agg)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[3]
#2714
outliers_a<-ifelse(knn.agg>best_split,1,0)

b <-ctree_split_finder(knn.agg,n)  #No Splits

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall=0.6666667
#Area under the curve: 0.8157
#Precision: 0.4444444

#(2)
#si presenta un diferente algoritmo in grado di gestire grandi dataset.
#usare come K una media, oppure il massimo o il minimo:
round(mean(c(log(n),p+1)),0)
#23
round(max(c(log(n),p+1)),0)
#42
round(min(c(log(n),p+1)),0)
#5
#Nello specifico caso funziona molto meglio il max:

#Media
nearest<-nn2(X, query =X , k = 23, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[2]
#2.9
outliers_a<-ifelse(score>best_split,1,0)

b <- ctree_split_finder(score, n) #NoSplits!!

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall=0.6666667
#Area under the curve: 0.8052
#Precision: 0.3333333

#Max
nearest<-nn2(X, query =X , k = 42, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[2]
#3.1
outliers_a<-ifelse(score>best_split,1,0)

b <- ctree_split_finder(score, n) #NoSplits!!

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall=0.6666667
#Area under the curve: 0.8052
#Precision: 0.3333333

#Min
nearest<-nn2(X, query =X , k = 5, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
abline(h=2.3, col="blue")
#In questo caso non si è soddisfatti degli split proposti dagli alberi, per questo motivo,
#si sceglie un taglio manuale con uno score di 2.3
outliers_a<-ifelse(score>2.3,1,0)

b <- ctree_split_finder(score, n) #NoSplits!!

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall=0.6666667
#Area under the curve: 0.8298
#Precision: 0.8

#Best k=5
#Best AUC=0.8298
#Best split= 2.3

#Oss: i risultati sono molto simili.



# WBC - 21 outliers ----------------------------
data=readMat("WBC.mat")
X=as.data.frame(data$X)
p=dim(X)[2]; n=dim(X)[1]
summary(X)   #Non normalizziamo perchè tutte le variabili sono già in scala (0,1)
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))


#(1)
#Usare un knn aggregato con la seguente scelta di parametri:

knn.agg<-KNN_AGG(X, k_min = min(log(n),p+1), k_max = max(log(n),p+1))
plot(knn.agg)

a <- rpart_split_finder(knn.agg, n)
plot(sort(knn.agg))
plot(sort(knn.agg), col=is_out[order(knn.agg)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[6]
#421
outliers_a<-ifelse(knn.agg>best_split,1,0)

b <-ctree_split_finder(knn.agg,n) 
plot(sort(knn.agg), col=is_out[order(knn.agg)]+1)
abline(h=b)
best_split=sort(b)
#207
outliers_b<-ifelse(knn.agg>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.3333333
#Area under the curve: 0.6611
#Precision: 0.6363636

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall=1
#Area under the curve: 0.9146
#Precision: 0.2560976

#(2)
#si presenta un diferente algoritmo in grado di gestire grandi dataset.
#usare come K una media, oppure il massimo o il minimo:
round(mean(c(log(n),p+1)),0)
#18
round(max(c(log(n),p+1)),0)
#31
round(min(c(log(n),p+1)),0)
#6
#Nello specifico caso funziona molto meglio il max:

#Media
nearest<-nn2(X, query =X , k = 18, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[5]
#0.86
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
best_split=sort(b)
#0.437
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.3333333
#Area under the curve: 0.6611
#Precision: 0.6363636

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.952381
#Area under the curve: 0.8964
#Precision: 0.2597403

#Max
nearest<-nn2(X, query =X , k = 31, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
#Scegliamo manualmente uno split che potrebbe sembrare più accurato:
abline(h=0.7,col="blue")
best_split=0.7
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
best_split=sort(b)
#0.459
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.6666667
#Area under the curve: 0.8109
#Precision: 0.4666667

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 1
#Area under the curve: 0.9118
#Precision: 0.25

#Min
nearest<-nn2(X, query =X , k = 6, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[4]
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
best_split=sort(b)
#0.4277
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.7142857 
#Area under the curve: 0.8249
#Precision: 0.3947368

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Area under the curve: 0.8249
#Precision: 0.3947368
#Recall: 0.7142857

#Best k=6
#Best AUC=0.8249
#Best split= 0.43

#Oss: i risultati sono molto molto simili e ambigui.



# Glass - 9 outliers ---------------------------
data=readMat("glass.mat")
X=as.data.frame(data$X)
summary(X)
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
p=dim(X)[2]; n=dim(X)[1]

#(1)
#Usare un knn aggregato con la seguente scelta di parametri:
knn.agg<-KNN_AGG(X, k_min = min(log(n),p+1), k_max = max(log(n),p+1))
plot(knn.agg)

a <- rpart_split_finder(knn.agg, n)
plot(sort(knn.agg))
plot(sort(knn.agg), col=is_out[order(knn.agg)]+1)
abline(h=a)
abline(h=45,col="blue")
best_split=45
#Il solo taglio preso in considerazione è quello scelto manualmente che non coincide con gli altri
#suggeriti da entrambi
outliers_a<-ifelse(knn.agg>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.5555556
#Area under the curve: 0.6949
#Precision: 0.1282051

#(2)
#si presenta un diferente algoritmo in grado di gestire grandi dataset.
#usare come K una media, oppure il massimo o il minimo:
round(mean(c(log(n),p+1)),0)
#8
round(max(c(log(n),p+1)),0)
#10
round(min(c(log(n),p+1)),0)
#5
#Nello specifico caso funziona molto meglio il max:

#Media
nearest<-nn2(X, query =X , k = 8, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score, col=is_out+1)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[7]
#1.30
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
best_split=sort(b)[2]
#0.609
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.1724138
#Area under the curve: 0.7192
#Precision: 0.5555556

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 1
#Area under the curve: 0.8732
#Precision: 0.14754

#Max
nearest<-nn2(X, query =X , k = 10, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[6]
outliers_a<-ifelse(score>best_split,1,0)

#Non consideriamo il ctree percè gli split proposti sono molto lontani da quelli che sceglieremmo, quindi
#Scegliamo solo il primo che è molto plausibile

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.444444
#Area under the curve:0.6612
#Precision: 0.137931


#Min
nearest<-nn2(X, query =X , k = 5, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
#scegliamo lo split manulamente:
abline(h=0.60, col="blue")
best_split=0.6
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[2]
#0.43978
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.7777778 
#Area under the curve: 0.784
#Precision: 0.14

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Area under the curve: 0.8537
#Precision: 0.1304348
#Recall: 1

#Best k=8
#Best AUC=0.7192
#Best split= 1.30

#Oss: i risultati sono molto molto simili e ambigui.


#Dati Scalati:
X_scale<-scale(X)

#(1)
#Usare un knn aggregato con la seguente scelta di parametri:
knn.agg<-KNN_AGG(X_scale, k_min = min(log(n),p+1), k_max = max(log(n),p+1))
plot(knn.agg)

a <- rpart_split_finder(knn.agg, n)
plot(sort(knn.agg))
plot(sort(knn.agg), col=is_out[order(knn.agg)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[6]
#Il solo taglio preso in considerazione è quello scelto manualmente che non coincide con gli altri
#suggeriti da entrambi
outliers_a<-ifelse(knn.agg>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.5555556
#Area under the curve: 0.6997
#Precision: 0.1351351

#(2)
#si presenta un diferente algoritmo in grado di gestire grandi dataset.
#usare come K una media, oppure il massimo o il minimo:
round(mean(c(log(n),p+1)),0)
#8
round(max(c(log(n),p+1)),0)
#10
round(min(c(log(n),p+1)),0)
#5
#Nello specifico caso funziona molto meglio il max:

#Media
nearest<-nn2(X_scale, query =X_scale , k = 8, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score, col=is_out+1)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
abline(h=2,col="blue")
best_split=2
#2
outliers_a<-ifelse(score>best_split,1,0)

#Scegliamo un solo split con h=2

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.3333333
#Area under the curve: 0.6081
#Precision: 0.1111111


#Max
nearest<-nn2(X_scale, query =X_scale , k = 10, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[6]
#1.65
outliers_a<-ifelse(score>best_split,1,0)

#Non consideriamo il ctree percè gli split proposti sono molto lontani da quelli che sceglieremmo, quindi
#Scegliamo solo il primo che è molto plausibile

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.5555556
#Area under the curve:0.6973
#Precision:  0.1315789


#Min
nearest<-nn2(X_scale, query =X_scale , k = 5, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
#scegliamo lo split manulamente:
abline(h=1.4, col="blue")
best_split=1.4
outliers_a<-ifelse(score>best_split,1,0)

#Non serve fare anche la b perchè non viene molto bene, rimaniamo su quello ipotizzato

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.4444444 
#Area under the curve: 0.6515
#Precision: 0.1212121


#Best k=8
#Best AUC=0.7192
#Best split= 1.30

#Oss: i risultati sono molto molto simili e ambigui.


# Vowels - 50 outliers -------------------------
data=readMat("vowels.mat")
X=as.data.frame(data$X)
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
p=dim(X)[2]; n=dim(X)[1]


#(1)
#Usare un knn aggregato con la seguente scelta di parametri:
knn.agg<-KNN_AGG(X, k_min = min(log(n),p+1), k_max = max(log(n),p+1))
plot(knn.agg)

a <- rpart_split_finder(knn.agg, n)
plot(sort(knn.agg), col=is_out[order(knn.agg)]+1)
abline(h=a)
sort(a)
#118
best_split=sort(a)[4]
#Il solo taglio preso in considerazione è quello scelto manualmente che non coincide con gli altri
#suggeriti da entrambi
outliers_a<-ifelse(knn.agg>best_split,1,0)

b <-ctree_split_finder(knn.agg,n)  
plot(sort(knn.agg), col=is_out[order(knn.agg)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[2]
#117.95495   (è uguale ad a)
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.84
#Area under the curve: 0.9086
#Precision: 0.5675676

#(2)
#si presenta un diferente algoritmo in grado di gestire grandi dataset.
#usare come K una media, oppure il massimo o il minimo:
round(mean(c(log(n),p+1)),0)
#10
round(max(c(log(n),p+1)),0)
#13
round(min(c(log(n),p+1)),0)
#7
#Nello specifico caso funziona molto meglio il max:

#Media
nearest<-nn2(X, query =X , k = 10, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score, col=is_out+1)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[5]
#1.81
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[4]
#1.82901
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.86
#Area under the curve: 0.9179
#Precision:0.5584416

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.86
#Area under the curve: 0.9179
#Precision:0.5584416

#Max
nearest<-nn2(X, query =X , k = 13, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[7]
#1.97
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[4]
#1.96445   Uguale a quello sopra
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.86
#Area under the curve:0.9161
#Precision: 0.5243902


#Min
nearest<-nn2(X, query =X , k = 7, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[6]
#1.66
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[5]
#0.43978
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.7 
#Area under the curve: 0.8408
#Precision: 0.5737705

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Area under the curve: 0.8415
#Precision: 0.5932203
#Recall: 0.7

#Best k=10
#Best AUC=0.9179
#Best split= 1.81



# Cardio - 176 outliers ------------------------
data=readMat("cardio.mat")
X=as.data.frame(data$X)
is_out<-as.vector(data$y)
summary(X)
dati<-data.frame(cbind(X, is_out))
p=dim(X)[2]; n=dim(X)[1]

#(1)
#Usare un knn aggregato con la seguente scelta di parametri:
knn.agg<-KNN_AGG(X, k_min = min(log(n),p+1), k_max = max(log(n),p+1))
plot(knn.agg)

a <- rpart_split_finder(knn.agg, n)
plot(sort(knn.agg), col=is_out[order(knn.agg)]+1)
abline(h=a)
sort(a)
#971
best_split=sort(a)[4]
outliers_a<-ifelse(knn.agg>best_split,1,0)

b <-ctree_split_finder(knn.agg,n)  
plot(sort(knn.agg), col=is_out[order(knn.agg)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[2]
#970.6521   (è uguale ad a)
outliers_b<-ifelse(knn.agg>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.1704545
#Area under the curve: 0.5819
#Precision: 0.7317073

#(2)
#si presenta un diferente algoritmo in grado di gestire grandi dataset.
#usare come K una media, oppure il massimo o il minimo:
round(mean(c(log(n),p+1)),0)
#15
round(max(c(log(n),p+1)),0)
#22
round(min(c(log(n),p+1)),0)
#8
#Nello specifico caso funziona molto meglio il max:

#Media
nearest<-nn2(X, query =X , k = 15, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score, col=is_out+1)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[6]
#4.33
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[2]
#4.26726
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.1590909
#Area under the curve: 0.5765
#Precision:0.7368421

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.1647727
#Area under the curve: 0.5791
#Precision:0.725

#Max
nearest<-nn2(X, query =X , k = 22, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[7]
#5.30
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=sort(b)
#5.19945   Uguale a quello sopra
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.09090909
#Area under the curve:0.5442
#Precision: 0.8


#Min
nearest<-nn2(X, query =X , k = 8, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
abline(h=3.5,col="blue")
best_split=3.5
#Segliamo lo split manualmente non soddisfatti di quelli proposti
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  #NoSplits!

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.09090909 
#Area under the curve: 0.5442
#Precision: 0.8


#Best k=15
#Best AUC=0.5765
#Best split= 4.33

#Viene malissimo!

# Thyroid - 93 outliers ------------------------
data=readMat("thyroid.mat")
X=as.data.frame(data$X)
Y=data$y
summary(X)
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
p=dim(X)[2]; n=dim(X)[1]

#(1)
#Usare un knn aggregato con la seguente scelta di parametri:
knn.agg<-KNN_AGG(X, k_min = min(log(n),p+1), k_max = max(log(n),p+1))
plot(knn.agg)

a <- rpart_split_finder(knn.agg, n)
plot(sort(knn.agg), col=is_out[order(knn.agg)]+1)
abline(h=a)
#Scegliamo manualmente la a), il taglio:
abline(h=2, col="blue")
#2
best_split=2
outliers_a<-ifelse(knn.agg>best_split,1,0)

b <-ctree_split_finder(knn.agg,n)  #No Split

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Area under the curve: 0.6082
#Precision: 0.2291667
#Recall: 0.2365591

#(2)
#si presenta un diferente algoritmo in grado di gestire grandi dataset.
#usare come K una media, oppure il massimo o il minimo:
round(mean(c(log(n),p+1)),0)
#8
round(max(c(log(n),p+1)),0)
#8
round(min(c(log(n),p+1)),0)
#7
#Nello specifico caso funziona molto meglio il max:

#Media=Max
nearest<-nn2(X, query =X , k = 8, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score, col=is_out+1)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
abline(h=0.1,col="blue")
#Scegliamo lo split manualmente:
best_split=0.1
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)   #No splits

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.4193548
#Area under the curve: 0.6949
#Precision:0.2635135

#Min
nearest<-nn2(X, query =X , k = 7, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
#Scegliamo noi manualmente lo split
abline(h=0.09, col="blue")
best_split=0.09
#0-09
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  #No Split

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.5053763 
#Area under the curve: 0.7352
#Precision: 0.2670455

#Best k=7
#Best AUC=0.7352
#Best split= 0.09


#####
dati_comp <- read.csv("ann_train.csv", sep=" ", header=F)
dati_comp <- dati_comp[,-c(23,24)]
fac <- c(2:16)
y <- dati_comp$V22
table(y) #is_out è la somma di 1 e 2
y <- ifelse(y==1, 1, 0)
table(y)
X <- dati_comp[, -22]
p=dim(X)[2]; n=dim(X)[1]
is_out <- y
dati<-data.frame(cbind(X, is_out)) 
for (i in 1:length(fac)){
  dati[,i]<-as.factor(dati[,i])
}
#One hot encoding:
dati<-cbind(as.data.frame(model.matrix(~.-1,data=dati[,-(p+1)])),is_out)
X=dati[,1:(ncol(dati)-1)]

#(1)
#Usare un knn aggregato con la seguente scelta di parametri:
knn.agg<-KNN_AGG(X, k_min = min(log(n),p+1), k_max = max(log(n),p+1))
plot(knn.agg)
a <- rpart_split_finder(knn.agg, n)
plot(sort(knn.agg), col=is_out[order(knn.agg)]+1)
abline(h=a)
#Scegliamo manualmente:
abline(h=200,col="blue")
#200
best_split=200
outliers_a<-ifelse(knn.agg>best_split,1,0)

b <-ctree_split_finder(knn.agg,n)           #No Split  

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.1612903
#Area under the curve: 0.4912
#Precision:0.02228826

#(2)
#si presenta un diferente algoritmo in grado di gestire grandi dataset.
#usare come K una media, oppure il massimo o il minimo:
round(mean(c(log(n),p+1)),0)
#15
round(max(c(log(n),p+1)),0)
#22
round(min(c(log(n),p+1)),0)
#8
#Nello specifico caso funziona molto meglio il max:

#Media
nearest<-nn2(X, query =X , k = 15, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score, col=is_out+1)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
abline(h=0.92,col="blue")
#Scegliamo lo split manualmente
best_split=0.92
#0.92
outliers_a<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.1505376
#Area under the curve: 0.4875
#Precision: 0.02121212


#Max
nearest<-nn2(X, query =X , k =22, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
#Split manuale:
abline(h=0.97,col="blue")
best_split=0.97
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)   #No Split 

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.1075269
#Area under the curve:0.4736
#Precision: 0.01666667


#Min
nearest<-nn2(X, query =X , k = 8, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)
a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
abline(h=0.89, col="blue")
best_split=0.89
#Scegliamo lo split manualmente
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  #No Split

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.06451613
#Area under the curve: 0.4987
#Precision: 0.02371542


#Best k=8
#Best AUC= 0.4987
#Best split= 0.89

#Tyroid completo e normalizzato:
X_scale<-scale(X)

#(1)
#Usare un knn aggregato con la seguente scelta di parametri:
knn.agg<-KNN_AGG(X_scale, k_min = min(log(n),p+1), k_max = max(log(n),p+1))
plot(knn.agg)

a <- rpart_split_finder(knn.agg, n)
plot(sort(knn.agg), col=is_out[order(knn.agg)]+1)
abline(h=a)
sort(a)
#2365
best_split=sort(a)[5]
#Il solo taglio preso in considerazione è quello scelto manualmente che non coincide con gli altri
#suggeriti da entrambi
outliers_a<-ifelse(knn.agg>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.08602151
#Area under the curve: 0.5291
#Precision:  0.07272727

#(2)
#si presenta un diferente algoritmo in grado di gestire grandi dataset.
#usare come K una media, oppure il massimo o il minimo:
round(mean(c(log(n),p+1)),0)
#15
round(max(c(log(n),p+1)),0)
#22
round(min(c(log(n),p+1)),0)
#8
#Nello specifico caso funziona molto meglio il max:

#Media
nearest<-nn2(X_scale, query =X_scale , k = 15, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score, col=is_out+1)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[5]
#10.29
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  #No split

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.08602151
#Area under the curve: 0.5282
#Precision: 0.06837607

#Max
nearest<-nn2(X_scale, query =X_scale , k = 22, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
#Split manuale
abline(h=10,col="blue")
best_split=10
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  #No Split

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.1075269
#Area under the curve: 0.5307
#Precision:  0.05555556


#Min
nearest<-nn2(X_scale, query =X_scale , k = 8, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[7]
#8.376
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n) #No split  

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.1075269 
#Area under the curve: 0.5373
#Precision: 0.07633588


#Best k=8
#Best AUC=  0.5373
#Best split= 8.736


#Dati Scalati:
X_scale=scale(X)

#(1)
#Usare un knn aggregato con la seguente scelta di parametri:
knn.agg<-KNN_AGG(X_scale, k_min = min(log(n),p+1), k_max = max(log(n),p+1))
plot(knn.agg)

a <- rpart_split_finder(knn.agg, n)
plot(sort(knn.agg),col=is_out[order(knn.agg)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[6]
#2365
outliers_a<-ifelse(knn.agg>best_split,1,0)

b <-ctree_split_finder(knn.agg,n)  #No splits!

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.08602151
#Area under the curve:0.5298
#Precision: 0.07619048



#(2)
#si presenta un diferente algoritmo in grado di gestire grandi dataset.
#usare come K una media, oppure il massimo o il minimo:
round(mean(c(log(n),p+1)),0)
#15
round(max(c(log(n),p+1)),0)
#22
round(min(c(log(n),p+1)),0)
#8
#Nello specifico caso funziona molto meglio il max:

#Media
nearest<-nn2(X_scale, query =X_scale , k = 15, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score, col=is_out+1)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[5]
#10.29
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  #No splits

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.08602151
#Area under the curve: 0.5282
#Precision: 0.06837607


#Max
nearest<-nn2(X_scale, query =X_scale , k = 22, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
abline(h=11, col="blue")
best_split=11
#11 split scelto manualmente
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  #No splits!

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.08602151
#Area under the curve: 0.5279
#Precision: 0.06722689


#Min
nearest<-nn2(X_scale, query =X_scale , k = 8, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
# 8.376
best_split=sort(a)[7]
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  #No splits!

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.1075269
#Area under the curve: 0.5373
#Precision: 0.07633588

#Best k=8
#Best AUC=0.5373
#Best split= 8.376



# Musk - 97 outliers ---------------------------
data=readMat("musk.mat")
X=as.data.frame(data$X)
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
p=dim(X)[2]; n=dim(X)[1]


#(1)
#Usare un knn aggregato con la seguente scelta di parametri:
knn.agg<-KNN_AGG(X, k_min = min(log(n),p+1), k_max = max(log(n),p+1))
plot(knn.agg)

a <- rpart_split_finder(knn.agg, n)
plot(sort(knn.agg),col=is_out[order(knn.agg)]+1)
abline(h=a)
abline(h=11500000,col="blue")
#Scegliamomanualmente lo split
#11500000
best_split=11500000
#Il solo taglio preso in considerazione è quello scelto manualmente che non coincide con gli altri
#suggeriti da entrambi
outliers_a<-ifelse(knn.agg>best_split,1,0)

b <-ctree_split_finder(knn.agg,n)  
plot(sort(knn.agg), col=is_out[order(knn.agg)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[4]
#11549932   (è uguale ad a)
outliers_b<-ifelse(knn.agg>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 1
#Area under the curve: 0.9788
#Precision: 0.4349776

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 1
#Area under the curve: 0.983
#Precision: 0.489899


#(2)
#si presenta un diferente algoritmo in grado di gestire grandi dataset.
#usare come K una media, oppure il massimo o il minimo:
round(mean(c(log(n),p+1)),0)
#88
round(max(c(log(n),p+1)),0)
#167
round(min(c(log(n),p+1)),0)
#8
#Nello specifico caso funziona molto meglio il max:

#Media
nearest<-nn2(X, query =X , k = 88, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score, col=is_out+1)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[7]
#810
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[4]
#809.5053 uguale a a
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.4948454
#Area under the curve:  0.7314
#Precision: 0.3356643

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.4948454
#Area under the curve:  0.7314
#Precision: 0.3356643

#Max
nearest<-nn2(X, query =X , k = 167, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1, xlab="n", ylab="kNN_score", main="kNN")
abline(h=sort(a)[7], col="blue")
sort(a)
best_split=sort(a)[7]
#978
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[4]
#948.6336  
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 1
#Area under the curve: 1
#Precision: 1

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 1
#Area under the curve: 0.9997
#Precision: 0.979798

#Min
nearest<-nn2(X, query =X , k = 8, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[6]
#453
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[4]
#453 Uguale al caso a)
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.05154639 
#Area under the curve: 0.4759
#Precision: 0.0166113

#Best k=167
#Best AUC=1
#Best split= 978

#Dati Scalati:
X_scale=scale(X)

#(1)
#Usare un knn aggregato con la seguente scelta di parametri:
knn.agg<-KNN_AGG(X_scale, k_min = min(log(n),p+1), k_max = max(log(n),p+1))
plot(knn.agg)

a <- rpart_split_finder(knn.agg, n)
plot(sort(knn.agg),col=is_out[order(knn.agg)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[6]
#173820
#Il solo taglio preso in considerazione è quello scelto manualmente che non coincide con gli altri
#suggeriti da entrambi
outliers_a<-ifelse(knn.agg>best_split,1,0)

b <-ctree_split_finder(knn.agg,n)  
plot(sort(knn.agg), col=is_out[order(knn.agg)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[4]
#173698.4
outliers_b<-ifelse(knn.agg>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 1
#Area under the curve:0.9992
#Precision: 0.9509804

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 1
#Area under the curve: 0.999
#Precision: 0.9417476


#(2)
#si presenta un diferente algoritmo in grado di gestire grandi dataset.
#usare come K una media, oppure il massimo o il minimo:
round(mean(c(log(n),p+1)),0)
#88
round(max(c(log(n),p+1)),0)
#167
round(min(c(log(n),p+1)),0)
#8
#Nello specifico caso funziona molto meglio il max:

#Media
nearest<-nn2(X_scale, query =X_scale , k = 88, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score, col=is_out+1)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[5]
#11.6
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[4]
#11.59452 uguale a a
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.9484536
#Area under the curve: 0.971
#Precision: 0.8288288

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.9484536
#Area under the curve: 0.971
#Precision: 0.8288288


#Max
nearest<-nn2(X_scale, query =X_scale , k = 167, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[5]
#14.3
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[4]
#13.73129  
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 1
#Area under the curve: 1
#Precision: 1

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 1
#Area under the curve: 1
#Precision: 1

#Min
nearest<-nn2(X_scale, query =X_scale , k = 8, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
abline(h=8, col="blue")
#8
best_split=8
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=8 #Uguale all' a)
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.1030928
#Area under the curve: 0.5509
#Precision: 0.7142857

#Best k=167
#Best AUC=1
#Best split= 14.3

# Satimage-2 - 71 outliers ---------------------
data=readMat("satimage-2.mat")
X=as.data.frame(data$X)
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
p=dim(X)[2]; n=dim(X)[1]

#(1)
#Usare un knn aggregato con la seguente scelta di parametri:
knn.agg<-KNN_AGG(X, k_min = min(log(n),p+1), k_max = max(log(n),p+1))
plot(knn.agg)

a <- rpart_split_finder(knn.agg, n)
plot(sort(knn.agg),col=is_out[order(knn.agg)]+1)
abline(h=a)
abline(h=32000,col="blue")
#Scegliamomanualmente lo split
#32000
best_split=32000
#Il solo taglio preso in considerazione è quello scelto manualmente che non coincide con gli altri
#suggeriti da entrambi
outliers_a<-ifelse(knn.agg>best_split,1,0)

b <-ctree_split_finder(knn.agg,n)  
plot(sort(knn.agg), col=is_out[order(knn.agg)]+1)
abline(h=b)
best_split=320000
#Uguale alla a)
outliers_b<-ifelse(knn.agg>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.6619718
#Area under the curve: 0.8162
#Precision: 0.2165899



#(2)
#si presenta un diferente algoritmo in grado di gestire grandi dataset.
#usare come K una media, oppure il massimo o il minimo:
round(mean(c(log(n),p+1)),0)
#23
round(max(c(log(n),p+1)),0)
#37
round(min(c(log(n),p+1)),0)
#9
#Nello specifico caso funziona molto meglio il max:

#Media
nearest<-nn2(X, query =X , k = 23, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score, col=is_out+1)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
#Facciamo lo split manuale!:
abline(h=43, col="blue")
best_split=43
#43
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=43, col="blue")
best_split=43
#43 uguale a a
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.7042254
#Area under the curve: 0.8273
#Precision: 0.1497006

#Max
nearest<-nn2(X, query =X , k = 37, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
#Aggiungiamo lo split manuale:
abline(h=45, col="blue")
best_split=45
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
#Aggiungiamo lo split manuale:
abline(h=45, col="blue")
best_split=45 
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: .9859155
#Area under the curve: 0.9641
#Precision: 0.1745636


#Min
nearest<-nn2(X, query =X , k = 9, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
#Scegliamo lo split manualmente:
abline(h=38, col="blue")
best_split=38
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
#Scegliamo lo split manualmente:
abline(h=38, col="blue")
best_split=38
#38 uguale ad a)
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.5070423 
#Area under the curve: 0.7381
#Precision: 0.1690141

#Best k=167
#Best AUC=1
#Best split= 978

#Dati Scalati:
X_scale=scale(X)

#(1)
#Usare un knn aggregato con la seguente scelta di parametri:
knn.agg<-KNN_AGG(X_scale, k_min = min(log(n),p+1), k_max = max(log(n),p+1))
plot(knn.agg)

a <- rpart_split_finder(knn.agg, n)
plot(sort(knn.agg),col=is_out[order(knn.agg)]+1)
abline(h=a)
#Scegliere lo split manuale:
abline(h=2000, col="blue")
best_split=2000
#Il solo taglio preso in considerazione è quello scelto manualmente che non coincide con gli altri
#suggeriti da entrambi
outliers_a<-ifelse(knn.agg>best_split,1,0)

b <-ctree_split_finder(knn.agg,n)  
plot(sort(knn.agg), col=is_out[order(knn.agg)]+1)
abline(h=b)
#Scegliere lo split manuale:
abline(h=2000, col="blue")
best_split=2000
#Stesso split di a)
outliers_b<-ifelse(knn.agg>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.7183099
#Area under the curve: 0.8414
#Precision: 0.2007874


#(2)
#si presenta un diferente algoritmo in grado di gestire grandi dataset.
#usare come K una media, oppure il massimo o il minimo:
round(mean(c(log(n),p+1)),0)
#23
round(max(c(log(n),p+1)),0)
#37
round(min(c(log(n),p+1)),0)
#9
#Nello specifico caso funziona molto meglio il max:

#Media
nearest<-nn2(X_scale, query =X_scale , k = 23, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score, col=is_out+1)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
#split manuale:
abline(h=3, col="blue")
best_split=3
#3
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
#split manuale:
abline(h=3, col="blue")
best_split=3
#3 uguale a a
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.2025316
#Area under the curve: 0.8215
#Precision: 0.6760563


#Max
nearest<-nn2(X_scale, query =X_scale , k = 37, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
#scegliere split manualmente:
abline(h=3.3, col="blue")
best_split=3
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
#scegliere split manualmente:
abline(h=3.3, col="blue")
best_split=3 #uguale ad a) 
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall:  0.9859155
#Area under the curve:  0.9672
#Precision: 0.1917808

#Min
nearest<-nn2(X_scale, query =X_scale , k = 9, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
#Split manuale
abline(h=2.3, col="blue")
#2.3
best_split=2.3
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
#Split manuale
abline(h=2.3, col="blue")
#2.3
best_split=2.3 #Uguale all' a)
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.5774648
#Area under the curve: 0.7648
#Precision:  0.1301587

#Best k=37
#Best AUC=0.9672
#Best split= 3

# Letter Recognition - 100 outliers ------------
path <- "letter.mat"
data=readMat(path)
X=data$X
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
summary(dati)
p=dim(X)[2]; n=dim(X)[1]


#(1)
#Usare un knn aggregato con la seguente scelta di parametri:
knn.agg<-KNN_AGG(X, k_min = min(log(n),p+1), k_max = max(log(n),p+1))
plot(knn.agg)

a <- rpart_split_finder(knn.agg, n)
plot(sort(knn.agg),col=is_out[order(knn.agg)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[7]
#5602
outliers_a<-ifelse(knn.agg>best_split,1,0)

b <-ctree_split_finder(knn.agg,n)  
plot(sort(knn.agg), col=is_out[order(knn.agg)]+1)
abline(h=b)
sort(b)
#5602.156 uguale ad a
best_split=sort(b)[6]
outliers_b<-ifelse(knn.agg>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.42
#Area under the curve: 0.69
#Precision:  0.4117

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.42
#Area under the curve: 0.69
#Precision:  0.4117


#(2)
#si presenta un diferente algoritmo in grado di gestire grandi dataset.
#usare come K una media, oppure il massimo o il minimo:
round(mean(c(log(n),p+1)),0)
#20
round(max(c(log(n),p+1)),0)
#33
round(min(c(log(n),p+1)),0)
#7
#Nello specifico caso funziona molto meglio il max:

#Media
nearest<-nn2(X, query =X , k = 20, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score, col=is_out+1)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
#spli manuale:
abline(h=10,col="blue")
best_split=10
#10
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
#spli manuale:
abline(h=10,col="blue")
best_split=10
#10 uguale a a
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.5
#Area under the curve:  0.7193
#Precision: 0.33


#Max
nearest<-nn2(X, query =X , k = 33, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[6]
#11.1
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[4]
#11.087
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.42
#Area under the curve: 0.6833
#Precision: 0.344

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.43
#Area under the curve: 0.688
#Precision: 0.3467

#Min
nearest<-nn2(X, query =X , k = 7, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
#Spli manuale:
abline(h=8,col="blue")
best_split=8
#8
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[4]
#7.34 Uguale al caso a)
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.58
#Area under the curve: 0.7663
#Precision: 0.44

confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.76
#Area under the curve:0.8313
#Precision: 0.342


#Best k=7
#Best AUC=0.8313
#Best split= 7.34

#Dati Scalati:
X_scale=scale(X)

#(1)
#Usare un knn aggregato con la seguente scelta di parametri:
knn.agg<-KNN_AGG(X_scale, k_min = min(log(n),p+1), k_max = max(log(n),p+1))
plot(knn.agg)

a <- rpart_split_finder(knn.agg, n)
plot(sort(knn.agg),col=is_out[order(knn.agg)]+1)
abline(h=a)
abline(h=2500,col="blue")
best_split=2500
#2500
outliers_a<-ifelse(knn.agg>best_split,1,0)

b <-ctree_split_finder(knn.agg,n)  
plot(sort(knn.agg), col=is_out[order(knn.agg)]+1)
abline(h=b)
abline(h=2500,col="blue")
best_split=2500
#2500  uguale ad a)
outliers_b<-ifelse(knn.agg>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.31
#Area under the curve:0.634
#Precision: 0.3297


#(2)
#si presenta un diferente algoritmo in grado di gestire grandi dataset.
#usare come K una media, oppure il massimo o il minimo:
round(mean(c(log(n),p+1)),0)
#20
round(max(c(log(n),p+1)),0)
#33
round(min(c(log(n),p+1)),0)
#7
#Nello specifico caso funziona molto meglio il max:

#Media
nearest<-nn2(X_scale, query =X_scale , k = 20, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score, col=is_out+1)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
abline(h=4.5,col="blue")
best_split=4.5
#4.5
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
abline(h=4.5,col="blue")
best_split=4.5
#4.5 uguale a a
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.39
#Area under the curve: 0.6683
#Precision: 0.3277


#Max
nearest<-nn2(X_scale, query =X_scale , k = 33, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[5]
#5.1
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[5]
#5.12482 
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.28
#Area under the curve: 0.6223
#Precision: 0.345679

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.28
#Area under the curve: 0.6237
#Precision: 0.3636

#Min
nearest<-nn2(X_scale, query =X_scale , k = 7, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
abline(h=3.5, col="blue")
#3.5
best_split=3.5
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[4]
#3.23858
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.53
#Area under the curve:  0.738
#Precision: 0.3955

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.71
#Area under the curve:  0.8073
#Precision: 0.3317757

#Best k=7
#Best AUC=0.8073
#Best split= 3.23858

# Speech - 61 outliers -------------------------
path <- "speech.mat"
data=readMat(path)
X=data$X
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
summary(dati)

is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
p=dim(X)[2]; n=dim(X)[1]


#(1)
#Usare un knn aggregato con la seguente scelta di parametri:
knn.agg<-KNN_AGG(X, k_min = min(log(n),p+1), k_max = max(log(n),p+1))
plot(knn.agg)

a <- rpart_split_finder(knn.agg, n)
plot(sort(knn.agg),col=is_out[order(knn.agg)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[7]
# 1995664
outliers_a<-ifelse(knn.agg>best_split,1,0)

b <-ctree_split_finder(knn.agg,n)  
plot(sort(knn.agg), col=is_out[order(knn.agg)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[6]
#1995600
outliers_b<-ifelse(knn.agg>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.09836066
#Area under the curve: 0.5148
#Precision: 0.02352941

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.09836066
#Area under the curve: 0.5148
#Precision: 0.02352941


#(2)
#si presenta un diferente algoritmo in grado di gestire grandi dataset.
#usare come K una media, oppure il massimo o il minimo:
round(mean(c(log(n),p+1)),0)
#205
round(max(c(log(n),p+1)),0)
#401
round(min(c(log(n),p+1)),0)
#8

#Media
nearest<-nn2(X, query =X , k = 205, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score, col=is_out+1)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[5]
#25
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=25
#25 uguale a a
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.06557377
#Area under the curve: 0.5109
#Precision: 0.02453988


#Max
nearest<-nn2(X, query =X , k = 401, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[5]
#25
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[5]
#25, stesso di a)
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.1147541
#Area under the curve: 0.5196
#Precision:  0.02491103


#Min
nearest<-nn2(X, query =X , k = 8, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
abline(h=20, col="blue")
#scelta manuale dello split: 20
best_split=20
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=20
#20 Uguale al caso a)
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.1147541 
#Area under the curve: 0.529
#Precision:  0.03286385

#Best k=8
#Best AUC=0.529
#Best split= 20

#Dati Scalati:
X_scale=scale(X)

#(1)
#Usare un knn aggregato con la seguente scelta di parametri:
knn.agg<-KNN_AGG(X_scale, k_min = min(log(n),p+1), k_max = max(log(n),p+1))

a <- rpart_split_finder(knn.agg, n)
plot(sort(knn.agg),col=is_out[order(knn.agg)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[8]
#2204043
outliers_a<-ifelse(knn.agg>best_split,1,0)

b <-ctree_split_finder(knn.agg,n)  
plot(sort(knn.agg), col=is_out[order(knn.agg)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[6]
#2203534
outliers_b<-ifelse(knn.agg>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.09836066
#Area under the curve:0.5161
#Precision: 0.02439024

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.09836066
#Area under the curve:0.5161
#Precision: 0.02439024


#(2)
#si presenta un diferente algoritmo in grado di gestire grandi dataset.
#usare come K una media, oppure il massimo o il minimo:
round(mean(c(log(n),p+1)),0)
#205
round(max(c(log(n),p+1)),0)
#401
round(min(c(log(n),p+1)),0)
#8
#Nello specifico caso funziona molto meglio il max:

#Media
nearest<-nn2(X_scale, query =X_scale , k = 205, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score, col=is_out+1)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
abline(h=28, col="blue")
best_split=28
#28 split scelto manualmente
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
abline(h=28, col="blue")
best_split=28
#28 split scelto manualmente
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.0212766
#Area under the curve: 0.5037
#Precision: 0.0212766

#Max
nearest<-nn2(X_scale, query =X_scale , k = 401, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[5]
#28
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[5]
#27.60051
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.06557377
#Area under the curve: 0.5096
#Precision: 0.02325581

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.1147
#Area under the curve: 0.5197
#Precision: 0.025

#Min
nearest<-nn2(X_scale, query =X_scale , k = 8, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
abline(h=22, col="blue")
#22  scegliamo manualmente lo split
best_split=22
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=22, col="blue")
#22  scegliamo manualmente lo split
best_split=22
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.1147541
#Area under the curve: 0.5265
#Precision: 0.03030303

#Best k=8
#Best AUC=0.5265
#Best split= 22


# Pima - 268 outliers --------------------------
path <- "pima.mat"
data=readMat(path)
X=data$X
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
head(dati)
summary(dati)
p=dim(X)[2]; n=dim(X)[1]

#(1)
#Usare un knn aggregato con la seguente scelta di parametri:
knn.agg<-KNN_AGG(X, k_min = min(log(n),p+1), k_max = max(log(n),p+1))
plot(knn.agg)

a <- rpart_split_finder(knn.agg, n)
plot(sort(knn.agg),col=is_out[order(knn.agg)]+1)
abline(h=a)
#Bisogna fare lo split manualmente
#1000
best_split=1000
outliers_a<-ifelse(knn.agg>best_split,1,0)

b <-ctree_split_finder(knn.agg,n) #No Split!

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.0261194
#Area under the curve: 0.5041
#Precision: 0.4375


#(2)
#si presenta un diferente algoritmo in grado di gestire grandi dataset.
#usare come K una media, oppure il massimo o il minimo:
round(mean(c(log(n),p+1)),0)
#8
round(max(c(log(n),p+1)),0)
#9
round(min(c(log(n),p+1)),0)
#7

#Media
nearest<-nn2(X, query =X , k = 8, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score, col=is_out+1)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
abline(h=40,col="blue")
best_split=40
#40 split manuale!
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  #No split!

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.03358209
#Area under the curve: 0.5068
#Precision: 0.4736842


#Max
nearest<-nn2(X, query =X , k = 9, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
abline(h=40,col="blue")
best_split=40
#40 split manuale!
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  #.No split

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.04477612
#Area under the curve: 0.5094
#Precision:  0.48


#Min
nearest<-nn2(X, query =X , k = 7, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
abline(h=30,col="blue")
best_split=30
#38 split manuale!
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  #Nosplit

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.09701493 
#Area under the curve: 0.5235
#Precision:  0.5098039

#Best k=7
#Best AUC=0.5235
#Best split= 30

#Dati Scalati:
X_scale=scale(X)

#(1)
#Usare un knn aggregato con la seguente scelta di parametri:
knn.agg<-KNN_AGG(X_scale, k_min = min(log(n),p+1), k_max = max(log(n),p+1))

a <- rpart_split_finder(knn.agg, n)
plot(sort(knn.agg),col=is_out[order(knn.agg)]+1)
abline(h=a)
abline(h=45,col="blue")
#45 split manuale
best_split=45
outliers_a<-ifelse(knn.agg>best_split,1,0)

b <-ctree_split_finder(knn.agg,n)  #No split  

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.1007463
#Area under the curve: 0.5304
#Precision: 0.5744681

#(2)
#si presenta un diferente algoritmo in grado di gestire grandi dataset.
#usare come K una media, oppure il massimo o il minimo:
round(mean(c(log(n),p+1)),0)
#8
round(max(c(log(n),p+1)),0)
#9
round(min(c(log(n),p+1)),0)
#7
#Nello specifico caso funziona molto meglio il max:

#Media
nearest<-nn2(X_scale, query =X_scale , k = 8, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score, col=is_out+1)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
abline(h=1.90, col="blue")
best_split=1.90
#1.90 split scelto manualmente
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n) #No Split

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.1007463
#Area under the curve: 0.5324
#Precision: 0.6

#Max
nearest<-nn2(X_scale, query =X_scale , k = 9, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
abline(h=2,col="blue")
best_split=2
#2 scelto manualmente
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  #No Split

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.09701493
#Area under the curve: 0.5325
#Precision: 0.6190476

#Min
nearest<-nn2(X_scale, query =X_scale , k = 7, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
abline(h=1.9, col="blue")
#1.9  scegliamo manualmente lo split
best_split=1.9
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n) #No Split  

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.0858209
#Area under the curve: 0.5299
#Precision: 0.6388889

#Best k=8
#Best AUC=0.5324
#Best split= 1.9

#Decisione complessa, tutto molto simile

# Satellite - 2036 outliers --------------------
path <- "satellite.mat"
data=readMat(path)
X=data$X
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
head(dati)
summary(dati)

p=dim(X)[2]; n=dim(X)[1]


#(1)
#Usare un knn aggregato con la seguente scelta di parametri:
knn.agg<-KNN_AGG(X, k_min = min(log(n),p+1), k_max = max(log(n),p+1))
plot(knn.agg)

a <- rpart_split_finder(knn.agg, n)
plot(sort(knn.agg),col=is_out[order(knn.agg)]+1)
abline(h=a)
abline(h=27000, col="blue")
best_split=27000
#Split manuale!
outliers_a<-ifelse(knn.agg>best_split,1,0)

b <-ctree_split_finder(knn.agg,n)  #No split

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.2588409
#Area under the curve: 0.6066
#Precision: 0.7239011


#(2)
#si presenta un diferente algoritmo in grado di gestire grandi dataset.
#usare come K una media, oppure il massimo o il minimo:
round(mean(c(log(n),p+1)),0)
#23
round(max(c(log(n),p+1)),0)
#37
round(min(c(log(n),p+1)),0)
#9

#Media
nearest<-nn2(X, query =X , k = 23, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score, col=is_out+1)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
abline(h=40, col="blue")
best_split=40
#Split manuale!
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n) #No split  

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.259332
#Area under the curve: 0.6067
#Precision: 0.7232877


#Max
nearest<-nn2(X, query =X , k = 37, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
abline(h=44, col="blue")
best_split=44
#44 split manuale!
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n) #No Split 

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.2504912
#Area under the curve: 0.6049
#Precision:  0.7402032


#Min
nearest<-nn2(X, query =X , k = 9, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
abline(h=33, col="blue")
#scelta manuale dello split: 33
best_split=33
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
abline(h=33, col="blue")
#scelta manuale dello split: 33 caso a)
best_split=33
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.2347741 
#Area under the curve: 0.5941
#Precision:  0.6998536

#Best k=23
#Best AUC=0.6067
#Best split= 40

#Dati Scalati:
X_scale=scale(X)

#(1)
#Usare un knn aggregato con la seguente scelta di parametri:
knn.agg<-KNN_AGG(X_scale, k_min = min(log(n),p+1), k_max = max(log(n),p+1))
a <- rpart_split_finder(knn.agg, n)
plot(sort(knn.agg),col=is_out[order(knn.agg)]+1)
abline(h=a)
abline(h=1500, col="blue")
best_split=1500
#1500 split scelto manualmente
outliers_a<-ifelse(knn.agg>best_split,1,0)

b <-ctree_split_finder(knn.agg,n) #No split! 

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.2519646
#Area under the curve: 0.6013
#Precision: 0.7027397

#(2)
#si presenta un diferente algoritmo in grado di gestire grandi dataset.
#usare come K una media, oppure il massimo o il minimo:
round(mean(c(log(n),p+1)),0)
#23
round(max(c(log(n),p+1)),0)
#37
round(min(c(log(n),p+1)),0)
#9
#Nello specifico caso funziona molto meglio il max:

#Media
nearest<-nn2(X_scale, query =X_scale , k = 23, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score, col=is_out+1)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
abline(h=2.2, col="blue")
best_split=2.2
#2.2 split scelto manualmente
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n) #No split! 

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.2568762
#Area under the curve: 0.6025
#Precision: 0.6964048

#Max
nearest<-nn2(X_scale, query =X_scale , k = 37, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
abline(h=2.3, col="blue")
best_split=2.3
#2.3 split scelto manualmente
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n) #No Split! 

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.2775049
#Area under the curve: 0.6089
#Precision: 0.6823671


#Min
nearest<-nn2(X_scale, query =X_scale , k = 9, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
abline(h=2.1, col="blue")
#2.1  scegliamo manualmente lo split
best_split=2.1
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
abline(h=2, col="blue")
#2  scegliamo manualmente lo split
best_split=2
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.1281925
#Area under the curve: 0.5527
#Precision: 0.7229917

#Best k=37
#Best AUC=0.6089
#Best split= 2.3



# Shuttle - 3511 outliers ----------------------
path <- "shuttle.mat"
data=readMat(path)
X=data$X
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
head(dati)
summary(dati)
p=dim(X)[2]; n=dim(X)[1]

#(1)
#Usare un knn aggregato con la seguente scelta di parametri:
knn.agg<-KNN_AGG(X, k_min = min(log(n),p+1), k_max = max(log(n),p+1))
plot(knn.agg)

a <- rpart_split_finder(knn.agg, n)
plot(30000:n,sort(knn.agg)[30000:n],col=is_out[order(knn.agg)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[4]
#Non è molto comprensibile in realtà
outliers_a<-ifelse(knn.agg>best_split,1,0)

b <-ctree_split_finder(knn.agg,n)  #No Split!  

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.06664768
#Area under the curve: 0.5301
#Precision: 0.442344


#(2)
#si presenta un diferente algoritmo in grado di gestire grandi dataset.
#usare come K una media, oppure il massimo o il minimo:
round(mean(c(log(n),p+1)),0)
#10
round(max(c(log(n),p+1)),0)
#11
round(min(c(log(n),p+1)),0)
#10
#Nello specifico caso funziona molto meglio il max:

#Media=Min
nearest<-nn2(X, query =X , k = 10, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score, col=is_out+1)

a <- rpart_split_finder(score, n)
plot(sort(score)[7000:20000], col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[1]
#5
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n) #NoSplit 

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.1330105
#Area under the curve:  0.557
#Precision: 0.3495509


#Max
nearest<-nn2(X, query =X , k = 11, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[1]
#5.2
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  #No split!

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.1321561
#Area under the curve: 0.5567
#Precision: 0.3528517


#Best k=11
#Best AUC= 0.5567
#Best split= 5.2

#Dati Scalati:
X_scale=scale(X)

#(1)
#Usare un knn aggregato con la seguente scelta di parametri:
knn.agg<-KNN_AGG(X_scale, k_min = min(log(n),p+1), k_max = max(log(n),p+1))
plot(knn.agg)

a <- rpart_split_finder(knn.agg, n)
plot(sort(knn.agg),col=is_out[order(knn.agg)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[4]
#2.8
#Il solo taglio preso in considerazione è quello scelto manualmente che non coincide con gli altri
#suggeriti da entrambi
outliers_a<-ifelse(knn.agg>best_split,1,0)

b <-ctree_split_finder(knn.agg,n) #No Split  

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.06750214
#Area under the curve:0.5312
#Precision: 0.5064103


#(2)
#si presenta un diferente algoritmo in grado di gestire grandi dataset.
#usare come K una media, oppure il massimo o il minimo:
round(mean(c(log(n),p+1)),0)
#10
round(max(c(log(n),p+1)),0)
#11
round(min(c(log(n),p+1)),0)
#10
#Nello specifico caso funziona molto meglio il max:

#Media = MIn
nearest<-nn2(X_scale, query =X_scale , k = 10, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score, col=is_out+1)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[7]
#0.771
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)   #No Split

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.02278553
#Area under the curve: 0.51
#Precision: 0.3902439


#Max
nearest<-nn2(X_scale, query =X_scale , k = 11, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[10]
#0.272
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n) #No Split 

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.06209057
#Area under the curve: 0.5286
#Precision: 0.490991

#Best k=11
#Best AUC=0.5286
#Best split= 0.272


# BreastW - 239 outliers -----------------------
path="breastw.mat"
data=readMat(path)
X=data$X
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
summary(dati)
p=dim(X)[2]; n=dim(X)[1]


#(1)
#Usare un knn aggregato con la seguente scelta di parametri:
knn.agg<-KNN_AGG(X, k_min = min(log(n),p+1), k_max = max(log(n),p+1))
plot(knn.agg)

a <- rpart_split_finder(knn.agg, n)
plot(sort(knn.agg),col=is_out[order(knn.agg)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[3]
#96.7
outliers_a<-ifelse(knn.agg>best_split,1,0)

b <-ctree_split_finder(knn.agg,n)  
plot(sort(knn.agg), col=is_out[order(knn.agg)]+1)
abline(h=b)
sort(b)
best_split=b
#96.3524  (è uguale ad a)
outliers_b<-ifelse(knn.agg>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.9958159
#Area under the curve: 0.9596
#Precision: 0.875

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.9958159
#Area under the curve: 0.9585
#Precision: 0.8717949


#(2)
#si presenta un diferente algoritmo in grado di gestire grandi dataset.
#usare come K una media, oppure il massimo o il minimo:
round(mean(c(log(n),p+1)),0)
#8
round(max(c(log(n),p+1)),0)
#10
round(min(c(log(n),p+1)),0)
#7
#Nello specifico caso funziona molto meglio il max:

#Media
nearest<-nn2(X, query =X , k = 8, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score, col=is_out+1)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[3]
#2.7
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=sort(b)
#2.61273
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.9958159
#Area under the curve: 0.9585
#Precision: 0.8717949

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.9958159
#Area under the curve:  0.9574
#Precision: 0.8686131

#Max
nearest<-nn2(X, query =X , k = 10, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[4]
#3
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=sort(b)
#2.946
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 1
#Area under the curve: 0.955
#Precision: 0.8566308

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.9958159
#Area under the curve: 0.9607
#Precision: 0.8782288

#Min
nearest<-nn2(X, query =X , k = 7, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[3]
#2.6
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=sort(b)
#2.51218
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.9958159 
#Area under the curve: 0.9585
#Precision: 0.8717949

#Best k=10
#Best AUC=0.9607
#Best split= 2.946
#Sono tutti molto molto simili



fac<-1:p
for (i in 1:length(fac)){
  dati[,i]<-as.factor(dati[,i])
}
#One hot encoding:
dati<-cbind(as.data.frame(model.matrix(~.-1,data=dati[,-(p+1)])),is_out)
X=dati[,1:(ncol(dati)-1)]
p=dim(X)[2]; n=dim(X)[1]

#(1)
#Usare un knn aggregato con la seguente scelta di parametri:
knn.agg<-KNN_AGG(X, k_min = min(log(n),p+1), k_max = max(log(n),p+1))
plot(knn.agg)

a <- rpart_split_finder(knn.agg, n)
plot(sort(knn.agg),col=is_out[order(knn.agg)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[6]
#7472
outliers_a<-ifelse(knn.agg>best_split,1,0)

b <-ctree_split_finder(knn.agg,n)  
plot(sort(knn.agg), col=is_out[order(knn.agg)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[4]
#7456.848
outliers_b<-ifelse(knn.agg>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.9916318
#Area under the curve:0.9699
#Precision: 0.9115385

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.9916318
#Area under the curve:0.9699
#Precision: 0.9115385


#(2)
#si presenta un diferente algoritmo in grado di gestire grandi dataset.
#usare come K una media, oppure il massimo o il minimo:
round(mean(c(log(n),p+1)),0)
#44
round(max(c(log(n),p+1)),0)
#82
round(min(c(log(n),p+1)),0)
#7
#Nello specifico caso funziona molto meglio il max:

#Media
nearest<-nn2(X, query =X , k = 44, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score, col=is_out+1)
a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[6]
#2.33
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[4]
#2.31789
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.9874477
#Area under the curve:  0.9746
#Precision: 0.9328063

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.9874477
#Area under the curve:  0.9746
#Precision: 0.9328063


#Max
nearest<-nn2(X, query =X , k = 82, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[8]
#2.3
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[4]
#2.28085  
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.9916318
#Area under the curve: 0.9688
#Precision: 0.908046

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.9916318
#Area under the curve: 0.9677
#Precision: 0.9045802

#Min
nearest<-nn2(X, query =X , k = 7, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[5]
#1.71
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=b #Uguale all' a)  1.70632
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.9916318
#Area under the curve: 0.9688
#Precision: 0.908046

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.9958159
#Area under the curve: 0.9709
#Precision: 0.9083969

#Best k=7
#Best AUC=0.9709
#Best split= 1.70632




# Arrhythmia - 66 outliers ---------------------
path="arrhythmia.mat"
data=readMat(path)
X=data$X
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
p=dim(X)[2]; n=dim(X)[1]
summary(dati)
togliere<-which(nearZeroVar(dati, saveMetrics = T)$zeroVar==TRUE)
dati=dati[,-togliere]
X=dati[,-ncol(dati)]
p=dim(X)[2]; n=dim(X)[1]

#(1)
#Usare un knn aggregato con la seguente scelta di parametri:
knn.agg<-KNN_AGG(X, k_min = min(log(n),p+1), k_max = max(log(n),p+1))
plot(knn.agg)

a <- rpart_split_finder(knn.agg, n)
plot(sort(knn.agg),col=is_out[order(knn.agg)]+1)
abline(h=a)
sort(a)
#7984937
best_split=sort(a)[7]
outliers_a<-ifelse(knn.agg>best_split,1,0)

b <-ctree_split_finder(knn.agg,n)  
plot(sort(knn.agg), col=is_out[order(knn.agg)]+1)
abline(h=b)
sort(b)
best_split=7984937
#Split manuale, mettiamo lo stesso di a
outliers_b<-ifelse(knn.agg>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.5151515
#Area under the curve: 0.7148
#Precision: 0.5074627

#(2)
#si presenta un diferente algoritmo in grado di gestire grandi dataset.
#usare come K una media, oppure il massimo o il minimo:
round(mean(c(log(n),p+1)),0)
#132
round(max(c(log(n),p+1)),0)
#258
round(min(c(log(n),p+1)),0)
#6
#Nello specifico caso funziona molto meglio il max:

#Media
nearest<-nn2(X, query =X , k = 132, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score, col=is_out+1)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[4]
#232
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=sort(b)
#232.2125 uguale a a
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.4545455
#Area under the curve:  0.7014
#Precision: 0.6

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.5151515
#Area under the curve:   0.7122
#Precision: 0.4927536

#Max
nearest<-nn2(X, query =X , k = 258, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[4]
#280
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=sort(b)
#255 
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.3636364
#Area under the curve: 0.6598
#Precision: 0.5853659

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.5
#Area under the curve: 0.7085
#Precision: 0.5076923

#Min
nearest<-nn2(X, query =X , k = 6, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
abline(h=160, col="blue")
best_split=160
#160 split scelto manualmente
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
best_split=b
#132
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.3787879 
#Area under the curve: 0.6596
#Precision: 0.5208333

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.6212121 
#Area under the curve: 0.7264
#Precision: 0.3867925

#Best k=6
#Best AUC=0.6212121
#Best split= 132

#Dati Scalati:
X_scale=scale(X)

#(1)
#Usare un knn aggregato con la seguente scelta di parametri:
knn.agg<-KNN_AGG(X_scale, k_min = min(log(n),p+1), k_max = max(log(n),p+1))

a <- rpart_split_finder(knn.agg, n)
plot(sort(knn.agg),col=is_out[order(knn.agg)]+1)
abline(h=a)
abline(h=750000,col="blue")
best_split=750000
#Split scelto manualmente
outliers_a<-ifelse(knn.agg>best_split,1,0)

b <-ctree_split_finder(knn.agg,n)  
plot(sort(knn.agg), col=is_out[order(knn.agg)]+1)
abline(h=b)
sort(b)
abline(h=750000,col="blue")
best_split=750000
#Split scelto manualmente
outliers_b<-ifelse(knn.agg>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.4242424
#Area under the curve: 0.6616
#Precision: 0.4179104

#(2)
#si presenta un diferente algoritmo in grado di gestire grandi dataset.
#usare come K una media, oppure il massimo o il minimo:
round(mean(c(log(n),p+1)),0)
#132
round(max(c(log(n),p+1)),0)
#258
round(min(c(log(n),p+1)),0)
#6
#Nello specifico caso funziona molto meglio il max:

#Media
nearest<-nn2(X_scale, query =X_scale , k = 132, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score, col=is_out+1)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[9]
#18
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=sort(b)
#17.25884
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.5606061
#Area under the curve: 0.7065
#Precision: 0.393617

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.6060606
#Area under the curve: 0.7149
#Precision: 0.3703704


#Max
nearest<-nn2(X_scale, query =X_scale , k = 258, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[4]
#20
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=sort(b)
#20
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.5151515
#Area under the curve: 0.6889
#Precision: 0.3908046

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.5909091
#Area under the curve: 0.7074
#Precision: 0.364486

#Min
nearest<-nn2(X_scale, query =X_scale , k = 6, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[5]
#12.4
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=b
#12.32811
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.5757576
#Area under the curve: 0.6998
#Precision: 0.3584906

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.5909091
#Area under the curve: 0.7074
#Precision: 0.364486

#Best k=132
#Best AUC= 0.7149
#Best split= 17.25884


#####
#Su dataset completo:
dati<-read.csv("C:/Users/Beatrice/Documents/CLAMSES/Statistical Learning/Project/Dataset_completi/arrhythmia.csv", sep=",", header=F)
is_out<-ifelse(dati$V280==3|dati$V280==4|dati$V280==5|dati$V280==7|dati$V280==8|dati$V280==9|
                 dati$V280==14|dati$V280== 15,1,0)
dati$V280<-is_out

fac<-c(2,22:27)
for (i in 1:length(fac)){
  dati[,i]<-as.factor(dati[,i])
}

#One hot encoding:
dati<-cbind(as.data.frame(model.matrix(~.-1,data=dati[,-ncol(dati)])),is_out)
X=dati[,1:(ncol(dati)-1)]
p=dim(X)[2]; n=dim(X)[1]

togliere<-which(nearZeroVar(dati, saveMetrics = T)$zeroVar==TRUE)
dati=dati[,-togliere]
X=dati[,-ncol(dati)]
p=dim(X)[2]; n=dim(X)[1]

#(1)
#Usare un knn aggregato con la seguente scelta di parametri:
knn.agg<-KNN_AGG(X, k_min = min(log(n),p+1), k_max = n-1)

a <- rpart_split_finder(knn.agg, n)
plot(sort(knn.agg),col=is_out[order(knn.agg)]+1)
abline(h=a)
sort(a)
#2.2e+07
best_split=sort(a)[2]
outliers_a<-ifelse(knn.agg>best_split,1,0)

b <-ctree_split_finder(knn.agg,n)  
plot(sort(knn.agg), col=is_out[order(knn.agg)]+1)
abline(h=b)
sort(b)
best_split=b
#22226794
outliers_b<-ifelse(knn.agg>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.6515152
#Area under the curve: 0.7429
#Precision: 0.4018692

#(2)
#si presenta un diferente algoritmo in grado di gestire grandi dataset.
#usare come K una media, oppure il massimo o il minimo: 
#In questo caso vistoche le misure eccedono il numero di n usiamo come k=min, k=n-1
n-1
#451
round(min(c(log(n),p+1)),0)
#6


#n-1
nearest<-nn2(X, query =X , k = 451, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score, col=is_out+1)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[4]
#295
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=295
#295 uguale ad a)
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.3636364
#Area under the curve:  0.6546
#Precision: 0.5333333

#Min
nearest<-nn2(X, query =X , k = 6, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[4]
#133
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
best_split=b
#132.8717
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.5151515 
#Area under the curve: 0.7058
#Precision: 0.4594595

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.5151515 
#Area under the curve: 0.7058
#Precision: 0.4594595

#k defaul=10
nearest<-nn2(X, query =X , k = 10, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[3]
#151
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
best_split=b
#150.55
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.6666667 
#Area under the curve: 0.7375
#Precision: 0.3728814

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.5 
#Area under the curve: 0.6982
#Precision: 0.4520548

#Best k=6
#Best AUC=0.7058
#Best split= 133

#Dati Scalati:
X_scale=scale(X)

#(1)
#Usare un knn aggregato con la seguente scelta di parametri:
knn.agg<-KNN_AGG(X_scale, k_min = min(log(n),p+1), k_max = n-1)

a <- rpart_split_finder(knn.agg, n)
plot(sort(knn.agg),col=is_out[order(knn.agg)]+1)
abline(h=a)
abline(h=5000000,col="blue")
best_split=5000000
#Split scelto manualmente
outliers_a<-ifelse(knn.agg>best_split,1,0)

b <-ctree_split_finder(knn.agg,n)  
plot(sort(knn.agg), col=is_out[order(knn.agg)]+1)
abline(h=b)
sort(b)
best_split=b
#Split scelto manualmente
outliers_b<-ifelse(knn.agg>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.5757576
#Area under the curve: 0.6868
#Precision: 0.3275862

#(2)
#Inquesto caso Media e MAx sono troppo grandi allora usiamo solo il min e n-1:
n-1
#451
round(min(c(log(n),p+1)),0)
#6

#n-1
nearest<-nn2(X_scale, query =X_scale , k = 451, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score, col=is_out+1)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
abline(h=56,col="blue")
best_split=56
#56, spli scelto manualmente
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
abline(h=56,col="blue")
best_split=56
#56, spli scelto manualmente
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.3787879
#Area under the curve: 0.635
#Precision: 0.3731343


#Min
nearest<-nn2(X_scale, query =X_scale , k = 6, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
abline(h=41,col="blue")
best_split=41
#Split scelto manualmente
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
abline(h=41,col="blue")
best_split=41
#Split scelto manualmente
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.3484848
#Area under the curve: 0.6276
#Precision: 0.3898305

#k=10 di dafault
nearest<-nn2(X_scale, query =X_scale , k = 10, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
abline(h=45,col="blue")
best_split=45
#45 split scelto manualmente
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
abline(h=45,col="blue")
best_split=45
#45 split scelto manualmente
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.3484848
#Area under the curve: 0.6302
#Precision: 0.4035088


#Best k=451
#Best AUC= 0.635
#Best split= 56
#Il risultato è pessimo e ambiguo



# Ionosphere - 126 outliers --------------------
path="ionosphere.mat"
data=readMat(path)
X=data$X
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
summary(dati)
p=dim(X)[2]; n=dim(X)[1]


#(1)
#Usare un knn aggregato con la seguente scelta di parametri:
knn.agg<-KNN_AGG(X, k_min = min(log(n),p+1), k_max = max(log(n),p+1))
plot(knn.agg)

a <- rpart_split_finder(knn.agg, n)
plot(sort(knn.agg),col=is_out[order(knn.agg)]+1)
abline(h=a)
abline(h=1300,col="blue")
#Scegliamomanualmente lo split
#1300
best_split=1300
#Il solo taglio preso in considerazione è quello scelto manualmente che non coincide con gli altri
#suggeriti da entrambi
outliers_a<-ifelse(knn.agg>best_split,1,0)

b <-ctree_split_finder(knn.agg,n)  
plot(sort(knn.agg), col=is_out[order(knn.agg)]+1)
abline(h=b)
abline(h=1300,col="blue")
#Scegliamomanualmente lo split
#1300(è uguale ad a)
best_split=1300  
outliers_b<-ifelse(knn.agg>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.6904762
#Area under the curve: 0.8341
#Precision: 0.9456522


#(2)
#si presenta un diferente algoritmo in grado di gestire grandi dataset.
#usare come K una media, oppure il massimo o il minimo:
round(mean(c(log(n),p+1)),0)
#20
round(max(c(log(n),p+1)),0)
#34
round(min(c(log(n),p+1)),0)
#6
#Nello specifico caso funziona molto meglio il max:

#Media
nearest<-nn2(X, query =X , k = 20, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score, col=is_out+1)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
abline(h=1.9,col="blue")
best_split=1.9
#1.9 split scelto manualmente
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
abline(h=1.9,col="blue")
best_split=1.9
#1.9 split scelto manualmente
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.7698413
#Area under the curve:  0.8605
#Precision: 0.8981481


#Max
nearest<-nn2(X, query =X , k = 34, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
abline(h=2.3, col="blue")
best_split=2.3
#2.3, split scelto manualmente!
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
abline(h=2.3, col="blue")
best_split=2.3
#2.3, split scelto manualmente! uguale ad a)
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.6984127
#Area under the curve: 0.8292
#Precision: 0.9072165


#Min
nearest<-nn2(X, query =X , k = 6, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
abline(h=1.35, col="blue")
best_split=1.35
#1.35 scelto manualmente
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
abline(h=1.35, col="blue")
best_split=1.35
#1.35 scelto manualmente ugual ad a)
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.8333333 
#Area under the curve: 0.8989
#Precision: 0.9292035

#Best k=6
#Best AUC=0.8989
#Best split= 1.35


# Mnist - 700 outliers -------------------------
path="mnist.mat"
#inserire path
data=readMat(path)
X=data$X
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
p=dim(X)[2]; n=dim(X)[1]
togliere<-which(nearZeroVar(dati, saveMetrics = T)$zeroVar==TRUE)
dati=dati[,-togliere]
X=dati[,-ncol(dati)]
p=dim(X)[2]; n=dim(X)[1]
summary(X)


#(1)
#Usare un knn aggregato con la seguente scelta di parametri:
knn.agg<-KNN_AGG(X, k_min = min(log(n),p+1), k_max = max(log(n),p+1))
plot(knn.agg)

a <- rpart_split_finder(knn.agg, n)
plot(sort(knn.agg),col=is_out[order(knn.agg)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[4]
#1885737
outliers_a<-ifelse(knn.agg>best_split,1,0)

b <-ctree_split_finder(knn.agg,n)  
plot(sort(knn.agg), col=is_out[order(knn.agg)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[3]
#1885606   (è uguale ad a)
outliers_b<-ifelse(knn.agg>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.3928571
#Area under the curve: 0.6733
#Precision: 0.462963

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.3928571
#Area under the curve: 0.6733
#Precision: 0.462963


#(2)
#si presenta un diferente algoritmo in grado di gestire grandi dataset.
#usare come K una media, oppure il massimo o il minimo:
round(mean(c(log(n),p+1)),0)
#44
round(max(c(log(n),p+1)),0)
#79
round(min(c(log(n),p+1)),0)
#9
#Nello specifico caso funziona molto meglio il max:

#Media
nearest<-nn2(X, query =X , k = 44, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score, col=is_out+1)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[4]
#596
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[3]
#596.1953 uguale a a
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.3928571
#Area under the curve:  0.6727
#Precision: 0.4568106

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.3928571
#Area under the curve:  0.6728
#Precision: 0.4575707

#Max
nearest<-nn2(X, query =X , k = 79, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[3]
#634
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[3]
#633.8566 
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.3871429
#Area under the curve: 0.6714
#Precision: 0.4696707

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.3871429
#Area under the curve: 0.6714
#Precision: 0.4696707

#Min
nearest<-nn2(X, query =X , k = 9, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[5]
#499
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[3]
#499.9310 Uguale al caso a)
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.2628571 
#Area under the curve: 0.6153
#Precision: 0.4520885

#Best k=44
#Best AUC=0.6728
#Best split= 596

#Dati Scalati:
X_scale=scale(X)

#(1)
#Usare un knn aggregato con la seguente scelta di parametri:
knn.agg<-KNN_AGG(X_scale, k_min = min(log(n),p+1), k_max = max(log(n),p+1))
plot(knn.agg)

a <- rpart_split_finder(knn.agg, n)
plot(sort(knn.agg),col=is_out[order(knn.agg)]+1)
abline(h=a)
abline(h=30000, col="blue")
sort(a)
best_split=30000
#30000 scelto manualmente
#Il solo taglio preso in considerazione è quello scelto manualmente che non coincide con gli altri
#suggeriti da entrambi
outliers_a<-ifelse(knn.agg>best_split,1,0)

b <-ctree_split_finder(knn.agg,n)  
plot(sort(knn.agg), col=is_out[order(knn.agg)]+1)
abline(h=30000, col="blue")
best_split=30000 #uguale ad a)
#30000 scelto manualmente
outliers_b<-ifelse(knn.agg>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.2085714
#Area under the curve:0.5949
#Precision: 0.5289855


#(2)
#si presenta un diferente algoritmo in grado di gestire grandi dataset.
#usare come K una media, oppure il massimo o il minimo:
round(mean(c(log(n),p+1)),0)
#44
round(max(c(log(n),p+1)),0)
#79
round(min(c(log(n),p+1)),0)
#9
#Nello specifico caso funziona molto meglio il max:

#Media
nearest<-nn2(X_scale, query =X_scale , k = 44, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score, col=is_out+1)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
abline(h=10, col="blue")
best_split=10
#10 scelto manualmente
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
abline(h=10, col="blue")
best_split=10
#10 scelto manualmente uguale ad a)
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.1842857
#Area under the curve: 0.585
#Precision: 0.5657895


#Max
nearest<-nn2(X_scale, query =X_scale , k = 79, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[4]
#8.8 
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[2]
#7.58052 
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.2857143
#Area under the curve: 0.6279
#Precision: 0.4926108

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.4557143
#Area under the curve: 0.6947
#Precision: 0.4105534

#Min
nearest<-nn2(X_scale, query =X_scale , k = 9, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
#7.8
best_split=sort(a)[4]
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=7.8 #Uguale all' a)
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.1614286
#Area under the curve: 0.5754
#Precision: 0.6075269

#Best k=79
#Best AUC=0.6947
#Best split= 7.58052


# Optdigits - 150 outliers ---------------------
path="optdigits.mat"
data=readMat(path)
X=data$X
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
p=dim(X)[2]; n=dim(X)[1]
summary(dati)
togliere<-which(nearZeroVar(dati, saveMetrics = T)$zeroVar==TRUE)
dati=dati[,-togliere]
X=dati[,-ncol(dati)]
p=dim(X)[2]; n=dim(X)[1]
summary(X)

#(1)
#Usare un knn aggregato con la seguente scelta di parametri:
knn.agg<-KNN_AGG(X, k_min = min(log(n),p+1), k_max = max(log(n),p+1))
plot(knn.agg)

a <- rpart_split_finder(knn.agg, n)
plot(sort(knn.agg),col=is_out[order(knn.agg)]+1)
abline(h=a)
abline(h=57000,col="blue")
#Scegliamomanualmente lo split
#57000
best_split=57000
#Il solo taglio preso in considerazione è quello scelto manualmente che non coincide con gli altri
#suggeriti da entrambi
outliers_a<-ifelse(knn.agg>best_split,1,0)

b <-ctree_split_finder(knn.agg,n)  
plot(sort(knn.agg), col=is_out[order(knn.agg)]+1)
abline(h=b)
abline(h=57000,col="blue")
#Scegliamomanualmente lo split
#57000
best_split=57000
outliers_b<-ifelse(knn.agg>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.03333333
#Area under the curve: 0.502
#Precision: 0.03246753


#(2)
#si presenta un diferente algoritmo in grado di gestire grandi dataset.
#usare come K una media, oppure il massimo o il minimo:
round(mean(c(log(n),p+1)),0)
#36
round(max(c(log(n),p+1)),0)
#63
round(min(c(log(n),p+1)),0)
#9
#Nello specifico caso funziona molto meglio il max:

#Media
nearest<-nn2(X, query =X , k = 36, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score, col=is_out+1)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
abline(h=27,col="blue")
best_split=27
#28 split scelto manualmente
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
abline(h=27,col="blue")
best_split=27
#28 scelto manualmente uguale a a
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.06
#Area under the curve:  0.4992
#Precision: 0.4992


#Max
nearest<-nn2(X, query =X , k = 63, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
abline(h=30, col="blue")
best_split=30
#30 scelto manualemente
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
abline(h=30, col="blue")
best_split=30
#30 scelto manualemente uguale ad a 
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.04666667
#Area under the curve: 0.504
#Precision: 0.03448276


#Min
nearest<-nn2(X, query =X , k = 9, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
abline(h=22,col="blue")
best_split=22
#22 scelto manualmente
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  #No split

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.01333333 
#Area under the curve: 0.4858
#Precision: 0.009389671

#Best k= 63
#Best AUC=0.504
#Best split= 30
#Unico che super 0.5, questometodo è dannoso su questo dataset

#Dati Scalati:
X_scale=scale(X)

#(1)
#Usare un knn aggregato con la seguente scelta di parametri:
knn.agg<-KNN_AGG(X_scale, k_min = min(log(n),p+1), k_max = max(log(n),p+1))
plot(knn.agg)

a <- rpart_split_finder(knn.agg, n)
plot(sort(knn.agg),col=is_out[order(knn.agg)]+1)
abline(h=a)
abline(h=15000, col="blue")
best_split=15000
#15000
#Il solo taglio preso in considerazione è quello scelto manualmente che non coincide con gli altri
#suggeriti da entrambi
outliers_a<-ifelse(knn.agg>best_split,1,0)

b <-ctree_split_finder(knn.agg,n) #No split 

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0
#Area under the curve:0.4828
#Precision: 0


#(2)
#si presenta un diferente algoritmo in grado di gestire grandi dataset.
#usare come K una media, oppure il massimo o il minimo:
round(mean(c(log(n),p+1)),0)
#36
round(max(c(log(n),p+1)),0)
#63
round(min(c(log(n),p+1)),0)
#9
#Nello specifico caso funziona molto meglio il max:

#Media
nearest<-nn2(X_scale, query =X_scale , k = 36, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score, col=is_out+1)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
abline(h=7.5,col="blue")
best_split=7.5
#7.5 scelto manualmente
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  #No splits

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0
#Area under the curve: 0.4829
#Precision: 0



#Max
nearest<-nn2(X_scale, query =X_scale , k = 63, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
abline(h=7.5, col="blue")
best_split=7.5
#7.5 scelto manualmente
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  #No split

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0
#Area under the curve: 0.4752
#Precision: 0



#Min
nearest<-nn2(X_scale, query =X_scale , k = 9, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
abline(h=5.5, col="blue")
#5.5 scelto maualemtne
best_split=5.5
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  #Nosplit

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0
#Area under the curve: 0.4811
#Precision: 0

#Best k=-
#Best AUC=-
#Best split= -

#nessun k riesce ad individuare un solo outlier


#Http - 2211 outliers --------------------------
path <- "y_http.mat"
data=readMat(path)
is_out<-as.vector(data$y)
path <- "X_http.mat"
data=readMat(path)
X <- data$X
dati<-data.frame(cbind(X, is_out))
head(dati)
summary(dati)
p=dim(X)[2]; n=dim(X)[1]

#
# skippata per motivi di memoria!!!!!!!!!!!!!!1
#

#(1)
#Usare un knn aggregato con la seguente scelta di parametri:
knn.agg<-KNN_AGG(X, k_min = min(log(n),p+1), k_max = max(log(n),p+1))

a <- rpart_split_finder(knn.agg, n)
plot(sort(knn.agg),col=is_out[order(knn.agg)]+1)
abline(h=a)
abline(h=11500000,col="blue")
#Scegliamomanualmente lo split
#11500000
best_split=11500000
#Il solo taglio preso in considerazione è quello scelto manualmente che non coincide con gli altri
#suggeriti da entrambi
outliers_a<-ifelse(knn.agg>best_split,1,0)

b <-ctree_split_finder(knn.agg,n)  
plot(sort(knn.agg), col=is_out[order(knn.agg)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[4]
#11549932   (è uguale ad a)
outliers_b<-ifelse(knn.agg>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 1
#Area under the curve: 0.9788
#Precision: 0.4349776

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 1
#Area under the curve: 0.983
#Precision: 0.489899

#
#SALTATA!!!!!!!!!!!!!!!!!!!!!!!
#

#(2)
#si presenta un diferente algoritmo in grado di gestire grandi dataset.
#usare come K una media, oppure il massimo o il minimo:
round(mean(c(log(n),p+1)),0)
#9
round(max(c(log(n),p+1)),0)
#13
round(min(c(log(n),p+1)),0)
#4
#Nello specifico caso funziona molto meglio il max:

#Media
nearest<-nn2(X, query =X , k = 9, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[6]
#0.0573
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[4]
#0.05731 uguale a a
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.03708729
#Area under the curve:  0.5162
#Precision: 0.02960289

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.03708729
#Area under the curve:  0.5162
#Precision: 0.02960289

#Max
nearest<-nn2(X, query =X , k = 13, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[7]
#0.071000
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[5]
#0.07130 
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.03844414
#Area under the curve: 0.5168
#Precision: 0.03010981

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.03844414
#Area under the curve: 0.5168
#Precision: 0.03010981

#Min
nearest<-nn2(X, query =X , k = 4, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[6]
#0.01
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[5]
#0.01045 Uguale al caso a)
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.03482587 
#Area under the curve: 0.5124
#Precision: 0.01348275

#K=200, proviamo qusto k data la numerosità del dataset
nearest<-nn2(X, query =X , k = 200, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[9]
#1.700000
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[6]
#0.26993 Uguale al caso a)
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.02035278 
#Area under the curve: 0.51
#Precision: 0.1704545

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.02035278 
#Area under the curve: 0.51
#Precision: 0.1704545

#Best k=13
#Best AUC=0.5168
#Best split= 0.07

#Dati Scalati:
X_scale=scale(X)


#
#sKIPPATA PER MOTIVI DI MEMORIA!!!!!!!!
#


#(1)
#Usare un knn aggregato con la seguente scelta di parametri:
knn.agg<-KNN_AGG(X_scale, k_min = min(log(n),p+1), k_max = max(log(n),p+1))
plot(knn.agg)

a <- rpart_split_finder(knn.agg, n)
plot(sort(knn.agg),col=is_out[order(knn.agg)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[6]
#173820
#Il solo taglio preso in considerazione è quello scelto manualmente che non coincide con gli altri
#suggeriti da entrambi
outliers_a<-ifelse(knn.agg>best_split,1,0)

b <-ctree_split_finder(knn.agg,n)  
plot(sort(knn.agg), col=is_out[order(knn.agg)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[4]
#173698.4
outliers_b<-ifelse(knn.agg>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 1
#Area under the curve:0.9992
#Precision: 0.9509804

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 1
#Area under the curve: 0.999
#Precision: 0.9417476

#
#SALTATA!!!!!!!!!!!!!!!!!!!!!!!!!
#


#(2)
#si presenta un diferente algoritmo in grado di gestire grandi dataset.
#usare come K una media, oppure il massimo o il minimo:
round(mean(c(log(n),p+1)),0)
#9
round(max(c(log(n),p+1)),0)
#13
round(min(c(log(n),p+1)),0)
#4
#Nello specifico caso funziona molto meglio il max:

#Media
nearest<-nn2(X_scale, query =X_scale , k = 9, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[7]
#0.074000
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[6]
#0.07416 uguale a a
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.03934871
#Area under the curve: 0.517
#Precision: 0.02791145

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.03934871
#Area under the curve: 0.517
#Precision: 0.02793834


#Max
nearest<-nn2(X_scale, query =X_scale , k = 13, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[7]
#0.660000
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[6]
#0.65714  
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.01718679
#Area under the curve: 0.5085
#Precision: 0.2289157

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.01718679
#Area under the curve: 0.5085
#Precision: 0.2289157

#Min
nearest<-nn2(X_scale, query =X_scale , k = 4, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[6]
#2.3e-02
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
#0.02313
best_split=sort(b)[5]
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.03482587
#Area under the curve: 0.514
#Precision: 0.01946903

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.03482587 
#Area under the curve: 0.514
#Precision: 0.01946903

#Best k=9
#Best AUC=0.517
#Best split= 0.074000


# ForestCover - 2747 outliers ------------------
path="cover.mat"
data=readMat(path)
X=data$X
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
p=dim(X)[2]; n=dim(X)[1]


#(1)
#Usare un knn aggregato con la seguente scelta di parametri:
knn.agg<-KNN_AGG(X, k_min = min(log(n),p+1), k_max = max(log(n),p+1))

a <- rpart_split_finder(knn.agg, n)
plot(sort(knn.agg),col=is_out[order(knn.agg)]+1)
abline(h=a)
sort(a)
#2276
best_split=sort(a)[7]
outliers_a<-ifelse(knn.agg>best_split,1,0)

b <-ctree_split_finder(knn.agg,n)  
plot(sort(knn.agg), col=is_out[order(knn.agg)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[6]
#1769.2674
outliers_b<-ifelse(knn.agg>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.2373498
#Area under the curve: 0.6108
#Precision: 0.1270707

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.4907171
#Area under the curve:  0.711
#Precision: 0.06467709


#(2)
#si presenta un diferente algoritmo in grado di gestire grandi dataset.
#usare come K una media, oppure il massimo o il minimo:
round(mean(c(log(n),p+1)),0)
#12
round(max(c(log(n),p+1)),0)
#13
round(min(c(log(n),p+1)),0)
#11
#Nello specifico caso funziona molto meglio il max:

#Media
nearest<-nn2(X, query =X , k = 12, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[6]
#70
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[6]
#69.76482 uguale a a
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.4790681
#Area under the curve:  0.7072
#Precision: 0.06702317

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.4845286
#Area under the curve:  0.7092
#Precision: 0.06639066

#Max
nearest<-nn2(X, query =X , k = 13, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[6]
#76
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[6]
#75.55673 
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.43502
#Area under the curve: 0.6928
#Precision: 0.07858223

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.4426647
#Area under the curve: 0.6957
#Precision: 0.07728977

#Min
nearest<-nn2(X, query =X , k = 11, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[7]
#72
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[7]
#71.75375
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.402621 
#Area under the curve: 0.6804
#Precision: 0.08521458

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.4073535 
#Area under the curve: 0.6823
#Precision: 0.08469573

#Best k=13
#Best AUC=0.6928
#Best split= 76

#Dati Scalati:
X_scale=scale(X)

#(1)
#Usare un knn aggregato con la seguente scelta di parametri:
knn.agg<-KNN_AGG(X_scale, k_min = min(log(n),p+1), k_max = max(log(n),p+1))

a <- rpart_split_finder(knn.agg, n)
plot(sort(knn.agg),col=is_out[order(knn.agg)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[8]
#22.7
outliers_a<-ifelse(knn.agg>best_split,1,0)

b <-ctree_split_finder(knn.agg,n)  
plot(sort(knn.agg), col=is_out[order(knn.agg)]+1)
abline(h=b)
sort(b)
best_split=20
#Split scelto manualmente
outliers_b<-ifelse(knn.agg>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.05314889
#Area under the curve:0.5242
#Precision: 0.09805238

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.09428467
#Area under the curve: 0.5423
#Precision: 0.08556326


#(2)
#si presenta un diferente algoritmo in grado di gestire grandi dataset.
#usare come K una media, oppure il massimo o il minimo:
round(mean(c(log(n),p+1)),0)
#12
round(max(c(log(n),p+1)),0)
#13
round(min(c(log(n),p+1)),0)
#11
#Nello specifico caso funziona molto meglio il max:

#Media
nearest<-nn2(X_scale, query =X_scale , k = 12, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[7]
#0.9
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=0.8
#0.8, split selezionato manualmentr
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.05169276
#Area under the curve: 0.5236
#Precision: 0.09867964

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.08663997
#Area under the curve: 0.5388
#Precision: 0.08533525


#Max
nearest<-nn2(X_scale, query =X_scale , k = 13, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=0.7
#0.7 split scelto manualmente
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=0.7
#0.7 split scelto manualmente uguale ad a)
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.1714598
#Area under the curve: 0.5741
#Precision: 0.0667517


#Min
nearest<-nn2(X_scale, query =X_scale , k = 11, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[6]
#0.87
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=0.8
#split sceltomanualmente
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.05278486
#Area under the curve: 0.524
#Precision: 0.09757739

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.0771751
#Area under the curve: 0.5348
#Precision: 0.08937605

#Best k=13
#Best AUC=0.5741
#Best split= 0.7


#####
#Su dataset completo:
dati=read.table("C:/Users/Beatrice/Documents/CLAMSES/Statistical Learning/Project/Dataset_completi/covtype.data", sep=",", header=F)
dati<-dati[which(dati$V55==2|dati$V55==4),]
is_out<-ifelse(dati$V55==4,1,0)
dati<-dati[,-55]
dati$is_out<-is_out
head(dati)

#fac<-c(11:54)
#for (i in 1:length(fac)){
#  dati[,i]<-as.factor(dati[,i])
#}

togliere<-which(nearZeroVar(dati, saveMetrics = T)$zeroVar==TRUE)
dati=dati[,-togliere]
X=dati[,-ncol(dati)]
p=dim(X)[2]; n=dim(X)[1]

#One hot encoding:
#dati<-cbind(as.data.frame(model.matrix(~.-1,data=dati[,-ncol(dati)])),is_out)
#X=dati[,1:(ncol(dati)-1)]

#(1)
#Usare un knn aggregato con la seguente scelta di parametri:

#
#--skippato per problemi di memoria--
#

knn.agg<-KNN_AGG(X, k_min = min(log(n),p+1), k_max = max(log(n),p+1))   

a <- rpart_split_finder(knn.agg, n)
plot(sort(knn.agg),col=is_out[order(knn.agg)]+1)
abline(h=a)
abline(h=11500000,col="blue")
#Scegliamomanualmente lo split
#11500000
best_split=11500000
#Il solo taglio preso in considerazione è quello scelto manualmente che non coincide con gli altri
#suggeriti da entrambi
outliers_a<-ifelse(knn.agg>best_split,1,0)

b <-ctree_split_finder(knn.agg,n)  
plot(sort(knn.agg), col=is_out[order(knn.agg)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[4]
#11549932   (è uguale ad a)
outliers_b<-ifelse(knn.agg>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 1
#Area under the curve: 0.9788
#Precision: 0.4349776

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 1
#Area under the curve: 0.983
#Precision: 0.489899

#
#Staltata!!!!!!!!!!!!
#

#(2)
#si presenta un diferente algoritmo in grado di gestire grandi dataset.
#usare come K una media, oppure il massimo o il minimo:
round(mean(c(log(n),p+1)),0)
#33
round(max(c(log(n),p+1)),0)
#53
round(min(c(log(n),p+1)),0)
#13
#Nello specifico caso funziona molto meglio il max:

#Media
nearest<-nn2(X, query =X , k = 33, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=85
#85 split scelto manualmente
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=85
#85 split scelto manualmente uguale a a
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.3447397
#Area under the curve:  0.6397
#Precision: 0.0487165


#Max
nearest<-nn2(X, query =X , k = 53, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=100
#100 split scelto manualmente!
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=100
#100 split scelto manualmente!  uguale ad a)
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall:0.3738624
#Area under the curve: 0.6468
#Precision: 0.04321481


#Min
nearest<-nn2(X, query =X , k = 13, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=60
#60 split scelto manualmente
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=60
#60 split scelto manualmente come nel caso a)
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.282854 
#Area under the curve: 0.6188
#Precision: 0.05706941

#Best k=53
#Best AUC=0.6468
#Best split= 100

#Dati Scalati:
X_scale=scale(X)

#
#Skippato per motivi di memoria!!!!!!!!
#

#(1)
#Usare un knn aggregato con la seguente scelta di parametri:
knn.agg<-KNN_AGG(X_scale, k_min = min(log(n),p+1), k_max = max(log(n),p+1))
plot(knn.agg)

a <- rpart_split_finder(knn.agg, n)
plot(sort(knn.agg),col=is_out[order(knn.agg)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[6]
#173820
#Il solo taglio preso in considerazione è quello scelto manualmente che non coincide con gli altri
#suggeriti da entrambi
outliers_a<-ifelse(knn.agg>best_split,1,0)

b <-ctree_split_finder(knn.agg,n)  
plot(sort(knn.agg), col=is_out[order(knn.agg)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[4]
#173698.4
outliers_b<-ifelse(knn.agg>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 1
#Area under the curve:0.9992
#Precision: 0.9509804

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 1
#Area under the curve: 0.999
#Precision: 0.9417476

#
#Saltata!!!!!!!!!!!!!!!!!!!!
#


#(2)
#si presenta un diferente algoritmo in grado di gestire grandi dataset.
#usare come K una media, oppure il massimo o il minimo:
round(mean(c(log(n),p+1)),0)
#33
round(max(c(log(n),p+1)),0)
#53
round(min(c(log(n),p+1)),0)
#13
#Nello specifico caso funziona molto meglio il max:

#Media
nearest<-nn2(X_scale, query =X_scale , k = 33, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[7]
#0.9
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[7]
#0.89745 uguale a a
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.4521296
#Area under the curve: 0.6826
#Precision: 0.04799073

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.4543138
#Area under the curve: 0.6832
#Precision: 0.04771554


#Max
nearest<-nn2(X_scale, query =X_scale , k = 53, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[7]
#1.10
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[7]
#1.10136  
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.4503094
#Area under the curve: 0.6886
#Precision: 0.05642991

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.4492173
#Area under the curve: 0.6883
#Precision: 0.05655362

#Min
nearest<-nn2(X_scale, query =X_scale , k = 13, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=0.8
#0.8 split scelto manualmente!
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[6] #Uguale all' a)
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.2475428
#Area under the curve: 0.6066
#Precision: 0.06537204

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.2948671
#Area under the curve: 0.6248
#Precision: 0.05930156

#Best k=53
#Best AUC=0.6886
#Best split= 1.10


# Smtp (KDDCUP99) - 30 outliers ----------------
path="smtp1.mat"
data=readMat(path)
X=data$X
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
summary(dati)
p=dim(X)[2]; n=dim(X)[1]
summary(X)

#(1)
#Usare un knn aggregato con la seguente scelta di parametri:
knn.agg<-KNN_AGG(X, k_min = min(log(n),p+1), k_max = max(log(n),p+1))
plot(knn.agg)

a <- rpart_split_finder(knn.agg, n)
plot(sort(knn.agg),col=is_out[order(knn.agg)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[7]
#9.2
outliers_a<-ifelse(knn.agg>best_split,1,0)

b <-ctree_split_finder(knn.agg,n)  #No Splits!

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.6666667
#Area under the curve: 0.8303
#Precision: 0.0331675

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 1
#Area under the curve: 0.983
#Precision: 0.489899


#(2)
#si presenta un diferente algoritmo in grado di gestire grandi dataset.
#usare come K una media, oppure il massimo o il minimo:
round(mean(c(log(n),p+1)),0)
#8
round(max(c(log(n),p+1)),0)
#11
round(min(c(log(n),p+1)),0)
#4
#Proviamo anche uno molto alto vista l'elevata numerosità del dataset
#k=2000
#Nello specifico caso funziona molto meglio il max:

#Media
nearest<-nn2(X, query =X , k = 8, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score, col=is_out+1)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[7]
#0.12292
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n) #No split!

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.6666667
#Area under the curve:  0.83
#Precision: 0.03021148

#Max
nearest<-nn2(X, query =X , k = 9, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[7]
#1.3e-01
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  #No Split!!

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.6666667
#Area under the curve: 0.8299
#Precision: 0.02998501

#Min
nearest<-nn2(X, query =X , k = 4, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
abline(h=0.05,col="blue")
best_split=0.05
#0.05 split scelto manualmente
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  #No Split

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.4666667 
#Area under the curve: 0.7264
#Precision: 0.01053424

#k=1000
nearest<-nn2(X, query =X , k = 1000, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[6]
#1.053
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  #No Split

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.6666667 
#Area under the curve: 0.8268
#Precision: 0.01576044

#Best k=8
#Best AUC=0.83
#Best split= 0.12292

#Dati Scalati:
X_scale=scale(X)

#(1)
#Usare un knn aggregato con la seguente scelta di parametri:
knn.agg<-KNN_AGG(X_scale, k_min = min(log(n),p+1), k_max = max(log(n),p+1))
plot(knn.agg)

a <- rpart_split_finder(knn.agg, n)
plot(sort(knn.agg),col=is_out[order(knn.agg)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[7]
#20
#Il solo taglio preso in considerazione è quello scelto manualmente che non coincide con gli altri
#suggeriti da entrambi
outliers_a<-ifelse(knn.agg>best_split,1,0)

b <-ctree_split_finder(knn.agg,n) #Nosplit!

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.6666667
#Area under the curve:0.8318
#Precision: 0.0625

#(2)
#si presenta un diferente algoritmo in grado di gestire grandi dataset.
#usare come K una media, oppure il massimo o il minimo:
round(mean(c(log(n),p+1)),0)
#8
round(max(c(log(n),p+1)),0)
#11
round(min(c(log(n),p+1)),0)
#4
#Nello specifico caso funziona molto meglio il max:

#Media
nearest<-nn2(X_scale, query =X_scale , k = 8, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score, col=is_out+1)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[5]
#0.2823
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n) #No Split!

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.6666667
#Area under the curve: 0.8318
#Precision: 0.06389776


#Max
nearest<-nn2(X_scale, query =X_scale , k = 11, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[8]
#0.32238
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  #No Splits!

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.6666667
#Area under the curve: 0.8317
#Precision: 0.06024096

#Min
nearest<-nn2(X_scale, query =X_scale , k = 4, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
#0.20168
best_split=sort(a)[6]
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[4] #0.20165 a)
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.2666667
#Area under the curve: 0.632
#Precision: 0.03125

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.2666667
#Area under the curve: 0.632
#Precision: 0.03125

#Best k=8
#Best AUC=0.8318
#Best split= 0.2823



# Mammography - 260 outliers -------------------
path="mammography.mat"
data=readMat(path)
X=data$X
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
head(dati)
summary(dati)

p=dim(X)[2]; n=dim(X)[1]

#(1)
#Usare un knn aggregato con la seguente scelta di parametri:
knn.agg<-KNN_AGG(X, k_min = min(log(n),p+1), k_max = max(log(n),p+1))
plot(knn.agg)

a <- rpart_split_finder(knn.agg, n)
plot(sort(knn.agg),col=is_out[order(knn.agg)]+1)
abline(h=a)
abline(h=20,col="blue")
#Scegliamomanualmente lo split
#20
best_split=20
#Il solo taglio preso in considerazione è quello scelto manualmente che non coincide con gli altri
#suggeriti da entrambi
outliers_a<-ifelse(knn.agg>best_split,1,0)

b <-ctree_split_finder(knn.agg,n)#No Split!

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.2153846
#Area under the curve: 0.5986
#Precision: 0.2204724


#(2)
#si presenta un diferente algoritmo in grado di gestire grandi dataset.
#usare come K una media, oppure il massimo o il minimo:
round(mean(c(log(n),p+1)),0)
#8
round(max(c(log(n),p+1)),0)
#9
round(min(c(log(n),p+1)),0)
#7
#Nello specifico caso funziona molto meglio il max:

#Media
nearest<-nn2(X, query =X , k = 8, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score, col=is_out+1)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[4]
#0.52
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  #No Split!

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.4948454
#Area under the curve:  0.649
#Precision: 0.3356643

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.3423077
#Area under the curve:  0.7314
#Precision: 0.1555944

#Max
nearest<-nn2(X, query =X , k = 9, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
abline(h=0.8,col="blue")
best_split=0.8
#0.8
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  #No split!

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.1923077
#Area under the curve: 0.5894
#Precision: 0.2525253


#Min
nearest<-nn2(X, query =X , k = 7, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[4]
#0.53
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  #No split

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.2884615 
#Area under the curve: 0.6254
#Precision: 0.1540041

#Best k=8
#Best AUC=0.7314
#Best split= 0.52

#Dati Scalati:
X_scale=scale(X)

#(1)
#Usare un knn aggregato con la seguente scelta di parametri:
knn.agg<-KNN_AGG(X_scale, k_min = min(log(n),p+1), k_max = max(log(n),p+1))
plot(knn.agg)

a <- rpart_split_finder(knn.agg, n)
plot(sort(knn.agg),col=is_out[order(knn.agg)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[3]
#16
#Il solo taglio preso in considerazione è quello scelto manualmente che non coincide con gli altri
#suggeriti da entrambi
outliers_a<-ifelse(knn.agg>best_split,1,0)

b <-ctree_split_finder(knn.agg,n) #No Splits!

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.2961538
#Area under the curve:0.6304
#Precision: 0.1663067

#(2)
#si presenta un diferente algoritmo in grado di gestire grandi dataset.
#usare come K una media, oppure il massimo o il minimo:
round(mean(c(log(n),p+1)),0)
#8
round(max(c(log(n),p+1)),0)
#9
round(min(c(log(n),p+1)),0)
#7
#Nello specifico caso funziona molto meglio il max:

#Media
nearest<-nn2(X_scale, query =X_scale , k = 8, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score, col=is_out+1)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[4]
#0.52
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  #No Split!

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.3423077
#Area under the curve: 0.649
#Precision: 0.1555944


#Max
nearest<-nn2(X_scale, query =X_scale , k = 9, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
abline(h=0.8, col="blue")
best_split=0.8
#0.8 split scelto manualmente
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)   #No Split!

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.1923077
#Area under the curve: 0.5894
#Precision:0.2525253

#Min
nearest<-nn2(X_scale, query =X_scale , k = 7, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[4]
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)   #No Split! 

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.2884615
#Area under the curve: 0.6254
#Precision: 0.1540041

#Best k=8
#Best AUC=0.649
#Best split= 0.52


# Annthyroid - 534 outliers --------------------
path <-"annthyroid.mat"
data=readMat(path)
X=data$X
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
head(dati)
summary(dati)

p=dim(X)[2]; n=dim(X)[1]

#(1)
#Usare un knn aggregato con la seguente scelta di parametri:
#Invece che mettere p+1 mettiamo p*2 perchè se no esce lo stesso valore
knn.agg<-KNN_AGG(X, k_min = min(log(n),p+1), k_max = max(log(n),p*2))
plot(knn.agg)

a <- rpart_split_finder(knn.agg, n)
plot(sort(knn.agg),col=is_out[order(knn.agg)]+1)
abline(h=a)
abline(h=2.5,col="blue")
#Scegliamomanualmente lo split
#2.5
best_split=2.5
#Il solo taglio preso in considerazione è quello scelto manualmente che non coincide con gli altri
#suggeriti da entrambi
outliers_a<-ifelse(knn.agg>best_split,1,0)

b <-ctree_split_finder(knn.agg,n)  
plot(sort(knn.agg), col=is_out[order(knn.agg)]+1)
abline(h=b)
abline(h=2.5,col="blue")
#Scegliamomanualmente lo split
#2.5
best_split=2.5 #(è uguale ad a)
outliers_b<-ifelse(knn.agg>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.1535581
#Area under the curve: 0.565
#Precision: 0.3430962

#(2)
#si presenta un diferente algoritmo in grado di gestire grandi dataset.
#usare come K una media, oppure il massimo o il minimo:
round(mean(c(log(n),p+1)),0)
#8
round(max(c(log(n),p+1)),0)
#9
round(min(c(log(n),p+1)),0)
#7
#Nello specifico caso funziona molto meglio il max:

#Media
nearest<-nn2(X, query =X , k = 8, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
abline(h=0.03, col="blue")
best_split=0.03
#0.03 split scelto manualmente
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
abline(h=0.03, col="blue")
best_split=0.03
#0.03 split scelto manualmente uguale ad a)
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.2265918
#Area under the curve:  0.5947
#Precision: 0.3279133

#Max
nearest<-nn2(X, query =X , k = 9, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
abline(h=0.03, col="blue")
best_split=0.03
#0.03 split scelto manualmente 
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
abline(h=0.03, col="blue")
best_split=0.03
#0.03 split scelto manualmente uguale ad a)
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.2453184
#Area under the curve: 0.6012
#Precision: 0.3141487

#Min
nearest<-nn2(X, query =X , k = 7, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
abline(h=0.03, col="blue")
best_split=0.03
#0.03 split scelto manualmente 
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
abline(h=0.03, col="blue")
best_split=0.03
#0.03 split scelto manualmente uguale ad a)
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.2003745 
#Area under the curve: 0.5835
#Precision: 0.3242424

#Best k=9
#Best AUC=0.6012
#Best split= 0.03

#Dati Scalati:
X_scale=scale(X)

#(1)
#Usare un knn aggregato con la seguente scelta di parametri:
knn.agg<-KNN_AGG(X_scale, k_min = min(log(n),p+1), k_max = max(log(n),p+1))

a <- rpart_split_finder(knn.agg, n)
plot(sort(knn.agg),col=is_out[order(knn.agg)]+1)
abline(h=a)
abline(h=20,col="blue")
best_split=20
#20
#Il solo taglio preso in considerazione è quello scelto manualmente che non coincide con gli altri
#suggeriti da entrambi
outliers_a<-ifelse(knn.agg>best_split,1,0)

b <-ctree_split_finder(knn.agg,n)  #No splits!

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.1048689
#Area under the curve:0.5422
#Precision: 0.2916667


#(2)
#si presenta un diferente algoritmo in grado di gestire grandi dataset.
#usare come K una media, oppure il massimo o il minimo:
round(mean(c(log(n),p+1)),0)
#8
round(max(c(log(n),p+1)),0)
#9
round(min(c(log(n),p+1)),0)
#7
#Nello specifico caso funziona molto meglio il max:

#Media
nearest<-nn2(X_scale, query =X_scale , k = 8, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score, col=is_out+1)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
abline(h=1,col="blue")
best_split=1
#1 split scelto manualmente
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  #No splits

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.164794
#Area under the curve: 0.5686
#Precision: 0.3235294

#Max
nearest<-nn2(X_scale, query =X_scale , k = 9, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
abline(h=1,col="blue")
best_split=1
#1 split scelto manualmente
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  #No split!

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.1853933
#Area under the curve: 0.5771
#Precision: 0.3224756

#Min
nearest<-nn2(X_scale, query =X_scale , k = 7, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
abline(h=1,col="blue")
best_split=1
#1 split scelto manualmente
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  #No Split!

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.1460674
#Area under the curve: 0.5609
#Precision: 0.325

#Best k=9
#Best AUC=0.5771
#Best split= 1

####----
train <- read.csv("ann_train.csv", sep=" ", header=F)
test <- read.csv("ann_test.csv", sep=" ", header=F)
dati_csv <- rbind(train, test)[-c(23,24)]
summary(dati_csv)

is_out<- dati_csv$V22
is_out<-ifelse(is_out==1|is_out==2,1,0)
dati_csv$V22<-is_out
dati <- dati_csv

fac <- c(2:16)
for (i in 1:length(fac)){
  dati[,i]<-as.factor(dati[,i])
}

#One hot encoding:
dati<-cbind(as.data.frame(model.matrix(~.-1,data=dati[,-ncol(dati)])),is_out)
X=dati[,1:(ncol(dati)-1)]
p=dim(X)[2]; n=dim(X)[1]

#(1)
#Usare un knn aggregato con la seguente scelta di parametri:
knn.agg<-KNN_AGG(X, k_min = min(log(n),p+1), k_max = max(log(n),p+1))

a <- rpart_split_finder(knn.agg, n)
plot(sort(knn.agg),col=is_out[order(knn.agg)]+1)
abline(h=a)
abline(h=9000,col="blue")
#Scegliamomanualmente lo split
#9000
best_split=9000
#Il solo taglio preso in considerazione è quello scelto manualmente che non coincide con gli altri
#suggeriti da entrambi
outliers_a<-ifelse(knn.agg>best_split,1,0)

b <-ctree_split_finder(knn.agg,n)  
plot(sort(knn.agg), col=is_out[order(knn.agg)]+1)
abline(h=b)
abline(h=9000,col="blue")
#Scegliamomanualmente lo split
#9000(è uguale ad a)
best_split=9000   
outliers_b<-ifelse(knn.agg>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.08052434
#Area under the curve: 0.4919
#Precision: 0.0625

#(2)
#si presenta un diferente algoritmo in grado di gestire grandi dataset.
#usare come K una media, oppure il massimo o il minimo:
round(mean(c(log(n),p+1)),0)
#64
round(max(c(log(n),p+1)),0)
#119
round(min(c(log(n),p+1)),0)
#8
#Nello specifico caso funziona molto meglio il max:

#Media
nearest<-nn2(X, query =X , k = 64, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score, col=is_out+1)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
abline(h=1.3, col="blue")
best_split=1.3
#1.3 split scelto manualmente
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
abline(h=1.3, col="blue")
best_split=1.3
#1.3 split scelto manualmente uguale ad a
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.05430712
#Area under the curve:  0.489
#Precision: 0.05400372


#Max
nearest<-nn2(X, query =X , k = 119, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
abline(h=1.3, col="blue")
best_split=1.3
#1.3 split scelto manualmente uguale ad a
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=1.3, col="blue")
best_split=1.3
#1.3 split scelto manualmente uguale ad a 
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.1048689
#Area under the curve: 0.4826
#Precision: 0.05673759


#Min
nearest<-nn2(X, query =X , k = 9, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
abline(h=0.65, col="blue")
best_split=0.65
#0.65 split scelto manualmente
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
abline(h=0.65, col="blue")
best_split=0.65
#0.65 split scelto manualmente
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.1872659 
#Area under the curve: 0.4802
#Precision: 0.06203474

#Best k=64
#Best AUC=0.489
#Best split= 1.3
#Funziona da schifo, nessuno super lo 0.49


#Dati Scalati:
X_scale=scale(X)

#(1)
#Usare un knn aggregato con la seguente scelta di parametri:
knn.agg<-KNN_AGG(X_scale, k_min = min(log(n),p+1), k_max = max(log(n),p+1))

a <- rpart_split_finder(knn.agg, n)
plot(sort(knn.agg),col=is_out[order(knn.agg)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[7]
#74942
#Il solo taglio preso in considerazione è quello scelto manualmente che non coincide con gli altri
#suggeriti da entrambi
outliers_a<-ifelse(knn.agg>best_split,1,0)

b <-ctree_split_finder(knn.agg,n)  
plot(sort(knn.agg), col=is_out[order(knn.agg)]+1)
abline(h=b)
sort(b)
best_split=74942
#74942 uguale ad a)
outliers_b<-ifelse(knn.agg>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall:0.082397
#Area under the curve:0.5115
#Precision: 0.1


#(2)
#si presenta un diferente algoritmo in grado di gestire grandi dataset.
#usare come K una media, oppure il massimo o il minimo:
round(mean(c(log(n),p+1)),0)
#64
round(max(c(log(n),p+1)),0)
#119
round(min(c(log(n),p+1)),0)
#9
#Nello specifico caso funziona molto meglio il max:

#Media
nearest<-nn2(X_scale, query =X_scale , k = 64, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score, col=is_out+1)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
abline(h=16,col="blue")
best_split=16
#16 split scelto manualmente
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
abline(h=16,col="blue")
best_split=16
#16 split scelto manualmente uguale a a
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.082397
#Area under the curve: 0.5115
#Precision: 0.1


#Max
nearest<-nn2(X_scale, query =X_scale , k = 119, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[6]
#11.4
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=11.4
#11.4 messo manualmente uguale ad a
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.08988764
#Area under the curve: 0.5125
#Precision:  0.1

#Min
nearest<-nn2(X_scale, query =X_scale , k = 9, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
abline(h=8, col="blue")
#8
best_split=8
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=8 #Uguale all' a)
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.04494382
#Area under the curve: 0.5074
#Precision: 0.1066667

#Best k=119
#Best AUC=0.5125
#Best split= 11.4

# Pendigits - 156 outliers ---------------------
path <-"pendigits.mat"
data=readMat(path)
X=data$X
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
head(dati)
summary(dati)
p=dim(X)[2]; n=dim(X)[1]


#(1)
#Usare un knn aggregato con la seguente scelta di parametri:
knn.agg<-KNN_AGG(X, k_min = min(log(n),p+1), k_max = max(log(n),p+1))

a <- rpart_split_finder(knn.agg, n)
plot(sort(knn.agg),col=is_out[order(knn.agg)]+1)
abline(h=a)
best_split=sort(a)[4]
#53
outliers_a<-ifelse(knn.agg>best_split,1,0)

b <-ctree_split_finder(knn.agg,n) #No Split!

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.1538462
#Area under the curve: 0.5526
#Precision: 0.06857143


#(2)
#si presenta un diferente algoritmo in grado di gestire grandi dataset.
#usare come K una media, oppure il massimo o il minimo:
round(mean(c(log(n),p+1)),0)
#13
round(max(c(log(n),p+1)),0)
#17
round(min(c(log(n),p+1)),0)
#9
#Nello specifico caso funziona molto meglio il max:

#Media
nearest<-nn2(X, query =X , k = 13, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score, col=is_out+1)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[2]
#0.45
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  #No spli!

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.1538462
#Area under the curve:  0.5528
#Precision: 0.06896552



#Max
nearest<-nn2(X, query =X , k = 17, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[2]
#0.48
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  #No spli!

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.1666667
#Area under the curve: 0.5576
#Precision: 0.07008086

#Min
nearest<-nn2(X, query =X , k = 9, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[3]
#0.41
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  #No split!

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.1474359 
#Area under the curve: 0.5509
#Precision: 0.06969697

#Best k=17
#Best AUC=0.5576
#Best split= 0.48

# Ecoli - 9 outliers ---------------------------
# data=read.csv("ecoli.csv", sep=";", header = F)
# data$V8 <- ifelse(data$V8=="omL"|data$V8=="imL"|data$V8=="imS", 1, 0)
# data$V8 <- as.factor(data$V8)
# summary(data$V8)
# write.csv(data, file="ecoli2.csv")
path <- "ecoli2.csv"
data=read.csv(path, sep=";", header=F)
is_out<-as.vector(data$V8)
X <- data[,-8]
dati<-data.frame(cbind(X, is_out))
head(dati)
summary(dati)

p=dim(X)[2]; n=dim(X)[1]

#(1)
#Usare un knn aggregato con la seguente scelta di parametri:
knn.agg<-KNN_AGG(X, k_min = min(log(n),p+1), k_max = max(log(n),p+1))

a <- rpart_split_finder(knn.agg, n)
plot(sort(knn.agg),col=is_out[order(knn.agg)]+1)
abline(h=a)
abline(h=4,col="blue")
#Scegliamomanualmente lo split
#4
best_split=4
#Il solo taglio preso in considerazione è quello scelto manualmente che non coincide con gli altri
#suggeriti da entrambi
outliers_a<-ifelse(knn.agg>best_split,1,0)

b <-ctree_split_finder(knn.agg,n)  
plot(sort(knn.agg), col=is_out[order(knn.agg)]+1)
abline(h=b)
abline(h=4,col="blue")
#Scegliamomanualmente lo split
#4 uguale ad a)
best_split=4
outliers_b<-ifelse(knn.agg>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.7777778
#Area under the curve: 0.8568
#Precision: 0.25


#(2)
#si presenta un diferente algoritmo in grado di gestire grandi dataset.
#usare come K una media, oppure il massimo o il minimo:
round(mean(c(log(n),p+1)),0)
#7
round(max(c(log(n),p+1)),0)
#8
round(min(c(log(n),p+1)),0)
#6
#Nello specifico caso funziona molto meglio il max:

#Media
nearest<-nn2(X, query =X , k = 7, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score, col=is_out+1)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
abline(h=0.19, col="blue")
sort(a)
best_split=0.19
#0.19 scegliamolo split manualmente
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
abline(h=a)
abline(h=0.19, col="blue")
best_split=0.19
#0.19 scegliamolo split manualmente gugale ad a)
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.7777778
#Area under the curve:  0.8568
#Precision: 0.25


#Max
nearest<-nn2(X, query =X , k = 8, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
abline(h=0.22, col="blue")
best_split=0.22
#0.22 scelto manualmente 
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=0.22
#0.22 scelto manualmente uguale ad a
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.7777778
#Area under the curve: 0.866
#Precision: 0.3181818


#Min
nearest<-nn2(X, query =X , k = 6, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
abline(h=0.18, col="blue")
best_split=0.18
#0.18 split scelto manualmente
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
abline(h=0.18, col="blue")
best_split=0.18
#0.18 split scelto manualmente uguale ad a
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.7777778 
#Area under the curve: 0.8568
#Precision: 0.25

#Best k=8
#Best AUC=0.866
#Best split= 0.22


# Wine - 10 outliers ---------------------------
path <- "wine.mat"
data=readMat(path)
X=data$X
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
head(dati)
summary(dati)
p=dim(X)[2]; n=dim(X)[1]

#(1)
#Usare un knn aggregato con la seguente scelta di parametri:
knn.agg<-KNN_AGG(X, k_min = min(log(n),p+1), k_max = max(log(n),p+1))

a <- rpart_split_finder(knn.agg, n)
plot(sort(knn.agg),col=is_out[order(knn.agg)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[4]
#4768
outliers_a<-ifelse(knn.agg>best_split,1,0)

b <-ctree_split_finder(knn.agg,n)  
plot(sort(knn.agg), col=is_out[order(knn.agg)]+1)
abline(h=b)
sort(b)
best_split=b
# 4731.968
outliers_b<-ifelse(knn.agg>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 1
#Area under the curve: 0.9874
#Precision: 0.7692308

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 1
#Area under the curve: 0.9874
#Precision: 0.7692308


#(2)
#si presenta un diferente algoritmo in grado di gestire grandi dataset.
#usare come K una media, oppure il massimo o il minimo:
round(mean(c(log(n),p+1)),0)
#9
round(max(c(log(n),p+1)),0)
#14
round(min(c(log(n),p+1)),0)
#5
#Nello specifico caso funziona molto meglio il max:

#Media
nearest<-nn2(X, query =X , k = 9, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score, col=is_out+1)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
abline(h=55,col="blue")
best_split=55
#55 split scelto manualmente
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
abline(h=55,col="blue")
best_split=55
#55 split scelto manualmente uguale ad a
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 1
#Area under the curve:  0.9916
#Precision: 0.8333333


#Max
nearest<-nn2(X, query =X , k = 14, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[4]
#70
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=sort(b)
#69.23422  
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 1
#Area under the curve:  0.9916
#Precision: 0.8333333

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 1
#Area under the curve: 0.9832
#Precision: 0.7142857

#Min
nearest<-nn2(X, query =X , k = 5, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[5]
#33.2
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=sort(b)
#28.98211
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 1
#Area under the curve: 0.9874
#Precision: 0.7692308

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_b)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 1
#Area under the curve: 0.9874
#Precision: 0.7692308


#Best k=14
#Best AUC=0.9916
#Best split= 70

#Dati Scalati:
X_scale=scale(X)

#(1)
#Usare un knn aggregato con la seguente scelta di parametri:
knn.agg<-KNN_AGG(X_scale, k_min = min(log(n),p+1), k_max = max(log(n),p+1))
plot(knn.agg)

a <- rpart_split_finder(knn.agg, n)
plot(sort(knn.agg),col=is_out[order(knn.agg)]+1)
abline(h=a)
abline(h=270, col="blue")
best_split=270
#270
#Split scelto manualmente
outliers_a<-ifelse(knn.agg>best_split,1,0)

b <-ctree_split_finder(knn.agg,n)  
plot(sort(knn.agg), col=is_out[order(knn.agg)]+1)
abline(h=270, col="blue")
best_split=270
#270
#Split scelto manualmente uguale ad a)
outliers_b<-ifelse(knn.agg>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0
#Area under the curve:0.4412
#Precision: 0

#(2)
#si presenta un diferente algoritmo in grado di gestire grandi dataset.
#usare come K una media, oppure il massimo o il minimo:
round(mean(c(log(n),p+1)),0)
#9
round(max(c(log(n),p+1)),0)
#14
round(min(c(log(n),p+1)),0)
#5
#Nello specifico caso funziona molto meglio il max:

#Media
nearest<-nn2(X_scale, query =X_scale , k = 9, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score, col=is_out+1)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
abline(h=2.7,col="blue")
best_split=2.7
#2.7 scelto manualmente
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
abline(h=2.7,col="blue")
best_split=2.7
#2.7 scelto manualmente uguale a a
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0
#Area under the curve: 0.416
#Precision: 0

#Max
nearest<-nn2(X_scale, query =X_scale , k = 14, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
abline(h=3, col="blue")
best_split=3
#3 split scelto manualmente
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
abline(h=3, col="blue")
best_split=3
#3 split scelto manualmente uguale ad a
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.2
#Area under the curve:0.516
#Precision: 0.09090909


#Min
nearest<-nn2(X_scale, query =X_scale , k = 5, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
#2
best_split=sort(a)[3]
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
sort(b)
best_split=2 #Uguale all' a)
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0
#Area under the curve: 0.3487
#Precision: 0

#Best k=14
#Best AUC=0.516
#Best split= 3



# Vertebral - 30 outliers -----------------------
path <- "vertebral.mat"
data=readMat(path)
X=data$X
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
head(dati)
summary(dati)

p=dim(X)[2]; n=dim(X)[1]


#(1)
#Usare un knn aggregato con la seguente scelta di parametri:
knn.agg<-KNN_AGG(X, k_min = min(log(n),p+1), k_max = max(log(n),p+1))

a <- rpart_split_finder(knn.agg, n)
plot(sort(knn.agg),col=is_out[order(knn.agg)]+1)
abline(h=a)
abline(h=300,col="blue")
#Scegliamomanualmente lo split
#300
best_split=300
#Il solo taglio preso in considerazione è quello scelto manualmente che non coincide con gli altri
#suggeriti da entrambi
outliers_a<-ifelse(knn.agg>best_split,1,0)

b <-ctree_split_finder(knn.agg,n)  #No Split

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0
#Area under the curve: 0.4524
#Precision: 0

#(2)
#si presenta un diferente algoritmo in grado di gestire grandi dataset.
#usare come K una media, oppure il massimo o il minimo:
round(mean(c(log(n),p+1)),0)
#6
round(max(c(log(n),p+1)),0)
#7
round(min(c(log(n),p+1)),0)
#5
#Nello specifico caso funziona molto meglio il max:

#Media
nearest<-nn2(X, query =X , k = 6, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score, col=is_out+1)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
abline(h=25,col="blue")
best_split=25
#25 split scelto manualmente
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  #No splits

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0
#Area under the curve:  0.469
#Precision: 0


#Max
nearest<-nn2(X, query =X , k = 7, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
abline(h=25,col="blue")
best_split=25
#25 split scelto manualmente
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  #No Splits!

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0
#Area under the curve: 0.4571
#Precision: 0

#Min
nearest<-nn2(X, query =X , k = 5, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[6]
#453
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  
plot(sort(score), col=is_out[order(score)]+1)
abline(h=b)
abline(h=25,col="blue")
best_split=25
#25 split scelto manualmente
outliers_b<-ifelse(score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0 
#Area under the curve: 0.4571
#Precision: 0

#Best k=-
#Best AUC=-
#Best split= -
#Nessun k trova un solo outlier

#Dati Scalati:
X_scale=scale(X)

#(1)
#Usare un knn aggregato con la seguente scelta di parametri:
knn.agg<-KNN_AGG(X_scale, k_min = min(log(n),p+1), k_max = max(log(n),p+1))

a <- rpart_split_finder(knn.agg, n)
plot(sort(knn.agg),col=is_out[order(knn.agg)]+1)
abline(h=a)
abline(h=20, col="blue")
best_split=20
#20
outliers_a<-ifelse(knn.agg>best_split,1,0)

b <-ctree_split_finder(knn.agg,n) #No splits!

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall:0
#Area under the curve:0.4786
#Precision: 0



#(2)
#si presenta un diferente algoritmo in grado di gestire grandi dataset.
#usare come K una media, oppure il massimo o il minimo:
round(mean(c(log(n),p+1)),0)
#6
round(max(c(log(n),p+1)),0)
#7
round(min(c(log(n),p+1)),0)
#5
#Nello specifico caso funziona molto meglio il max:

#Media
nearest<-nn2(X_scale, query =X_scale , k = 6, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score, col=is_out+1)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
abline(h=1.5, col="blue")
best_split=1.5
#1.5 scelto manualmente
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  #No Splits!

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0
#Area under the curve: 0.4786
#Precision: 0


#Max
nearest<-nn2(X_scale, query =X_scale , k = 7, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
abline(h=1.5, col="blue")
best_split=1.5
#1.5 scelto manualmente
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  #No Splits!

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0
#Area under the curve: 0.4667
#Precision: 0


#Min
nearest<-nn2(X_scale, query =X_scale , k = 5, treetype = c("kd"),
             searchtype = c("standard"), radius = 0, eps = 0)
nd<-nearest$nn.dists
score<-rowMeans(nd)
plot(score)

a <- rpart_split_finder(score, n)
plot(sort(score), col=is_out[order(score)]+1)
abline(h=a)
abline(h=1.5, col="blue")
#1.5
best_split=1.5
outliers_a<-ifelse(score>best_split,1,0)

b <-ctree_split_finder(score,n)  #No splits

#Confusion Matrix & AUC:
confusion_matrix<-table(KNN=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_KNN<- roc(response = is_out, predictor = outliers_a)
auc(roc_KNN)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0
#Area under the curve: 0.481
#Precision: 0

#Best k=-
#Best AUC=-
#Best split= -


# Yeast - 64 outliers ---------------------------
# path <- "yeast.csv"
# data=read.csv(path, sep=";", header = F)
# summary(data)
# table(data$V10)
# data$V10 <- ifelse(data$V10=="ERL"|data$V10=="POX"|data$V10=="VAC", 1, 0)
# data$V10 <- as.factor(data$V10)
# summary(data$V10)
# write.csv(data, file="yeast2.csv")
path <- "yeast2.csv"
data=read.csv(path, sep=";", header=F)
is_out<-as.vector(data$V9)
X <- data[,-9]
dati<-data.frame(cbind(X, is_out))
head(dati)
summary(dati)


# Seismic - 170 outliers ------------------------
dati <- read.csv("seismic.csv", sep=",", header=F)
dati <- dati[,-c(1,2,14,15,16)] #tolgo le prime due colonne perchè era arff e le altre perchè erano degeneri
summary(dati)

dati_comp <- dati
fac <- c(1,6)

dati <- dati[,-c(1,6)]
is_out <- dati$V19
X <- dati[,-12]


# Heart - 10 outliers ---------------------------
# train <- read.csv("SPECTF.train.csv", sep=",", header = F)
# test <- read.csv("SPECTF.test.csv", sep=",", header = F)
# dati <- rbind(train, test)
# summary(dati)
# table(dati$V1)
# dati$V1 <- ifelse(dati$V1==0, 1, 0)
# is_out<-as.vector(data$V1)
# X <- dati[,-45]
# write.csv(dati, "heart.csv")
dati <- read.csv("heart.csv", sep=";", header=F)
summary(dati)
is_out<-as.vector(data$V1)
X <- dati[,-45]



