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
setwd("E:\\Statistical_Learning\\Project")

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

mod = iForest(X, 100 , 20, seed=123)
If_score = predict(mod, X)

a <- rpart_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[4]
#0.55
outliers_a<-ifelse(If_score>best_split,1,0)

b <- ctree_split_finder(If_score, n) #No Split!!!

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_a)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.5
#Area under the curve: 0.7218
#Precision: 0.2727273

# WBC - 21 outliers ----------------------------
data=readMat("WBC.mat")
X=as.data.frame(data$X)
summary(X)
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
p=dim(X)[2]; n=dim(X)[1]

mod = iForest(X, 100 , 30, seed=123)
If_score = predict(mod, X)

a <- rpart_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=a)
abline(h=0.57, col="blue")
best_split=0.57
#0.57 split individuato manualmente!
outliers_a<-ifelse(If_score>best_split,1,0)

b <- ctree_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=b)
abline(h=0.57, col="blue")
best_split=0.57
#0.57 split individuato manualmente, uguale ad a)
outliers_b<-ifelse(If_score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_a)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.6666667
#Area under the curve: 0.8137
#Precision: 0.5


# Glass - 9 outliers ---------------------------
data=readMat("glass.mat")
X=as.data.frame(data$X)
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
summary(X)
p=dim(X)[2]; n=dim(X)[1]

mod = iForest(X, 100 , phi=20, seed=123)
If_score = predict(mod, X)

a <- rpart_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[3]
#0.53
outliers_a<-ifelse(If_score>best_split,1,0)

b <- ctree_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[2]
#0.52957, uguale ad a)
outliers_b<-ifelse(If_score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_a)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.2222222
#Area under the curve: 0.4818
#Precision: 0.03636364

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_b)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.3333333
#Area under the curve: 0.5374
#Precision: 0.05357143

#Dati Scalati:
X_scale<-scale(X)

mod = iForest(X_scale, 100 , phi=20, seed=123)
If_score = predict(mod, X)

a <- rpart_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[3]
#0.63
outliers_a<-ifelse(If_score>best_split,1,0)

b <- ctree_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=b)
sort(b)
best_split=sort(b)
#0.62585, uguale ad a)
outliers_b<-ifelse(If_score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_a)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0
#Area under the curve: 0.4439
#Precision: 0

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_b)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0
#Area under the curve: 0.3951
#Precision: 0

# Vowels - 50 outliers -------------------------
data=readMat("vowels.mat")
X=as.data.frame(data$X)
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
summary(X)
p=dim(X)[2]; n=dim(X)[1]

mod = iForest(X, 1000 , phi=256, seed=123)
If_score = predict(mod, X)

a <- rpart_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[4]
#0.50
outliers_a<-ifelse(If_score>best_split,1,0)

b <- ctree_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[3]
#0.50325, uguale ad a)
outliers_b<-ifelse(If_score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_a)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.36
#Area under the curve: 0.6363
#Precision: 0.1276596

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_b)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.36
#Area under the curve: 0.6398
#Precision: 0.1374046

#Dati Scalati:
X_scale<-scale(X)

mod = iForest(X_scale, 1000 , phi=200, seed=123)
If_score = predict(mod, X)

a <- rpart_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[3]
#0.49
outliers_a<-ifelse(If_score>best_split,1,0)

b <- ctree_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=b)
sort(b)
best_split=sort(b)
#0.49232, uguale ad a)
outliers_b<-ifelse(If_score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_a)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.42
#Area under the curve: 0.6353
#Precision: 0.09090909

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_b)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.42
#Area under the curve: 0.6396
#Precision: 0.09589041

# Cardio - 176 outliers ------------------------
data=readMat("cardio.mat")
X=as.data.frame(data$X)
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
summary(X)
p=dim(X)[2]; n=dim(X)[1]

mod = iForest(X, 1000 , phi=256, seed=123)
If_score = predict(mod, X)

a <- rpart_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=a)
abline(h=0.52,col="blue")
best_split=0.52
#0.52, split scelto manualmente
outliers_a<-ifelse(If_score>best_split,1,0)

b <- ctree_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=b)
abline(h=0.52,col="blue")
best_split=0.52
#0.52, split scelto manualmente uguale ad a)
outliers_b<-ifelse(If_score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_a)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.3465909
#Area under the curve: 0.6615
#Precision: 0.61


#Dati Scalati:
X_scale<-scale(X)

mod = iForest(X_scale, 100 , phi=20, seed=123)
If_score = predict(mod, X)

a <- rpart_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=a)
abline(h=0.525, col="blue")
best_split=0.525
#0.525 split scelto manualmente
outliers_a<-ifelse(If_score>best_split,1,0)

b <- ctree_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=b)
sort(b)
abline(h=0.525, col="blue")
best_split=0.525
#0.525 split scelto manualmente uguale ad a)
outliers_b<-ifelse(If_score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_a)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.6647727
#Area under the curve: 0.7985
#Precision: 0.510917


# Thyroid - 93 outliers ------------------------
data=readMat("thyroid.mat")
X=as.data.frame(data$X)
Y=data$y
summary(X)
dati=cbind(X,Y)
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
summary(X)
p=dim(X)[2]; n=dim(X)[1]

mod = iForest(X, 1000 , phi=256, seed=123)
If_score = predict(mod, X)

a <- rpart_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=a)
sort(a)
abline(h=0.5, col="blue")
best_split=0.5
#0.50 scelto manualmente
outliers_a<-ifelse(If_score>best_split,1,0)

b <- ctree_split_finder(If_score, n)  #No Splits!

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_a)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.9462366
#Area under the curve: 0.9381
#Precision: 0.2543353


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

mod = iForest(X, 1000 , phi=256, seed=123)
If_score = predict(mod, X)

a <- rpart_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[2]
#0.35
outliers_a<-ifelse(If_score>best_split,1,0)

b <- ctree_split_finder(If_score, n) #No split

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_a)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.172043
#Area under the curve: 0.5565
#Precision: 0.06866953

#Dati Scalati:
X_scale<-scale(X)

mod = iForest(X_scale, 500 , phi=256, seed=123)
If_score = predict(mod, X)

a <- rpart_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[2]
#0.33
outliers_a<-ifelse(If_score>best_split,1,0)

b <- ctree_split_finder(If_score, n) #No splits!

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_a)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.06451613
#Area under the curve: 0.4779
#Precision: 0.01477833


# Musk - 97 outliers ---------------------------
data=readMat("musk.mat")
X=as.data.frame(data$X)
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
summary(X)
p=dim(X)[2]; n=dim(X)[1]

mod = iForest(X, 1000 , phi=256, seed=123)
If_score = predict(mod, X)

a <- rpart_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[6]
#0.53
outliers_a<-ifelse(If_score>best_split,1,0)

b <- ctree_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[5]
#0.52239
outliers_b<-ifelse(If_score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_a)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 1
#Area under the curve: 1
#Precision: 1

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_b)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 1
#Area under the curve: 0.9998
#Precision: 0.9897959

#Dati Scalati:
X_scale<-scale(X)

mod = iForest(X_scale, 500 , phi=256, seed=123)
If_score = predict(mod, X)

a <- rpart_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[3]
#0.68
outliers_a<-ifelse(If_score>best_split,1,0)

b <- ctree_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[5]
#0.67876, uguale ad a)
outliers_b<-ifelse(If_score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_a)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.4742268
#Area under the curve: 0.7304
#Precision: 0.5348837

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_b)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.5979381
#Area under the curve: 0.7889
#Precision: 0.4915254

# Satimage-2 - 71 outliers ---------------------
data=readMat("satimage-2.mat")
X=as.data.frame(data$X)
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
summary(X)
p=dim(X)[2]; n=dim(X)[1]

mod = iForest(X, 500 , phi=256, seed=123)
If_score = predict(mod, X)

a <- rpart_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[3]
#0.62
outliers_a<-ifelse(If_score>best_split,1,0)

b <- ctree_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[1]
#0.61853, uguale ad a)
outliers_b<-ifelse(If_score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_a)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.8169014
#Area under the curve: 0.9084
#Precision: 0.9830508

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_b)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.8169014
#Area under the curve: 0.9084
#Precision: 0.9830508

#Dati Scalati:
X_scale<-scale(X)

mod = iForest(X_scale, 500 , phi=256, seed=123)
If_score = predict(mod, X)

#PROBLEMA NON INDIFFERENTE:
summary(If_score)
#Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#0.7024  0.7024  0.7024  0.7024  0.7024  0.7024


# Letter Recognition - 100 outliers ------------
path <- "letter.mat"
data=readMat(path)
X=data$X
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
summary(X)
p=dim(X)[2]; n=dim(X)[1]

mod = iForest(X, 500 , phi=256, seed=123)
If_score = predict(mod, X)

a <- rpart_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[5]
#0.50
outliers_a<-ifelse(If_score>best_split,1,0)

b <- ctree_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[5]
#0.50393, uguale ad a)
outliers_b<-ifelse(If_score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_a)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.13
#Area under the curve: 0.5303
#Precision: 0.1111111

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_b)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.12
#Area under the curve: 0.532
#Precision: 0.125

#Dati Scalati:
X_scale<-scale(X)

mod = iForest(X_scale, 500 , phi=256, seed=123)
If_score = predict(mod, X)

a <- rpart_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=a)
abline(h=0.735, col="blue")
sort(a)
best_split=0.735
#0.735 Split scelto manualmente! 
outliers_a<-ifelse(If_score>best_split,1,0)

b <- ctree_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=b)
abline(h=0.735, col="blue")
best_split=0.735
#0.735 Split scelto manualmente! uguale ad a)
outliers_b<-ifelse(If_score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_a)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.09
#Area under the curve: 0.523
#Precision: 0.12


# Speech - 61 outliers -------------------------
path <- "speech.mat"
data=readMat(path)
X=data$X
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
summary(X)
p=dim(X)[2]; n=dim(X)[1]

mod = iForest(X, 200 , phi=256, seed=123)
If_score = predict(mod, X)

a <- rpart_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[5]
#0.47
outliers_a<-ifelse(If_score>best_split,1,0)

b <- ctree_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[2]
#0.47187, uguale ad a)
outliers_b<-ifelse(If_score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_a)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.06557377
#Area under the curve: 0.5111
#Precision: 0.02484472

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_b)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.6393443
#Area under the curve: 0.4747
#Precision: 0.01535433

#Dati Scalati:
X_scale<-scale(X)

mod = iForest(X_scale, 200 , phi=256, seed=123)
If_score = predict(mod, X)

a <- rpart_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[5]
#0.45
outliers_a<-ifelse(If_score>best_split,1,0)

b <- ctree_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[3]
#0.45056
outliers_b<-ifelse(If_score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_a)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.09836066
#Area under the curve: 0.5095
#Precision: 0.02040816

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_b)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.06557377
#Area under the curve: 0.4951
#Precision: 0.01444043

# Pima - 268 outliers --------------------------
path <- "pima.mat"
data=readMat(path)
X=data$X
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
summary(X)
p=dim(X)[2]; n=dim(X)[1]

mod = iForest(X, 500 , phi=200, seed=123)
If_score = predict(mod, X)

a <- rpart_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=a)
abline(h=0.54, col="blue")
best_split=0.54
#0.54 Split scelto maualmente
outliers_a<-ifelse(If_score>best_split,1,0)

b <- ctree_split_finder(If_score, n)  #No splits!

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_a)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.0858209
#Area under the curve: 0.5259
#Precision: 0.575


#Dati Scalati:
X_scale<-scale(X)

mod = iForest(X_scale, 500 , phi=200, seed=123)
If_score = predict(mod, X)

a <- rpart_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=a)
abline(h=0.69, col="blue")
best_split=0.69
#0.69 Split scelto manualmente
outliers_a<-ifelse(If_score>best_split,1,0)

b <- ctree_split_finder(If_score, n) #No splits

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_a)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.6156716
#Area under the curve: 0.4958
#Precision: 0.3459119

# Satellite - 2036 outliers --------------------
path <- "satellite.mat"
data=readMat(path)
X=data$X
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
head(dati)
summary(dati)
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
summary(X)
p=dim(X)[2]; n=dim(X)[1]

mod = iForest(X, 500 , phi=256, seed=123)
If_score = predict(mod, X)

a <- rpart_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[5]
#0.55
outliers_a<-ifelse(If_score>best_split,1,0)

b <- ctree_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=b)
sort(b)
best_split=0.55 #uguale ad a)
outliers_b<-ifelse(If_score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_a)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.2421415
#Area under the curve: 0.6189
#Precision: 0.9628906

#Dati Scalati:
X_scale<-scale(X)

mod = iForest(X_scale, 500 , phi=256, seed=123)
If_score = predict(mod, X)

a <- rpart_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)

#Problema non indifferente:
#Summary IF_score:

summary(If_score)
#Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#0.6942  0.6942  0.6942  0.6942  0.6942  0.6942 

# Shuttle - 3511 outliers ----------------------
path <- "shuttle.mat"
data=readMat(path)
X=data$X
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
summary(X)
p=dim(X)[2]; n=dim(X)[1]

mod = iForest(X, 500 , phi=256, seed=123)
If_score = predict(mod, X)

a <- rpart_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=a)
abline(h=0.56, col="blue")
sort(a)
best_split=0.56
#0.56 scelto manualmente
outliers_a<-ifelse(If_score>best_split,1,0)

b <- ctree_split_finder(If_score, n)#No splits

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_a)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.9490174
#Area under the curve: 0.9734
#Precision: 0.9708625


#Dati Scalati:
X_scale<-scale(X)

mod = iForest(X_scale, 500 , phi=256, seed=123)
If_score = predict(mod, X)

a <- rpart_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=a)
abline(h=0.743, col="blue")
best_split=0.743
#0.743 split scelto manualmente
outliers_a<-ifelse(If_score>best_split,1,0)

b <- ctree_split_finder(If_score, n) #No splits

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_a)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.381088
#Area under the curve: 0.6537
#Precision: 0.2848627



# BreastW - 239 outliers -----------------------
path="breastw.mat"
data=readMat(path)
X=data$X
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
summary(X)
p=dim(X)[2]; n=dim(X)[1]

mod = iForest(X, 500 , phi=200, seed=123)
If_score = predict(mod, X)

a <- rpart_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=a)
abline(h=0.50,col="blue")
best_split=0.5
#0.5 split scelto manualmente
outliers_a<-ifelse(If_score>best_split,1,0)

b <- ctree_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=b)
abline(h=0.50,col="blue")
best_split=0.5
#0.5 split scelto manualmente, uguale ad a)
outliers_b<-ifelse(If_score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_a)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.9790795
#Area under the curve: 0.9614
#Precision: 0.9034749


#Dati Scalati:
X_scale<-scale(X)

mod = iForest(X_scale, 500 , phi=200, seed=123)
If_score = predict(mod, X)

a <- rpart_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[4]
#0.61
outliers_a<-ifelse(If_score>best_split,1,0)

b <- ctree_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=b)
sort(b)
best_split=0.61
#messo manualmente uguale ad a)
outliers_b<-ifelse(If_score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_a)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.8242678
#Area under the curve: 0.8412
#Precision: 0.7576923


fac<-1:p
for (i in 1:length(fac)){
  dati[,i]<-as.factor(dati[,i])
}
#One hot encoding:
dati<-cbind(as.data.frame(model.matrix(~.-1,data=dati[,-(p+1)])),is_out)
X=dati[,1:(ncol(dati)-1)]
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
summary(X)
p=dim(X)[2]; n=dim(X)[1]

mod = iForest(X, 500 , phi=200, seed=123)
If_score = predict(mod, X)

a <- rpart_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=a)
sort(a)
abline(h=0.415, col="blue")
best_split=0.415
#0.415 split scelto manualmente
outliers_a<-ifelse(If_score>best_split,1,0)

b <- ctree_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=b)
abline(h=0.415, col="blue")
best_split=0.415
#0.415 split scelto manualmente uguale ad a)
outliers_b<-ifelse(If_score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_a)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.8661088
#Area under the curve: 0.9162
#Precision: 0.9324324


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

mod = iForest(X, 200 , phi=200, seed=123)
If_score = predict(mod, X)

a <- rpart_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[3]
#0.42
outliers_a<-ifelse(If_score>best_split,1,0)

b <- ctree_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=b)
sort(b)
best_split=0.42
#0.42 manualmente uguale ad a)
outliers_b<-ifelse(If_score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_a)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.4393939
#Area under the curve: 0.6847
#Precision: 0.5178571


#Dati Scalati:
X_scale<-scale(X)

mod = iForest(X_scale, 200 , phi=200, seed=123)
If_score = predict(mod, X)

a <- rpart_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[3]
#0.66
outliers_a<-ifelse(If_score>best_split,1,0)

b <- ctree_split_finder(If_score, n) #No Splits!


#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_a)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.09090909
#Area under the curve: 0.526
#Precision: 0.2857143


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

is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
summary(X)
p=dim(X)[2]; n=dim(X)[1]

mod = iForest(X, 200 , phi=200, seed=123)
If_score = predict(mod, X)

a <- rpart_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=a)
abline(h=0.335,col="blue")
best_split=0.335
#0.335 split scelto manualmente
outliers_a<-ifelse(If_score>best_split,1,0)

b <- ctree_split_finder(If_score, n) #No splits

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_a)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.4090909
#Area under the curve: 0.6851
#Precision: 0.6428571


#Dati Scalati:
X_scale<-scale(X)

mod = iForest(X_scale, 200 , phi=200, seed=123)
If_score = predict(mod, X)

a <- rpart_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[2]
#0.39
outliers_a<-ifelse(If_score>best_split,1,0)

b <- ctree_split_finder(If_score, n) #No Split

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_a)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.2121212
#Area under the curve: 0.5296
#Precision: 0.1917808


# Ionosphere - 126 outliers --------------------
path="ionosphere.mat"
data=readMat(path)
X=data$X
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
summary(X)
p=dim(X)[2]; n=dim(X)[1]

mod = iForest(X, 500 , phi=100, seed=123)
If_score = predict(mod, X)

a <- rpart_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[5]
#0.55
outliers_a<-ifelse(If_score>best_split,1,0)

b <- ctree_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=b)
sort(b)
best_split=0.55
#0.55 split scelto maualmente uguale ad a)
outliers_b<-ifelse(If_score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_a)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.4603175
#Area under the curve: 0.7168
#Precision: 0.90625

# Mnist - 700 outliers -------------------------
path="mnist.mat"
#inserire path
data=readMat(path)
X=data$X
X<-X[,-100]
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
summary(X)
p=dim(X)[2]; n=dim(X)[1]

mod = iForest(X, 200 , phi=256, seed=123)
If_score = predict(mod, X)

a <- rpart_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[4]
#0.54
outliers_a<-ifelse(If_score>best_split,1,0)

b <- ctree_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[3]
#0.54043, uguale ad a)
outliers_b<-ifelse(If_score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_a)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.2285714
#Area under the curve: 0.5932
#Precision: 0.3547672

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_b)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.2271429
#Area under the curve: 0.5933
#Precision: 0.3621868

#Dati Scalati:
X_scale<-scale(X)

mod = iForest(X_scale, 200 , phi=256, seed=123)
If_score = predict(mod, X)

a <- rpart_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[3]
#0.60
outliers_a<-ifelse(If_score>best_split,1,0)

b <- ctree_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=b)
sort(b)
best_split=0.60
#0.60 scelto manualmente, uguale ad a)
outliers_b<-ifelse(If_score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_a)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall:  0.06
#Area under the curve: 0.5214
#Precision: 0.2608696


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

mod = iForest(X, 200 , phi=256, seed=123)
If_score = predict(mod, X)

a <- rpart_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=a)
abline(h=0.52,col="blue")
best_split=0.52
#0.52 scelto manualmente
outliers_a<-ifelse(If_score>best_split,1,0)

b <- ctree_split_finder(If_score, n) #No splits!

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_a)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.1333333
#Area under the curve: 0.5211
#Precision: 0.04149378


#Dati Scalati:
X_scale<-scale(X)

mod = iForest(X_scale, 200 , phi=256, seed=123)
If_score = predict(mod, X)

a <- rpart_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[4]
#0.62
outliers_a<-ifelse(If_score>best_split,1,0)

b <- ctree_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=b)
sort(b)
best_split=sort(b)
#0.61127, uguale ad a)
outliers_b<-ifelse(If_score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_a)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.4466667
#Area under the curve: 0.6965
#Precision: 0.1976401

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_b)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.62
#Area under the curve: 0.7615
#Precision: 0.1592466

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
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
summary(X)
p=dim(X)[2]; n=dim(X)[1]

mod = iForest(X, 200 , phi=256, seed=123)
If_score1 = predict(mod, X)

a <- rpart_split_finder(If_score1, n)
plot(sort(If_score1), col=is_out[order(If_score1)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[5]
#0.72
outliers_a<-ifelse(If_score1>best_split,1,0)

b <- ctree_split_finder(If_score1, n)
plot(sort(If_score1), col=is_out[order(If_score1)]+1)
abline(h=b)
sort(b)
best_split=0.72
#0.72 manualmente uguale ad a)
outliers_b<-ifelse(If_score1>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_a)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.987336
#Area under the curve: 0.9935
#Precision: 0.9114823


#Dati Scalati:
X_scale<-scale(X)

mod = iForest(X_scale, 200 , phi=256, seed=123)
If_score2 = predict(mod, X)

a <- rpart_split_finder(If_score2, n)
plot(sort(If_score2), col=is_out[order(If_score2)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[4]
#0.76
outliers_a<-ifelse(If_score2>best_split,1,0)

b <- ctree_split_finder(If_score2, n)
plot(sort(If_score2), col=is_out[order(If_score2)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[3]
#0.75543
outliers_b<-ifelse(If_score2>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_a)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.9963817
#Area under the curve: 0.996
#Precision: 0.4697228

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_b)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.9963817
#Area under the curve: 0.9957
#Precision: 0.4436166


# ForestCover - 2747 outliers ------------------
path="cover.mat"
data=readMat(path)
X=data$X
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
summary(X)
p=dim(X)[2]; n=dim(X)[1]

mod = iForest(X, 200 , phi=256, seed=123)
If_score3 = predict(mod, X)

a <- rpart_split_finder(If_score3, n)
plot(sort(If_score3), col=is_out[order(If_score3)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[5]
#0.55
outliers_a<-ifelse(If_score3>best_split,1,0)

b <- ctree_split_finder(If_score3, n)
plot(sort(If_score3), col=is_out[order(If_score3)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[5]
#0.54548
outliers_b<-ifelse(If_score3>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_a)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.2475428
#Area under the curve: 0.605
#Precision: 0.06010253

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_b)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.271205
#Area under the curve: 0.6141
#Precision: 0.05773404

#Dati Scalati:
X_scale<-scale(X)

mod = iForest(X_scale, 200 , phi=256, seed=123)
If_score4 = predict(mod, X)

a <- rpart_split_finder(If_score4, n)
plot(sort(If_score4), col=is_out[order(If_score4)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[3]
#0.74
outliers_a<-ifelse(If_score4>best_split,1,0)

b <- ctree_split_finder(If_score4, n)
plot(sort(If_score4), col=is_out[order(If_score4)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[7]
#0.73780 uguale ad a)
outliers_b<-ifelse(If_score4>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_a)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.5023662
#Area under the curve: 0.3684
#Precision: 0.006322097

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_b)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.5023662
#Area under the curve: 0.3684
#Precision: 0.006321923

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


mod = iForest(X, 200 , phi=256, seed=123)
If_score5 = predict(mod, X)

a <- rpart_split_finder(If_score5, n)
plot(sort(If_score5), col=is_out[order(If_score5)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[4]
#0.45
outliers_a<-ifelse(If_score5>best_split,1,0)

b <- ctree_split_finder(If_score5, n)
plot(sort(If_score5), col=is_out[order(If_score5)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[6]
#0.45465
outliers_b<-ifelse(If_score5>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_a)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.8798689
#Area under the curve: 0.9126
#Precision: 0.1350581

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_b)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.8565708
#Area under the curve: 0.9041
#Precision: 0.146349

#Dati Scalati:
X_scale<-scale(X)

mod = iForest(X_scale, 200 , phi=256, seed=123)
If_score6 = predict(mod, X)

a <- rpart_split_finder(If_score6, n)
plot(sort(If_score6), col=is_out[order(If_score6)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[1]
#0.54
outliers_a<-ifelse(If_score6>best_split,1,0)

b <- ctree_split_finder(If_score6, n)
plot(sort(If_score6), col=is_out[order(If_score6)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[1]
#0.54165, uguale ad a)
outliers_b<-ifelse(If_score6>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_a)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.5078267
#Area under the curve: 0.3773
#Precision: 0.006495169

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_b)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.5041864
#Area under the curve: 0.3998
#Precision: 0.006891096

# Smtp (KDDCUP99) - 30 outliers ----------------
path="smtp1.mat"
data=readMat(path)
X=data$X
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
summary(dati)
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
summary(X)
p=dim(X)[2]; n=dim(X)[1]

mod = iForest(X, 200 , phi=256, seed=123)
If_score7 = predict(mod, X)

a <- rpart_split_finder(If_score7, n)
plot(sort(If_score7), col=is_out[order(If_score7)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[4]
#0.60
outliers_a<-ifelse(If_score7>best_split,1,0)

b <- ctree_split_finder(If_score7, n)
plot(sort(If_score7), col=is_out[order(If_score7)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[4]
#0.59722
outliers_b<-ifelse(If_score7>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_a)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall:  0.7
#Area under the curve: 0.8295
#Precision: 0.005362615

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_b)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.7
#Area under the curve: 0.8289
#Precision: 0.005216095

#Dati Scalati:
X_scale<-scale(X)

mod = iForest(X_scale, 200 , phi=256, seed=123)
If_score8 = predict(mod, X)

a <- rpart_split_finder(If_score8, n)
plot(sort(If_score8), col=is_out[order(If_score8)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[3]
#0.76
outliers_a<-ifelse(If_score8>best_split,1,0)

b <- ctree_split_finder(If_score8, n)
plot(sort(If_score8), col=is_out[order(If_score8)]+1)
abline(h=b)
sort(b)
best_split=sort(b)[5]
#0.75621
outliers_b<-ifelse(If_score8>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_a)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0
#Area under the curve: 0.4202
#Precision: 0

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_b, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_b)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.1333333
#Area under the curve: 0.6991
#Precision: 7.910767e-05

# Mammography - 260 outliers -------------------
path="mammography.mat"
data=readMat(path)
X=data$X
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
head(dati)
summary(dati)
dati_resp <- read.csv("mammo_resp.csv", sep=",")
head(dati_resp)
summary(as.factor(dati[,7]))
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
summary(X)
p=dim(X)[2]; n=dim(X)[1]

mod = iForest(X, 200 , phi=256, seed=123)
If_score = predict(mod, X)

a <- rpart_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[1]
#0.62
outliers_a<-ifelse(If_score>best_split,1,0)

  b <- ctree_split_finder(If_score, n) #No Splits!

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_a)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.1615385
#Area under the curve: 0.5772
#Precision: 0.3471074


#Dati Scalati:
X_scale<-scale(X)

mod = iForest(X_scale, 200 , phi=256, seed=123)
If_score = predict(mod, X)

a <- rpart_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[1]
#0.62
outliers_a<-ifelse(If_score>best_split,1,0)

b <- ctree_split_finder(If_score, n)   #No Splits!

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_a)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.1615385
#Area under the curve: 0.5772
#Precision: 0.3471074


# Annthyroid - 534 outliers --------------------
path <-"annthyroid.mat"
data=readMat(path)
X=data$X
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
head(dati)
summary(dati)
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
summary(X)
p=dim(X)[2]; n=dim(X)[1]

mod = iForest(X, 500 , phi=256, seed=123)
If_score = predict(mod, X)

a <- rpart_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=a)
abline(h=0.5, col="blue")
best_split=0.5
#0.50 splits scelto manualmente
outliers_a<-ifelse(If_score>best_split,1,0)

b <- ctree_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=b)
abline(h=0.5, col="blue")
best_split=0.5
#0.50 splits scelto manualmente uguale ad a)
outliers_b<-ifelse(If_score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_a)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.3539326
#Area under the curve: 0.6425
#Precision: 0.2916667


#Dati Scalati:
X_scale<-scale(X)

mod = iForest(X_scale, 200 , phi=256, seed=123)
If_score = predict(mod, X)

a <- rpart_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[2]
#0.37
outliers_a<-ifelse(If_score>best_split,1,0)

b <- ctree_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=b)
sort(b)
best_split=0.37
#0.37 sceltomanualmente uguale ad a)
outliers_b<-ifelse(If_score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_a)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.2565543
#Area under the curve: 0.5939
#Precision: 0.2302521

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

mod = iForest(X, 200 , phi=256, seed=123)
If_score = predict(mod, X)

a <- rpart_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[3]
#0.35
outliers_a<-ifelse(If_score>best_split,1,0)

b <- ctree_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=b)
sort(b)
best_split=0.35
#0.35 sceltomanualmente uguale ad a)
outliers_b<-ifelse(If_score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_a)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.1011236
#Area under the curve: 0.5214
#Precision: 0.1218962


#Dati Scalati:
X_scale<-scale(X)

mod = iForest(X_scale, 200 , phi=200, seed=123)
If_score = predict(mod, X)

a <- rpart_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[2]
#0.33
outliers_a<-ifelse(If_score>best_split,1,0)

b <- ctree_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=b)
sort(b)
best_split=0.33
#0.33 scelto manualmente uguale ad a)
outliers_b<-ifelse(If_score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_a)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.05992509
#Area under the curve: 0.5067
#Precision: 0.09356725

# Pendigits - 156 outliers ---------------------
path <-"pendigits.mat"
data=readMat(path)
X=data$X
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
head(dati)
summary(dati)
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
summary(X)
p=dim(X)[2]; n=dim(X)[1]

mod = iForest(X, 200 , phi=256, seed=123)
If_score = predict(mod, X)

a <- rpart_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=a)
abline(h=0.58, col="blue")
best_split=0.58
#0.58 split scelto maualmente
outliers_a<-ifelse(If_score>best_split,1,0)

b <- ctree_split_finder(If_score, n) #no splits!

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_a)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.3589744
#Area under the curve: 0.668
#Precision: 0.2666667

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

mod = iForest(X, 500 , phi=150, seed=123)
If_score = predict(mod, X)

a <- rpart_split_finder(If_score, n)
plot(If_score)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=a)
abline(h=0.55,col="blue")
best_split=0.55
#0.55 scelto maualmente
outliers_a<-ifelse(If_score>best_split,1,0)

b <- ctree_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=b)
sort(b)
best_split=0.55
#0.55scelto manualmente, uguale ad a)
outliers_b<-ifelse(If_score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_a)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.7777778
#Area under the curve: 0.8736
#Precision: 0.4117647


# Wine - 10 outliers ---------------------------
path <- "wine.mat"
data=readMat(path)
X=data$X
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
head(dati)
summary(dati)
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
summary(X)
p=dim(X)[2]; n=dim(X)[1]

mod = iForest(X, 500 , phi=50, seed=123)
If_score = predict(mod, X)

a <- rpart_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[3]
#0.52
outliers_a<-ifelse(If_score>best_split,1,0)

b <- ctree_split_finder(If_score, n)  #No splits

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_a)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.6
#Area under the curve: 0.7496
#Precision: 0.3333333


#Dati Scalati:
X_scale<-scale(X)

mod = iForest(X_scale, 500 , phi=50, seed=123)
If_score = predict(mod, X)

a <- rpart_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[5]
#0.68
outliers_a<-ifelse(If_score>best_split,1,0)

b <- ctree_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=b)
sort(b)
best_split=0.68
#0.68 scelto manualmente, uguale ad a)
outliers_b<-ifelse(If_score>best_split,1,0)

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_a)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.9
#Area under the curve: 0.887
#Precision: 0.375


# Vertebral - 30 outliers -----------------------
path <- "vertebral.mat"
data=readMat(path)
X=data$X
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
summary(X)
p=dim(X)[2]; n=dim(X)[1]

mod = iForest(X, 500 , phi=50, seed=123)
If_score = predict(mod, X)

a <- rpart_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=a)
sort(a)
best_split=sort(a)[2]
#0.53
outliers_a<-ifelse(If_score>best_split,1,0)

b <- ctree_split_finder(If_score, n) #No splits!

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_a)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0
#Area under the curve: 0.4333
#Precision: 0


#Dati Scalati:
X_scale<-scale(X)

mod = iForest(X_scale, 500 , phi=50, seed=123)
If_score = predict(mod, X)

a <- rpart_split_finder(If_score, n)
plot(sort(If_score), col=is_out[order(If_score)]+1)
abline(h=a)
abline(h=0.7, col="blue")
best_split=0.70 
#0.70 scelto manualmente
outliers_a<-ifelse(If_score>best_split,1,0)

b <- ctree_split_finder(If_score, n)#No splits

#Confusion Matrix & AUC:
confusion_matrix<-table(IForest=outliers_a, Actual=is_out); confusion_matrix
#Valutazione performance:
roc_IForest<- roc(response = is_out, predictor = outliers_a)
auc(roc_IForest)
precision=confusion_matrix[2,2]/sum(confusion_matrix[2,]);precision
recall=confusion_matrix[2,2]/sum(confusion_matrix[,2]);recall
#Recall: 0.4666667
#Area under the curve: 0.6929
#Precision: 0.07253886

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

