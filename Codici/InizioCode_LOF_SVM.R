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
p=dim(X)[2]; n=dim(X)[1]
is_out=data$y
dati<-data.frame(cbind(X, is_out)) 
fac <- c(1:18) #tutte le variabili
for (i in 1:length(fac)){
  dati[,i]<-as.factor(dati[,i])
}
#One hot encoding:
dati<-model.matrix(~.-1,dati[,-(p+1)])
X <- dati[,1:(ncol(dati)-1)]
p=dim(X)[2]; n=dim(X)[1]

### LOF:
outlier.scores <- lof(X, k=mean(c(log(n), p+1, 2*p)))

rfind <- rpart_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=18)
abline(h=rfind, col=4)
rbest_taglio <- sort(rfind)[2] #1.1
out_previsti <- rep(0,n)
out_previsti <- ifelse(outlier.scores>rbest_taglio, 1, 0)

#Confusion Matrix & AUC:
table(LOF=out_previsti, Actual=is_out)
#Valutazione performance:
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.7465
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.75
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.5

cfind <- ctree_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=19)
abline(h=cfind, col=cfind+1)
cbest_taglio <- sort(cfind) #nessun taglio con ctree
# out_previsti=rep(0,n)
# out_previsti[out_previsti>numero]=1
# confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
# precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); 

#BEST LOF: taglio = 1.1 ; AUC = 0.7465 ; precision = 0.75 ; recall=0.5

### SVM:

# nu with 4 values:
nu_val <- c(0.01, 0.1, 0.9, 0.5)
auc_svm <- c()
precision <- c()
recall <- c()
for (el in nu_val){
  svm.model<-svm(X, y=NULL,
                 type='one-classification',
                 nu=el,
                 kernel="radial")
  
  svm.predtrain<-predict(svm.model, X)
  out <- rep(0, n)
  out[svm.predtrain==FALSE] <- 1
  roc_lof<- roc(response = is_out, predictor = out)
  auc_svm <- c(auc_svm ,auc(roc_lof))
  confusion_matrix<-table(SVM=out, Actual=is_out)
  precision <- c(precision, confusion_matrix[2,2]/sum(confusion_matrix[2,]))
  recall <- c(recall, confusion_matrix[2,2]/sum(confusion_matrix[,2]))
}
auc_svm; precision; recall

#BEST SVM NU: nu = 0.5 ; AUC = 0.7606 ; precision = 0.0811 ; recall = 1.00

# OCSVM paper with k=7 and k=Best knn value:
#K=7:
distanze<-kNNdist(X,k=7)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,  
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.6667
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.0658
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.833

#K=BEST knn:
distanze<-kNNdist(X,k=5)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE, 
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.75
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.078
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #1

#BEST SVM K: k = 5 ; AUC = 0.75 ; precision = 0.078 ; recall = 1.00

# WBC - 21 outliers ----------------------------
data=readMat("WBC.mat")
X=as.data.frame(data$X)
p=dim(X)[2]; n=dim(X)[1]
is_out=data$y
dati<-data.frame(cbind(X, is_out)) 

### LOF:
outlier.scores <- lof(X, k=mean(c(log(n), p+1, 2*p)))

rfind <- rpart_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=18)
abline(h=rfind, col=4)
abline(h=1.85) #Vedendo i tagli suggeriti, 1.85 sembra i taglio migliore non presente 
rbest_taglio <- 1.85 #1.85
out_previsti <- rep(0,n)
out_previsti <- ifelse(outlier.scores>rbest_taglio, 1, 0)

#Confusion Matrix & AUC:
table(LOF=out_previsti, Actual=is_out)
#Valutazione performance:
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.6611
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.636
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.33

cfind <- ctree_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=19)
abline(h=cfind, col=cfind+1)
cbest_taglio <- sort(cfind)[1]
out_previsti=rep(0,n)
out_previsti <- ifelse(outlier.scores>cbest_taglio, 1, 0)
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.8585
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.34
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.81

#BEST LOF: taglio = 1.353 ; AUC = 0.858 ; precision = 0.34 ; recall = 0.81

### SVM:

# nu with 4 values:
nu_val <- c(0.01, 0.1, 0.9, 0.5)
auc_svm <- c()
precision <- c()
recall <- c()
for (el in nu_val){
  svm.model<-svm(X, y=NULL,
                 type='one-classification',
                 nu=el,
                 kernel="radial")
  
  svm.predtrain<-predict(svm.model, X)
  out <- rep(0, n)
  out[svm.predtrain==FALSE] <- 1
  roc_lof<- roc(response = is_out, predictor = out)
  auc_svm <- c(auc_svm ,auc(roc_lof))
  confusion_matrix<-table(SVM=out, Actual=is_out)
  precision <- c(precision, confusion_matrix[2,2]/sum(confusion_matrix[2,]))
  recall <- c(recall, confusion_matrix[2,2]/sum(confusion_matrix[,2]))
}
auc_svm; precision; recall

#BEST SVM NU: nu = 0.5 ; AUC = 0.7619 ; precision = 0.1099 ; recall = 1.00

# OCSVM paper with k=7 and k=Best knn value:
#K=7:
distanze<-kNNdist(X,k=7)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.7619
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.1099
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #1.00

#K=BEST knn:
distanze<-kNNdist(X,k=6)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.7619
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.1099
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #1.00

#BEST SVM K: k = 7 ; AUC = 0.7619 ; precision = 0.1099; recall = 1.00

# Glass - 9 outliers ---------------------------
data=readMat("glass.mat")
X=as.data.frame(data$X)
p=dim(X)[2]; n=dim(X)[1]
is_out=data$y
dati<-data.frame(cbind(X, is_out)) 

### LOF:
outlier.scores <- lof(X, k=mean(c(log(n), p+1, 2*p)))

rfind <- rpart_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=18)
abline(h=rfind, col=4)
rbest_taglio <- max(rfind)
out_previsti <- rep(0,n)
out_previsti <- ifelse(outlier.scores>rbest_taglio, 1, 0)

#Confusion Matrix & AUC:
table(LOF=out_previsti, Actual=is_out)
#Valutazione performance:
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.5989
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.2857
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.222

cfind <- ctree_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=19)
abline(h=cfind, col=cfind+1)
cbest_taglio <- sort(cfind)[1] 
out_previsti=rep(0,n)
out_previsti <- ifelse(outlier.scores>cbest_taglio, 1, 0)
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.5867
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.167
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.22

#BEST LOF: taglio = 2.9 ; AUC = 0.5989 ; precision = 0.286 ; recall = 0.222

### SVM:

# nu with 4 values:
nu_val <- c(0.01, 0.1, 0.9, 0.5)
auc_svm <- c()
precision <- c()
recall <- c()
for (el in nu_val){
  svm.model<-svm(X, y=NULL,
                 type='one-classification',
                 nu=el,
                 kernel="radial")
  
  svm.predtrain<-predict(svm.model, X)
  out <- rep(0, n)
  out[svm.predtrain==FALSE] <- 1
  roc_lof<- roc(response = is_out, predictor = out)
  auc_svm <- c(auc_svm ,auc(roc_lof))
  confusion_matrix<-table(SVM=out, Actual=is_out)
  precision <- c(precision, confusion_matrix[2,2]/sum(confusion_matrix[2,]))
  recall <- c(recall, confusion_matrix[2,2]/sum(confusion_matrix[,2]))
}
auc_svm; precision; recall

#BEST SVM NU: nu = 0.01 ; AUC = 0.628 ; precision = 0.158 ; recall = 0.333

# OCSVM paper with k=7 and k=Best knn value:
#K=7:
distanze<-kNNdist(X,k=7)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))

svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.675
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.08
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.667

#K=BEST knn:
distanze<-kNNdist(X,k=8)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.6314
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.08
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.55

#BEST SVM K: k = 7 ; AUC = 0.675 ; precision = 0.08 ; recall = 0.667

#Glass SCALATO -----
X <- scale(X)
### LOF:
outlier.scores <- lof(X, k=mean(c(log(n), p+1, 2*p)))

rfind <- rpart_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=18)
abline(h=rfind, col=4)
rbest_taglio <- sort(rfind)[7] #2.73
out_previsti <- rep(0,n)
out_previsti <- ifelse(outlier.scores>rbest_taglio, 1, 0)

#Confusion Matrix & AUC:
table(LOF=out_previsti, Actual=is_out)
#Valutazione performance:
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.5989
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.2857
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.222

cfind <- ctree_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=19)
abline(h=cfind, col=cfind+1)
cbest_taglio <- sort(cfind)[1] 
out_previsti=rep(0,n)
out_previsti <- ifelse(outlier.scores>cbest_taglio, 1, 0)
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.5745
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.118
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.222

#BEST LOF: taglio = 2.9 ; AUC = 0.5989 ; precision = 0.286 ; recall = 0.222

### SVM:

# nu with 4 values:
nu_val <- c(0.01, 0.1, 0.9, 0.5)
auc_svm <- c()
precision <- c()
recall <- c()
for (el in nu_val){
  svm.model<-svm(X, y=NULL,
                 type='one-classification',
                 nu=el,
                 kernel="radial")
  
  svm.predtrain<-predict(svm.model, X)
  out <- rep(0, n)
  out[svm.predtrain==FALSE] <- 1
  roc_lof<- roc(response = is_out, predictor = out)
  auc_svm <- c(auc_svm ,auc(roc_lof))
  confusion_matrix<-table(SVM=out, Actual=is_out)
  precision <- c(precision, confusion_matrix[2,2]/sum(confusion_matrix[2,]))
  recall <- c(recall, confusion_matrix[2,2]/sum(confusion_matrix[,2]))
}
auc_svm; precision; recall

#BEST SVM NU: nu = 0.01 ; AUC = 0.628 ; precision = 0.158 ; recall = 0.333

# OCSVM paper with k=7 and k=Best knn value:
#K=7:
distanze<-kNNdist(X,k=7)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-(sum(SK)-x.max_cf)/sum(SK)
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.611
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.115
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.33

#K=BEST knn:
distanze<-kNNdist(X,k=10)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-(sum(SK)-x.max_cf)/sum(SK)
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.632
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.097
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.44

#BEST SVM K: k = 10 ; AUC = 0.632 ; precision = 0.097 ; recall = 0.44

# Vowels - 50 outliers -------------------------
data=readMat("vowels.mat")
X=as.data.frame(data$X)
p=dim(X)[2]; n=dim(X)[1]
is_out=data$y
dati<-data.frame(cbind(X, is_out)) 

### LOF:
outlier.scores <- lof(X, k=mean(c(log(n), p+1, 2*p)))

rfind <- rpart_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=18)
abline(h=rfind, col=4)
rbest_taglio <- sort(rfind)[5] #1.36
out_previsti <- rep(0,n)
out_previsti <- ifelse(outlier.scores>rbest_taglio, 1, 0)

#Confusion Matrix & AUC:
table(LOF=out_previsti, Actual=is_out)
#Valutazione performance:
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.6686
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.36
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.36

cfind <- ctree_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=19)
abline(h=cfind, col=cfind+1)
cbest_taglio <- sort(cfind)[3] 
out_previsti=rep(0,n)
out_previsti <- ifelse(outlier.scores>cbest_taglio, 1, 0)
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.669
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.367
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.36

#BEST LOF: taglio = 1.36 ; AUC = 0.669 ; precision = 0.367 ; recall = 0.36

### SVM:

# nu with 4 values:
nu_val <- c(0.01, 0.1, 0.9, 0.5)
auc_svm <- c()
precision <- c()
recall <- c()
for (el in nu_val){
  svm.model<-svm(X, y=NULL,
                 type='one-classification',
                 nu=el,
                 kernel="radial")
  
  svm.predtrain<-predict(svm.model, X)
  out <- rep(0, n)
  out[svm.predtrain==FALSE] <- 1
  roc_lof<- roc(response = is_out, predictor = out)
  auc_svm <- c(auc_svm ,auc(roc_lof))
  confusion_matrix<-table(SVM=out, Actual=is_out)
  precision <- c(precision, confusion_matrix[2,2]/sum(confusion_matrix[2,]))
  recall <- c(recall, confusion_matrix[2,2]/sum(confusion_matrix[,2]))
}
auc_svm; precision; recall

#BEST SVM NU: nu = 0.5 ; AUC = 0.708 ; precision = 0.062 ; recall = 0.90

# OCSVM paper with k=7 and k=Best knn value:
#K=7:
distanze<-kNNdist(X,k=7)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))

svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.735
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.104
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.68

#K=BEST knn:
distanze<-kNNdist(X,k=8)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.755
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.102
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.74

#BEST SVM K: k = 8 ; AUC = 0.755 ; precision = 0.102 ; recall = 0.74

# Cardio - 176 outliers ------------------------
data=readMat("cardio.mat")
X <- data$X
p=dim(X)[2]; n=dim(X)[1]
is_out <- data$y
dati<-data.frame(cbind(X, is_out)) 

### LOF:
outlier.scores <- lof(X, k=mean(c(log(n), p+1, 2*p)))

rfind <- rpart_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=18)
abline(h=rfind, col=4)
rbest_taglio <- sort(rfind)[5] #3.03 forse meglio 2.9
rbest_taglio <- 2.9
out_previsti <- rep(0,n)
out_previsti <- ifelse(outlier.scores>rbest_taglio, 1, 0)

#Confusion Matrix & AUC:
table(LOF=out_previsti, Actual=is_out)
#Valutazione performance:
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.516
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.75
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.034

cfind <- ctree_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=19)
abline(h=cfind, col=cfind+1) #nessun taglio
# cbest_taglio <- sort(cfind)[1] 
# out_previsti=rep(0,n)
# out_previsti <- ifelse(outlier.scores>cbest_taglio, 1, 0)
# roc_lof<- roc(response = is_out, predictor = out_previsti)
# auc(roc_lof) #0.5867
# confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
# precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.167
# recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.22

#BEST LOF: taglio = 2.9 ; AUC = 0.516 ; precision = 0.75 ; recall = 0.034

### SVM:

# nu with 4 values:
nu_val <- c(0.01, 0.1, 0.9, 0.5)
auc_svm <- c()
precision <- c()
recall <- c()
for (el in nu_val){
  svm.model<-svm(X, y=NULL,
                 type='one-classification',
                 nu=el,
                 kernel="radial")
  
  svm.predtrain<-predict(svm.model, X)
  out <- rep(0, n)
  out[svm.predtrain==FALSE] <- 1
  roc_lof<- roc(response = is_out, predictor = out)
  auc_svm <- c(auc_svm ,auc(roc_lof))
  confusion_matrix<-table(SVM=out, Actual=is_out)
  precision <- c(precision, confusion_matrix[2,2]/sum(confusion_matrix[2,]))
  recall <- c(recall, confusion_matrix[2,2]/sum(confusion_matrix[,2]))
}
auc_svm; precision; recall

#BEST SVM NU: nu = 0.5 ; AUC = 0.776 ; precision = 0.192 ; recall = 1.00

# OCSVM paper with k=7 and k=Best knn value:
#K=7:
distanze<-kNNdist(X,k=7)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))

svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.7897
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.207
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.983

#K=BEST knn:
distanze<-kNNdist(X,k=15)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.7339
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.172
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.96

#BEST SVM K: k = 7 ; AUC = 0.7897 ; precision = 0.208 ; recall = 0.983

# Thyroid - 93 outliers ------------------------
data=readMat("thyroid.mat")
X <- data$X
p=dim(X)[2]; n=dim(X)[1]
is_out=data$y
dati<-data.frame(cbind(X, is_out)) 

### LOF:
outlier.scores <- lof(X, k=mean(c(log(n), p+1, 2*p)))

rfind <- rpart_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=18)
abline(h=rfind, col=4) #nessun taglio sembra ottimale, proviamo a tagliare al ginocchio
abline(h=1.55, col=2)
rbest_taglio <- 1.55
out_previsti <- rep(0,n)
out_previsti <- ifelse(outlier.scores>rbest_taglio, 1, 0)

#Confusion Matrix & AUC:
table(LOF=out_previsti, Actual=is_out)
#Valutazione performance:
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.5531
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.079
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.15

cfind <- ctree_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=19)
abline(h=cfind, col=cfind+1) #nessun taglio
# cbest_taglio <- sort(cfind)[1] 
# out_previsti=rep(0,n)
# out_previsti <- ifelse(outlier.scores>cbest_taglio, 1, 0)
# roc_lof<- roc(response = is_out, predictor = out_previsti)
# auc(roc_lof) #0.5867
# confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
# precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision 
# recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall 

#BEST LOF: taglio = 1.55 ; AUC = 0.5531 ; precision = 0.079 ; recall = 0.15

### SVM:

# nu with 4 values:
nu_val <- c(0.01, 0.1, 0.9, 0.5)
auc_svm <- c()
precision <- c()
recall <- c()
for (el in nu_val){
  svm.model<-svm(X, y=NULL,
                 type='one-classification',
                 nu=el,
                 kernel="radial")
  
  svm.predtrain<-predict(svm.model, X)
  out <- rep(0, n)
  out[svm.predtrain==FALSE] <- 1
  roc_lof<- roc(response = is_out, predictor = out)
  auc_svm <- c(auc_svm ,auc(roc_lof))
  confusion_matrix<-table(SVM=out, Actual=is_out)
  precision <- c(precision, confusion_matrix[2,2]/sum(confusion_matrix[2,]))
  recall <- c(recall, confusion_matrix[2,2]/sum(confusion_matrix[,2]))
}
auc_svm; precision; recall

#BEST SVM NU: nu = 0.1 ; AUC = 0.857 ; precision = 0.197 ; recall = 0.796

# OCSVM paper with k=7 and k=Best knn value:
#K=7:
distanze<-kNNdist(X,k=7)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))
nu <- 0.5 #perchè assume un valore sopra a 1
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.7398
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.05
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.967

#K=BEST knn:
#k=7 come nel caso precedente

#BEST SVM K: k = / ; AUC = 0.7398 ; precision = 0.05 ; recall = 0.967

### Thyroid COMPLETO -----
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

### LOF:
outlier.scores <- lof(X, k=mean(c(log(n), p+1, 2*p)))

rfind <- rpart_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=18)
abline(h=rfind, col=4)
rbest_taglio <- min(rfind)
out_previsti <- rep(0,n)
out_previsti <- ifelse(outlier.scores>rbest_taglio, 1, 0)

#Confusion Matrix & AUC:
table(LOF=out_previsti, Actual=is_out)
#Valutazione performance:
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.4987
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.02
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.02

cfind <- ctree_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=19)
abline(h=cfind, col=cfind+1)
cbest_taglio <- sort(cfind)[1] 
out_previsti=rep(0,n)
out_previsti <- ifelse(outlier.scores>cbest_taglio, 1, 0)
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.4952
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.01
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.01

#BEST LOF: taglio = 20 ; AUC = 0.4987 ; precision = 0.02 ; recall = 0.02

### SVM:

# nu with 4 values:
nu_val <- c(0.01, 0.1, 0.9, 0.5)
auc_svm <- c()
precision <- c()
recall <- c()
for (el in nu_val){
  svm.model<-svm(X, y=NULL,
                 type='one-classification',
                 nu=el,
                 kernel="radial")
  
  svm.predtrain<-predict(svm.model, X)
  out <- rep(0, n)
  out[svm.predtrain==FALSE] <- 1
  roc_lof<- roc(response = is_out, predictor = out)
  auc_svm <- c(auc_svm ,auc(roc_lof))
  confusion_matrix<-table(SVM=out, Actual=is_out)
  precision <- c(precision, confusion_matrix[2,2]/sum(confusion_matrix[2,]))
  recall <- c(recall, confusion_matrix[2,2]/sum(confusion_matrix[,2]))
}
auc_svm; precision; recall

#BEST SVM NU: nu = 0.5 ; AUC = 0.6627 ; precision = 0.04 ; recall = 0.81

# OCSVM paper with k=7 and k=Best knn value:
#K=7:
distanze<-kNNdist(X,k=7)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK)) #supera 1 quindi teniamo valore di default
nu <- 0.5
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.6075
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.035
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.709

#K=BEST knn:
distanze<-kNNdist(X,k=8)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))
nu <- 0.05
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.4807
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.008
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.02

#BEST SVM K: k = 7 ; AUC = 0.6075 ; precision = 0.035 ; recall = 0.709

### Thyroid COMPLETO SCALATO ----
X <- scale(X)
dati<-data.frame(cbind(X, is_out)) 

### LOF:
outlier.scores <- lof(X, k=mean(c(log(n), p+1, 2*p)))

rfind <- rpart_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=18)
abline(h=rfind, col=4) #poco sensato
abline(h=9.5, col=2) #trovato graficamente
rbest_taglio <- 9.5
out_previsti <- rep(0,n)
out_previsti <- ifelse(outlier.scores>rbest_taglio, 1, 0)

#Confusion Matrix & AUC:
table(LOF=out_previsti, Actual=is_out)
#Valutazione performance:
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.5111
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.031
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.107

cfind <- ctree_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=19)
abline(h=cfind, col=cfind+1)
cbest_taglio <- sort(cfind)[1] 
out_previsti=rep(0,n)
out_previsti <- ifelse(outlier.scores>cbest_taglio, 1, 0)
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.4952
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.013
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.011

#BEST LOF: taglio = 9.5 ; AUC = 0.5111 ; precision = 0.031 ; recall = 0.107

### SVM:

# nu with 4 values:
nu_val <- c(0.01, 0.1, 0.9, 0.5)
auc_svm <- c()
precision <- c()
recall <- c()
for (el in nu_val){
  svm.model<-svm(X, y=NULL,
                 type='one-classification',
                 nu=el,
                 kernel="radial")
  
  svm.predtrain<-predict(svm.model, X)
  out <- rep(0, n)
  out[svm.predtrain==FALSE] <- 1
  roc_lof<- roc(response = is_out, predictor = out)
  auc_svm <- c(auc_svm ,auc(roc_lof))
  confusion_matrix<-table(SVM=out, Actual=is_out)
  precision <- c(precision, confusion_matrix[2,2]/sum(confusion_matrix[2,]))
  recall <- c(recall, confusion_matrix[2,2]/sum(confusion_matrix[,2]))
}
auc_svm; precision; recall

#BEST SVM NU: nu = 0.5 ; AUC = 0.663 ; precision = 0.04 ; recall = 0.817

# OCSVM paper with k=7 and k=Best knn value:
#K=7:
distanze<-kNNdist(X,k=7)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))
nu <- 0.05 #perchè era maggiore a 1
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.4717
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0

#K=BEST knn:
distanze<-kNNdist(X,k=8)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))
nu <- 0.05
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.4807
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.008
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.02

#BEST SVM K: k = 8 ; AUC = 0.4807 ; precision = 0.08 ; recall = 0.02

# Musk - 97 outliers ---------------------------
data=readMat("musk.mat")
X=as.data.frame(data$X)
p=dim(X)[2]; n=dim(X)[1]
is_out=data$y
dati<-data.frame(cbind(X, is_out)) 

### LOF:
outlier.scores <- lof(X, k=mean(c(log(n), p+1, 2*p)))

rfind <- rpart_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=18)
abline(h=rfind, col=4)
rbest_taglio <- sort(rfind)[6]
out_previsti <- rep(0,n)
out_previsti <- ifelse(outlier.scores>rbest_taglio, 1, 0)

#Confusion Matrix & AUC:
table(LOF=out_previsti, Actual=is_out)
#Valutazione performance:
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #1
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #1
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #1

cfind <- ctree_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=19)
abline(h=cfind, col=cfind+1)
cbest_taglio <- sort(cfind)[5] 
out_previsti=rep(0,n)
out_previsti <- ifelse(outlier.scores>cbest_taglio, 1, 0)
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #1
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #1
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #1

#BEST LOF: taglio = 1.23 ; AUC = 1 ; precision = 1 ; recall = 1

### SVM:

# nu with 4 values:
nu_val <- c(0.01, 0.1, 0.9, 0.5)
auc_svm <- c()
precision <- c()
recall <- c()
for (el in nu_val){
  svm.model<-svm(X, y=NULL,
                 type='one-classification',
                 nu=el,
                 kernel="radial")
  
  svm.predtrain<-predict(svm.model, X)
  out <- rep(0, n)
  out[svm.predtrain==FALSE] <- 1
  roc_lof<- roc(response = is_out, predictor = out)
  auc_svm <- c(auc_svm ,auc(roc_lof))
  confusion_matrix<-table(SVM=out, Actual=is_out)
  precision <- c(precision, confusion_matrix[2,2]/sum(confusion_matrix[2,]))
  recall <- c(recall, confusion_matrix[2,2]/sum(confusion_matrix[,2]))
}
auc_svm; precision; recall

#BEST SVM NU: nu = 0.1 ; AUC = 0.857 ; precision = 0.247 ; recall = 0.794

# OCSVM paper with k=7 and k=Best knn value:
#K=7:
distanze<-kNNdist(X,k=7)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))

svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.3711
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.023
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.742

#K=BEST knn:
distanze<-kNNdist(X,k=167)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.4124
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.027
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.825

#BEST SVM K: k = 167 ; AUC = 0.4124 ; precision = 0.027 ; recall = 0.825

#Musk SCALATO -----
X <- scale(X)
### LOF:
outlier.scores <- lof(X, k=mean(c(log(n), p+1, 2*p)))

rfind <- rpart_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=18)
abline(h=rfind, col=4)
rbest_taglio <- sort(rfind)[8] #1.39
out_previsti <- rep(0,n)
out_previsti <- ifelse(outlier.scores>rbest_taglio, 1, 0)

#Confusion Matrix & AUC:
table(LOF=out_previsti, Actual=is_out)
#Valutazione performance:
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #1
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #1
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #1

cfind <- ctree_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=19)
abline(h=cfind, col=cfind+1)
cbest_taglio <- max(cfind)
out_previsti=rep(0,n)
out_previsti <- ifelse(outlier.scores>cbest_taglio, 1, 0)
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #1
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #1
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #1

#BEST LOF: taglio = 1.39 ; AUC = 1 ; precision = 1 ; recall = 1

### SVM:

# nu with 4 values:
nu_val <- c(0.01, 0.1, 0.9, 0.5)
auc_svm <- c()
precision <- c()
recall <- c()
for (el in nu_val){
  svm.model<-svm(X, y=NULL,
                 type='one-classification',
                 nu=el,
                 kernel="radial")
  
  svm.predtrain<-predict(svm.model, X)
  out <- rep(0, n)
  out[svm.predtrain==FALSE] <- 1
  roc_lof<- roc(response = is_out, predictor = out)
  auc_svm <- c(auc_svm ,auc(roc_lof))
  confusion_matrix<-table(SVM=out, Actual=is_out)
  precision <- c(precision, confusion_matrix[2,2]/sum(confusion_matrix[2,]))
  recall <- c(recall, confusion_matrix[2,2]/sum(confusion_matrix[,2]))
}
auc_svm; precision; recall

#BEST SVM NU: nu = 0.1 ; AUC = 0.857 ; precision = 0.248 ; recall = 0.794

# OCSVM paper with k=7 and k=Best knn value:
#K=7:
distanze<-kNNdist(X,k=7)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-(sum(SK)-x.max_cf)/sum(SK)
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.6589
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.02
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.41

#K=BEST knn:
distanze<-kNNdist(X,k=167)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-(sum(SK)-x.max_cf)/sum(SK)
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.5533
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.035
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #1

#BEST SVM K: k = 7 ; AUC = 0.6589 ; precision = 0.02 ; recall = 0.41

# Satimage-2 - 71 outliers ---------------------
data=readMat("satimage-2.mat")
X=as.data.frame(data$X)
p=dim(X)[2]; n=dim(X)[1]
is_out=data$y
dati<-data.frame(cbind(X, is_out)) 

### LOF:
outlier.scores <- lof(X, k=mean(c(log(n), p+1, 2*p)))

rfind <- rpart_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=18)
abline(h=rfind, col=4)
rbest_taglio <- max(rfind)
out_previsti <- rep(0,n)
out_previsti <- ifelse(outlier.scores>rbest_taglio, 1, 0)

#Confusion Matrix & AUC:
table(LOF=out_previsti, Actual=is_out)
#Valutazione performance:
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.5433
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.051
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.112

cfind <- ctree_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=19)
abline(h=cfind, col=cfind+1) #nessun taglio
# cbest_taglio <- sort(cfind)[5] 
# out_previsti=rep(0,n)
# out_previsti <- ifelse(outlier.scores>cbest_taglio, 1, 0)
# roc_lof<- roc(response = is_out, predictor = out_previsti)
# auc(roc_lof) 
# confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
# precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision 
# recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall 

#BEST LOF: taglio = 1.56 ; AUC = 0.5433 ; precision = 0.051 ; recall = 0.112

### SVM:

# nu with 4 values:
nu_val <- c(0.01, 0.1, 0.9, 0.5)
auc_svm <- c()
precision <- c()
recall <- c()
for (el in nu_val){
  svm.model<-svm(X, y=NULL,
                 type='one-classification',
                 nu=el,
                 kernel="radial")
  
  svm.predtrain<-predict(svm.model, X)
  out <- rep(0, n)
  out[svm.predtrain==FALSE] <- 1
  roc_lof<- roc(response = is_out, predictor = out)
  auc_svm <- c(auc_svm ,auc(roc_lof))
  confusion_matrix<-table(SVM=out, Actual=is_out)
  precision <- c(precision, confusion_matrix[2,2]/sum(confusion_matrix[2,]))
  recall <- c(recall, confusion_matrix[2,2]/sum(confusion_matrix[,2]))
}
auc_svm; precision; recall

#BEST SVM NU: nu = 0.1 ; AUC = 0.948 ; precision = 0.12 ; recall = 0.986

# OCSVM paper with k=7 and k=Best knn value:
#K=7:
distanze<-kNNdist(X,k=7)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))

svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.5304
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.013
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #1.00

#K=BEST knn:
distanze<-kNNdist(X,k=37)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.5318
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.013
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #1

#BEST SVM K: k = 7 ; AUC = 0.531 ; precision = 0.013 ; recall = 1

#Satimage-2 SCALATO -----
X <- scale(X)
### LOF:
outlier.scores <- lof(X, k=mean(c(log(n), p+1, 2*p)))

rfind <- rpart_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=18)
abline(h=rfind, col=4)
rbest_taglio <- max(rfind) #1.6
out_previsti <- rep(0,n)
out_previsti <- ifelse(outlier.scores>rbest_taglio, 1, 0)

#Confusion Matrix & AUC:
table(LOF=out_previsti, Actual=is_out)
#Valutazione performance:
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.5532
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.071
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.127

cfind <- ctree_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=19)
abline(h=cfind, col=cfind+1)
# cbest_taglio <- sort(cfind)[1] 
# out_previsti=rep(0,n)
# out_previsti <- ifelse(outlier.scores>cbest_taglio, 1, 0)
# roc_lof<- roc(response = is_out, predictor = out_previsti)
# auc(roc_lof) 
# confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
# precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision 
# recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall 

#BEST LOF: taglio = 1.6 ; AUC = 0.5532 ; precision = 0.071 ; recall = 0.127

### SVM:

# nu with 4 values:
nu_val <- c(0.01, 0.1, 0.9, 0.5)
auc_svm <- c()
precision <- c()
recall <- c()
for (el in nu_val){
  svm.model<-svm(X, y=NULL,
                 type='one-classification',
                 nu=el,
                 kernel="radial")
  
  svm.predtrain<-predict(svm.model, X)
  out <- rep(0, n)
  out[svm.predtrain==FALSE] <- 1
  roc_lof<- roc(response = is_out, predictor = out)
  auc_svm <- c(auc_svm ,auc(roc_lof))
  confusion_matrix<-table(SVM=out, Actual=is_out)
  precision <- c(precision, confusion_matrix[2,2]/sum(confusion_matrix[2,]))
  recall <- c(recall, confusion_matrix[2,2]/sum(confusion_matrix[,2]))
}
auc_svm; precision; recall

#BEST SVM NU: nu = 0.1 ; AUC = 0.948 ; precision = 0.121 ; recall = 0.986

# OCSVM paper with k=7 and k=Best knn value:
#K=7:
distanze<-kNNdist(X,k=7)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-(sum(SK)-x.max_cf)/sum(SK)
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.8186
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.033
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #1

#K=BEST knn:
distanze<-kNNdist(X,k=37)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-(sum(SK)-x.max_cf)/sum(SK)
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.7674
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.026
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #1

#BEST SVM K: k = 7 ; AUC = 0.8186 ; precision = 0.033 ; recall = 1.00

# Letter Recognition - 100 outliers ------------
path <- "letter.mat"
data=readMat(path)
X=data$X
p <- dim(X)[2]; n <- dim(X)[1]
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
# dati_csv <- read.csv("letter.csv", sep=",", header=F)
# X=as.data.frame(data$X)
# p=dim(X)[2]; n=dim(X)[1]
# is_out=data$y
# dati<-data.frame(cbind(X, is_out)) 

### LOF:
outlier.scores <- lof(X, k=mean(c(log(n), p+1, 2*p)))

rfind <- rpart_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=18)
abline(h=rfind, col=4)
rbest_taglio <- sort(rfind)[2]
out_previsti <- rep(0,n)
out_previsti <- ifelse(outlier.scores>rbest_taglio, 1, 0)

#Confusion Matrix & AUC:
table(LOF=out_previsti, Actual=is_out)
#Valutazione performance:
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.655
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.524
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.33

cfind <- ctree_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=19)
abline(h=cfind, col=cfind+1) 
cbest_taglio <- sort(cfind)[1]
out_previsti=rep(0,n)
out_previsti <- ifelse(outlier.scores>cbest_taglio, 1, 0)
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.693
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.394
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.43

#BEST LOF: taglio = 1.15346 ; AUC = 0.693 ; precision = 0.394 ; recall = 0.43

### SVM:

# nu with 4 values:
nu_val <- c(0.01, 0.1, 0.9, 0.5)
auc_svm <- c()
precision <- c()
recall <- c()
for (el in nu_val){
  svm.model<-svm(X, y=NULL,
                 type='one-classification',
                 nu=el,
                 kernel="radial")
  
  svm.predtrain<-predict(svm.model, X)
  out <- rep(0, n)
  out[svm.predtrain==FALSE] <- 1
  roc_lof<- roc(response = is_out, predictor = out)
  auc_svm <- c(auc_svm ,auc(roc_lof))
  confusion_matrix<-table(SVM=out, Actual=is_out)
  precision <- c(precision, confusion_matrix[2,2]/sum(confusion_matrix[2,]))
  recall <- c(recall, confusion_matrix[2,2]/sum(confusion_matrix[,2]))
}
auc_svm; precision; recall

#BEST SVM NU: nu = 0.1 ; AUC = 0.583 ; precision = 0.157 ; recall = 0.93

# OCSVM paper with k=7 and k=Best knn value:
#K=7:
distanze<-kNNdist(X,k=7)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK)); nu
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.596
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.076
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #1

#K=BEST knn:
#k=7

#BEST SVM K: k = 7 ; AUC =0.596 ; precision = 0.076 ; recall = 1

#Letter SCALATO -----
X <- scale(X)
### LOF:
outlier.scores <- lof(X, k=mean(c(log(n), p+1, 2*p)))

rfind <- rpart_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=18)
abline(h=rfind, col=4)
rbest_taglio <- sort(rfind)[length(rfind)-1] #1.3
out_previsti <- rep(0,n)
out_previsti <- ifelse(outlier.scores>rbest_taglio, 1, 0)

#Confusion Matrix & AUC:
table(LOF=out_previsti, Actual=is_out)
#Valutazione performance:
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.5723
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.652
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.15

cfind <- ctree_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=19)
abline(h=cfind, col=cfind+1)
cbest_taglio <- max(cfind)
out_previsti=rep(0,n)
out_previsti <- ifelse(outlier.scores>cbest_taglio, 1, 0)
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.6043
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.564
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.22

#BEST LOF: taglio = 1.258 ; AUC = 0.6043 ; precision = 0.564 ; recall = 0.22

### SVM:

# nu with 4 values:
nu_val <- c(0.01, 0.1, 0.9, 0.5)
auc_svm <- c()
precision <- c()
recall <- c()
for (el in nu_val){
  svm.model<-svm(X, y=NULL,
                 type='one-classification',
                 nu=el,
                 kernel="radial")
  
  svm.predtrain<-predict(svm.model, X)
  out <- rep(0, n)
  out[svm.predtrain==FALSE] <- 1
  roc_lof<- roc(response = is_out, predictor = out)
  auc_svm <- c(auc_svm ,auc(roc_lof))
  confusion_matrix<-table(SVM=out, Actual=is_out)
  precision <- c(precision, confusion_matrix[2,2]/sum(confusion_matrix[2,]))
  recall <- c(recall, confusion_matrix[2,2]/sum(confusion_matrix[,2]))
}
auc_svm; precision; recall

#BEST SVM NU: nu = 0.1 ; AUC = 0.583 ; precision = 0.157 ; recall = 0.26

# OCSVM paper with k=7 and k=Best knn value:
#K=7:
distanze<-kNNdist(X,k=7)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-(sum(SK)-x.max_cf)/sum(SK)
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.6677
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.092
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.97

#K=BEST knn:
#k=7

#BEST SVM K: k = 7 ; AUC = 0.6677 ; precision = 0.092 ; recall = 0.97

# Speech - 61 outliers -------------------------
path <- "speech.mat"
data=readMat(path)
X=data$X
p <- dim(X)[2]; n <- dim(X)[1]
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))

### LOF:
outlier.scores <- lof(X, k=mean(c(log(n), p+1, 2*p)))

rfind <- rpart_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=18)
abline(h=rfind, col=4)
abline(h=1.27, col=2) #buon senso
rbest_taglio <- 1.27
out_previsti <- rep(0,n)
out_previsti <- ifelse(outlier.scores>rbest_taglio, 1, 0)

#Confusion Matrix & AUC:
table(LOF=out_previsti, Actual=is_out)
#Valutazione performance:
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.5051
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.024
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.033

cfind <- ctree_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=19)
abline(h=cfind, col=cfind+1) 
cbest_taglio <- sort(cfind)[length(cfind)]
out_previsti=rep(0,n)
out_previsti <- ifelse(outlier.scores>cbest_taglio, 1, 0)
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.5197
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.025
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.115

#BEST LOF: taglio =1.22825 ; AUC = 0.5197 ; precision = 0.025 ; recall = 0.115

### SVM:

# nu with 4 values:
nu_val <- c(0.01, 0.1, 0.9, 0.5)
auc_svm <- c()
precision <- c()
recall <- c()
for (el in nu_val){
  svm.model<-svm(X, y=NULL,
                 type='one-classification',
                 nu=el,
                 kernel="radial")
  
  svm.predtrain<-predict(svm.model, X)
  out <- rep(0, n)
  out[svm.predtrain==FALSE] <- 1
  roc_lof<- roc(response = is_out, predictor = out)
  auc_svm <- c(auc_svm ,auc(roc_lof))
  confusion_matrix<-table(SVM=out, Actual=is_out)
  precision <- c(precision, confusion_matrix[2,2]/sum(confusion_matrix[2,]))
  recall <- c(recall, confusion_matrix[2,2]/sum(confusion_matrix[,2]))
}
auc_svm; precision; recall

#BEST SVM NU: nu = 0.5 ; AUC = 0.5292 ; precision = 0.015 ; recall = 0.442

# OCSVM paper with k=7 and k=Best knn value:
#K=7:
distanze<-kNNdist(X,k=7)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK)); nu
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.4877
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.015
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.311

#K=BEST knn:
distanze<-kNNdist(X,k=8)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.4706
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.015
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.606

#BEST SVM K: k = 7 ; AUC = 0.4877 ; precision = 0.015 ; recall = 0.311

#Speech SCALATO -----
X <- scale(X)
### LOF:
outlier.scores <- lof(X, k=mean(c(log(n), p+1, 2*p)))

rfind <- rpart_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=18)
abline(h=rfind, col=4)
rbest_taglio <- sort(rfind)[length(rfind)] #1.2
out_previsti <- rep(0,n)
out_previsti <- ifelse(outlier.scores>rbest_taglio, 1, 0)

#Confusion Matrix & AUC:
table(LOF=out_previsti, Actual=is_out)
#Valutazione performance:
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.5143
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.019
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.18

cfind <- ctree_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=19)
abline(h=cfind, col=cfind+1)
cbest_taglio <- max(cfind) #1.22736
out_previsti=rep(0,n)
out_previsti <- ifelse(outlier.scores>cbest_taglio, 1, 0)
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.5188
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.024
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.115

#BEST LOF: taglio = 1.22736 ; AUC = 0.5188 ; precision = 0.024 ; recall = 0.115

### SVM:

# nu with 4 values:
nu_val <- c(0.01, 0.1, 0.9, 0.5)
auc_svm <- c()
precision <- c()
recall <- c()
for (el in nu_val){
  svm.model<-svm(X, y=NULL,
                 type='one-classification',
                 nu=el,
                 kernel="radial")
  
  svm.predtrain<-predict(svm.model, X)
  out <- rep(0, n)
  out[svm.predtrain==FALSE] <- 1
  roc_lof<- roc(response = is_out, predictor = out)
  auc_svm <- c(auc_svm ,auc(roc_lof))
  confusion_matrix<-table(SVM=out, Actual=is_out)
  precision <- c(precision, confusion_matrix[2,2]/sum(confusion_matrix[2,]))
  recall <- c(recall, confusion_matrix[2,2]/sum(confusion_matrix[,2]))
}
auc_svm; precision; recall

#BEST SVM NU: nu = 0.5 ; AUC = 0.5292 ; precision = 0.015 ; recall = 0.443

# OCSVM paper with k=7 and k=Best knn value:
#K=7:
distanze<-kNNdist(X,k=7)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-(sum(SK)-x.max_cf)/sum(SK)
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.5073
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.017
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.426

#K=BEST knn:
distanze<-kNNdist(X,k=8)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-(sum(SK)-x.max_cf)/sum(SK)
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.5663
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.027
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.328

#BEST SVM K: k = 8 ; AUC = 0.5663 ; precision = 0.027 ; recall = 0.328

# Pima - 268 outliers --------------------------
path <- "pima.mat"
data=readMat(path)
X=data$X
p <- dim(X)[2]; n <- dim(X)[1]
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))

### LOF:
outlier.scores <- lof(X, k=mean(c(log(n), p+1, 2*p)))

rfind <- rpart_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=18)
abline(h=rfind, col=4)
abline(h=1.6, col=2) #buon senso
rbest_taglio <- 1.6
out_previsti <- rep(0,n)
out_previsti <- ifelse(outlier.scores>rbest_taglio, 1, 0)

#Confusion Matrix & AUC:
table(LOF=out_previsti, Actual=is_out)
#Valutazione performance:
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.5028
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.027
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.033

cfind <- ctree_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=19)
abline(h=cfind, col=cfind+1)#nessun taglio
# cbest_taglio <- sort(cfind)[1]
# out_previsti=rep(0,n)
# out_previsti <- ifelse(outlier.scores>cbest_taglio, 1, 0)
# roc_lof<- roc(response = is_out, predictor = out_previsti)
# auc(roc_lof) #
# confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
# precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #
# recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #

#BEST LOF: taglio = 1.6 ; AUC = 0.5028 ; precision = 0.027 ; recall = 0.033

### SVM:

# nu with 4 values:
nu_val <- c(0.01, 0.1, 0.9, 0.5)
auc_svm <- c()
precision <- c()
recall <- c()
for (el in nu_val){
  svm.model<-svm(X, y=NULL,
                 type='one-classification',
                 nu=el,
                 kernel="radial")
  
  svm.predtrain<-predict(svm.model, X)
  out <- rep(0, n)
  out[svm.predtrain==FALSE] <- 1
  roc_lof<- roc(response = is_out, predictor = out)
  auc_svm <- c(auc_svm ,auc(roc_lof))
  confusion_matrix<-table(SVM=out, Actual=is_out)
  precision <- c(precision, confusion_matrix[2,2]/sum(confusion_matrix[2,]))
  recall <- c(recall, confusion_matrix[2,2]/sum(confusion_matrix[,2]))
}
auc_svm; precision; recall

#BEST SVM NU: nu = 0.5 ; AUC = 0.5936 ; precision = 0.434 ; recall = 0.623

# OCSVM paper with k=7 and k=Best knn value:
#K=7:
distanze<-kNNdist(X,k=7)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK)); nu
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.5067
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.352
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.951

#K=BEST knn:
#k=7

#BEST SVM K: k = 7 ; AUC = 0.5067 ; precision = 0.352 ; recall = 0.951

#Pima SCALATO -----
X <- scale(X)
### LOF:
outlier.scores <- lof(X, k=mean(c(log(n), p+1, 2*p)))

rfind <- rpart_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=18)
abline(h=rfind, col=4)
rbest_taglio <- sort(rfind)[4] #1.30 ginocchio
out_previsti <- rep(0,n)
out_previsti <- ifelse(outlier.scores>rbest_taglio, 1, 0)

#Confusion Matrix & AUC:
table(LOF=out_previsti, Actual=is_out)
#Valutazione performance:
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.5112
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.429
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.078

cfind <- ctree_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=19)
abline(h=cfind, col=cfind+1)#nessun taglio

#BEST LOF: taglio = 1.3 ; AUC = 0.5112 ; precision = 0.429 ; recall = 0.078

### SVM:

# nu with 4 values:
nu_val <- c(0.01, 0.1, 0.9, 0.5)
auc_svm <- c()
precision <- c()
recall <- c()
for (el in nu_val){
  svm.model<-svm(X, y=NULL,
                 type='one-classification',
                 nu=el,
                 kernel="radial")
  
  svm.predtrain<-predict(svm.model, X)
  out <- rep(0, n)
  out[svm.predtrain==FALSE] <- 1
  roc_lof<- roc(response = is_out, predictor = out)
  auc_svm <- c(auc_svm ,auc(roc_lof))
  confusion_matrix<-table(SVM=out, Actual=is_out)
  precision <- c(precision, confusion_matrix[2,2]/sum(confusion_matrix[2,]))
  recall <- c(recall, confusion_matrix[2,2]/sum(confusion_matrix[,2]))
}
auc_svm; precision; recall

#BEST SVM NU: nu = 0.5 ; AUC = 0.5936 ; precision = 0.434 ; recall = 0.623

# OCSVM paper with k=7 and k=Best knn value:
#K=7:
distanze<-kNNdist(X,k=7)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-(sum(SK)-x.max_cf)/sum(SK)
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.5787
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.483
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.369

#K=BEST knn:
distanze<-kNNdist(X,k=8)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-(sum(SK)-x.max_cf)/sum(SK)
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.5723
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.464
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.381

#BEST SVM K: k = 7 ; AUC = 0.5787 ; precision = 0.483 ; recall = 0.369

# Satellite - 2036 outliers --------------------
path <- "satellite.mat"
data=readMat(path)
X=data$X
p <- dim(X)[2]; n <- dim(X)[1]
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))

### LOF:
outlier.scores <- lof(X, k=mean(c(log(n), p+1, 2*p)))

rfind <- rpart_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=18)
abline(h=rfind, col=4)
rbest_taglio <- sort(rfind)[4] #ginocchio
out_previsti <- rep(0,n)
out_previsti <- ifelse(outlier.scores>rbest_taglio, 1, 0)

#Confusion Matrix & AUC:
table(LOF=out_previsti, Actual=is_out)
#Valutazione performance:
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.5271
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.508
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.098

cfind <- ctree_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=19)
abline(h=cfind, col=cfind+1) 
cbest_taglio <- sort(cfind)[1]
out_previsti=rep(0,n)
out_previsti <- ifelse(outlier.scores>cbest_taglio, 1, 0)
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.5249
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.51
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.0898

#BEST LOF: taglio = 1.4 ; AUC = 0.5271 ; precision = 0.508 ; recall = 0.0898

### SVM:

# nu with 4 values:
nu_val <- c(0.01, 0.1, 0.9, 0.5)
auc_svm <- c()
precision <- c()
recall <- c()
for (el in nu_val){
  svm.model<-svm(X, y=NULL,
                 type='one-classification',
                 nu=el,
                 kernel="radial")
  
  svm.predtrain<-predict(svm.model, X)
  out <- rep(0, n)
  out[svm.predtrain==FALSE] <- 1
  roc_lof<- roc(response = is_out, predictor = out)
  auc_svm <- c(auc_svm ,auc(roc_lof))
  confusion_matrix<-table(SVM=out, Actual=is_out)
  precision <- c(precision, confusion_matrix[2,2]/sum(confusion_matrix[2,]))
  recall <- c(recall, confusion_matrix[2,2]/sum(confusion_matrix[,2]))
}
auc_svm; precision; recall

#BEST SVM NU: nu = 0.5 ; AUC = 0.5887 ; precision = 0.393 ; recall = 0.621

# OCSVM paper with k=7 and k=Best knn value:
#K=7:
distanze<-kNNdist(X,k=7)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK)); nu
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.5413
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.336
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.962

#K=BEST knn:
distanze<-kNNdist(X,k=23)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.5291
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.33
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.962

#BEST SVM K: k = 7 ; AUC = 0.5413 ; precision = 0.336 ; recall = 0.961

#Satellite SCALATO -----
X <- scale(X)
### LOF:
outlier.scores <- lof(X, k=mean(c(log(n), p+1, 2*p)))

rfind <- rpart_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=18)
abline(h=rfind, col=4)
rbest_taglio <- sort(rfind)[3] #1.39
out_previsti <- rep(0,n)
out_previsti <- ifelse(outlier.scores>rbest_taglio, 1, 0)

#Confusion Matrix & AUC:
table(LOF=out_previsti, Actual=is_out)
#Valutazione performance:
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.5229
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.487
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.0894

cfind <- ctree_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=19)
abline(h=cfind, col=cfind+1)
cbest_taglio <- sort(cfind)[1]
out_previsti=rep(0,n)
out_previsti <- ifelse(outlier.scores>cbest_taglio, 1, 0)
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.5231
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.488
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.0899

#BEST LOF: taglio = 1.38993 ; AUC = 0.5231 ; precision = 0.488 ; recall = 0.0899

### SVM:

# nu with 4 values:
nu_val <- c(0.01, 0.1, 0.9, 0.5)
auc_svm <- c()
precision <- c()
recall <- c()
for (el in nu_val){
  svm.model<-svm(X, y=NULL,
                 type='one-classification',
                 nu=el,
                 kernel="radial")
  
  svm.predtrain<-predict(svm.model, X)
  out <- rep(0, n)
  out[svm.predtrain==FALSE] <- 1
  roc_lof<- roc(response = is_out, predictor = out)
  auc_svm <- c(auc_svm ,auc(roc_lof))
  confusion_matrix<-table(SVM=out, Actual=is_out)
  precision <- c(precision, confusion_matrix[2,2]/sum(confusion_matrix[2,]))
  recall <- c(recall, confusion_matrix[2,2]/sum(confusion_matrix[,2]))
}
auc_svm; precision; recall

#BEST SVM NU: nu = 0.5 ; AUC = 0.5887 ; precision = 0.393 ; recall = 0.621

# OCSVM paper with k=7 and k=Best knn value:
#K=7:
distanze<-kNNdist(X,k=7)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-(sum(SK)-x.max_cf)/sum(SK)
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.6251
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.4997
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.466

#K=BEST knn:
distanze<-kNNdist(X,k=37)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-(sum(SK)-x.max_cf)/sum(SK)
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.6396
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.463
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.603

#BEST SVM K: k = 37 ; AUC = 0.6393 ; precision = 0.463 ; recall = 0.603

# Shuttle - 3511 outliers ----------------------
path <- "shuttle.mat"
data=readMat(path)
X=data$X
p <- dim(X)[2]; n <- dim(X)[1]
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))

### LOF:
#Non si può fare perchè con dati così ad alta dimensionalità si devono allocare vettori di 
#dimensioni enormi

### SVM:

# nu with 4 values:
nu_val <- c(0.01, 0.1, 0.9, 0.5)
auc_svm <- c()
precision <- c()
recall <- c()
for (el in nu_val){
  svm.model<-svm(X, y=NULL,
                 type='one-classification',
                 nu=el,
                 kernel="radial")
  
  svm.predtrain<-predict(svm.model, X)
  out <- rep(0, n)
  out[svm.predtrain==FALSE] <- 1
  roc_lof<- roc(response = is_out, predictor = out)
  auc_svm <- c(auc_svm ,auc(roc_lof))
  confusion_matrix<-table(SVM=out, Actual=is_out)
  precision <- c(precision, confusion_matrix[2,2]/sum(confusion_matrix[2,]))
  recall <- c(recall, confusion_matrix[2,2]/sum(confusion_matrix[,2]))
}
auc_svm; precision; recall

#BEST SVM NU: nu = 0.1 ; AUC = 0.7942 ; precision = 0.462 ; recall = 0.646

# OCSVM paper with k=7 and k=Best knn value:
#K=7:
distanze<-kNNdist(X,k=7)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK)); nu
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.7424
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.13
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.996

#K=BEST knn:
distanze<-kNNdist(X,k=11)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.7126
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.118
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #1

#BEST SVM K: k = 7 ; AUC = 0.7424 ; precision = 0.13 ; recall = 0.996

#Shuttle SCALATO -----
X <- scale(X)
### LOF:
#Alta dimensionalità, NON applicabile

### SVM:

# nu with 4 values:
nu_val <- c(0.01, 0.1, 0.9, 0.5)
auc_svm <- c()
precision <- c()
recall <- c()
for (el in nu_val){
  svm.model<-svm(X, y=NULL,
                 type='one-classification',
                 nu=el,
                 kernel="radial")
  
  svm.predtrain<-predict(svm.model, X)
  out <- rep(0, n)
  out[svm.predtrain==FALSE] <- 1
  roc_lof<- roc(response = is_out, predictor = out)
  auc_svm <- c(auc_svm ,auc(roc_lof))
  confusion_matrix<-table(SVM=out, Actual=is_out)
  precision <- c(precision, confusion_matrix[2,2]/sum(confusion_matrix[2,]))
  recall <- c(recall, confusion_matrix[2,2]/sum(confusion_matrix[,2]))
}
auc_svm; precision; recall

#BEST SVM NU: nu = 0.1 ; AUC = 0.7942 ; precision = 0.462 ; recall = 0.646

# OCSVM paper with k=7 and k=Best knn value:
#K=7:
distanze<-kNNdist(X,k=7)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-(sum(SK)-x.max_cf)/sum(SK); nu
nu <- 0.5
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.7653
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.142
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.992

#K=BEST knn:
distanze<-kNNdist(X,k=11)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-(sum(SK)-x.max_cf)/sum(SK); nu
nu <- 0.5
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.7653
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.142
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.992

#BEST SVM K: k = 7/11 stessi risultati ; AUC = 0.7653 ; precision = 0.142 ; recall = 0.992

# BreastW - 239 outliers -----------------------
path="breastw.mat"
data=readMat(path)
X=data$X
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))

fac<-1:p
for (i in 1:length(fac)){
  dati[,i]<-as.factor(dati[,i])
}
#One hot encoding:
dati<-cbind(as.data.frame(model.matrix(~.-1,data=dati[,-(p+1)])),is_out)
X=dati[,1:(ncol(dati)-1)]

p <- dim(X)[2]; n <- dim(X)[1]

### LOF:
outlier.scores <- lof(X, k=mean(c(log(n), p+1, 2*p)))

rfind <- rpart_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=18)
abline(h=rfind, col=4)
abline(h=1.68, col=2) #buon senso
rbest_taglio <- 1.68
out_previsti <- rep(0,n)
out_previsti <- ifelse(outlier.scores>rbest_taglio, 1, 0)

#Confusion Matrix & AUC:
table(LOF=out_previsti, Actual=is_out)
#Valutazione performance:
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.954
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.945
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.937

cfind <- ctree_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=19)
abline(h=cfind, col=cfind+1)#non sensato allora teniamo lo split precedente
# cbest_taglio <- sort(cfind)[1]
# out_previsti=rep(0,n)
# out_previsti <- ifelse(outlier.scores>cbest_taglio, 1, 0)
# roc_lof<- roc(response = is_out, predictor = out_previsti)
# auc(roc_lof) #
# confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
# precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #
# recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #

#BEST LOF: taglio = 1.68 ; AUC = 0.954 ; precision = 0.945 ; recall = 0.937

### SVM:

# nu with 4 values:
nu_val <- c(0.01, 0.1, 0.9, 0.5)
auc_svm <- c()
precision <- c()
recall <- c()
for (el in nu_val){
  svm.model<-svm(X, y=NULL,
                 type='one-classification',
                 nu=el,
                 kernel="radial")
  
  svm.predtrain<-predict(svm.model, X)
  out <- rep(0, n)
  out[svm.predtrain==FALSE] <- 1
  roc_lof<- roc(response = is_out, predictor = out)
  auc_svm <- c(auc_svm ,auc(roc_lof))
  confusion_matrix<-table(SVM=out, Actual=is_out)
  precision <- c(precision, confusion_matrix[2,2]/sum(confusion_matrix[2,]))
  recall <- c(recall, confusion_matrix[2,2]/sum(confusion_matrix[,2]))
}
auc_svm; precision; recall

#BEST SVM NU: nu = 0.5 ; AUC = 0.8829 ; precision = 0.697 ; recall = 1.00

# OCSVM paper with k=7 and k=Best knn value:
#K=7:
distanze<-kNNdist(X,k=7)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK)); nu
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.8829
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.697
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #1

#K=BEST knn:
#k=7

#BEST SVM K: k = 7 ; AUC =0.8829 ; precision = 0.697 ; recall = 1

# BreastW SCALATO ----
X <- scale(X)
### LOF:
outlier.scores <- lof(X, k=mean(c(log(n), p+1, 2*p)))

rfind <- rpart_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=18)
abline(h=rfind, col=4)
abline(h=2.28, col=2) #buon senso
rbest_taglio <- 2.28
out_previsti <- rep(0,n)
out_previsti <- ifelse(outlier.scores>rbest_taglio, 1, 0)
#Confusion Matrix & AUC:
table(LOF=out_previsti, Actual=is_out)
#Valutazione performance:
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.9693
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.928
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.979

cfind <- ctree_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=19)
abline(h=cfind, col=cfind+1)#non sensato allora teniamo lo split precedente
# cbest_taglio <- sort(cfind)[1]
# out_previsti=rep(0,n)
# out_previsti <- ifelse(outlier.scores>cbest_taglio, 1, 0)
# roc_lof<- roc(response = is_out, predictor = out_previsti)
# auc(roc_lof) #
# confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
# precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #
# recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #

#BEST LOF: taglio = 2.28 ; AUC = 0.9693 ; precision = 0.928 ; recall = 0.979

### SVM:

# nu with 4 values:
nu_val <- c(0.01, 0.1, 0.9, 0.5)
auc_svm <- c()
precision <- c()
recall <- c()
for (el in nu_val){
  svm.model<-svm(X, y=NULL,
                 type='one-classification',
                 nu=el,
                 kernel="radial")
  
  svm.predtrain<-predict(svm.model, X)
  out <- rep(0, n)
  out[svm.predtrain==FALSE] <- 1
  roc_lof<- roc(response = is_out, predictor = out)
  auc_svm <- c(auc_svm ,auc(roc_lof))
  confusion_matrix<-table(SVM=out, Actual=is_out)
  precision <- c(precision, confusion_matrix[2,2]/sum(confusion_matrix[2,]))
  recall <- c(recall, confusion_matrix[2,2]/sum(confusion_matrix[,2]))
}
auc_svm; precision; recall

#BEST SVM NU: nu = 0.5 ; AUC = 0.8829 ; precision = 0.697 ; recall = 1.00

# OCSVM paper with k=7 and k=Best knn value:
#K=7:
distanze<-kNNdist(X,k=7)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK)); nu
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.6261
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.418
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #1

#K=BEST knn:
distanze<-kNNdist(X,k=10)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK)); nu
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.6002
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.402
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #1

#BEST SVM K: k = 7 ; AUC =0.6261 ; precision = 0.418 ; recall = 1

# Arrhythmia - 66 outliers ---------------------
path="arrhythmia.mat"
data=readMat(path)
X=data$X
p <- dim(X)[2]; n <- dim(X)[1]
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))

### LOF:
outlier.scores <- lof(X, k=mean(c(log(n), p+1, 2*p)))

rfind <- rpart_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=18)
abline(h=rfind, col=4)
abline(h=1.35, col=2) #buon senso
rbest_taglio <- 1.35
out_previsti <- rep(0,n)
out_previsti <- ifelse(outlier.scores>rbest_taglio, 1, 0)

#Confusion Matrix & AUC:
table(LOF=out_previsti, Actual=is_out)
#Valutazione performance:
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.6221
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.621
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.273

cfind <- ctree_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=19)
abline(h=cfind, col=cfind+1) #vuoto

#BEST LOF: taglio = 1.35 ; AUC = 0.6221 ; precision = 0.621 ; recall = 0.273

### SVM:

# nu with 4 values:
nu_val <- c(0.01, 0.1, 0.9, 0.5)
auc_svm <- c()
precision <- c()
recall <- c()
for (el in nu_val){
  svm.model<-svm(X, y=NULL,
                 type='one-classification',
                 nu=el,
                 kernel="radial")
  
  svm.predtrain<-predict(svm.model, X)
  out <- rep(0, n)
  out[svm.predtrain==FALSE] <- 1
  roc_lof<- roc(response = is_out, predictor = out)
  auc_svm <- c(auc_svm ,auc(roc_lof))
  confusion_matrix<-table(SVM=out, Actual=is_out)
  precision <- c(precision, confusion_matrix[2,2]/sum(confusion_matrix[2,]))
  recall <- c(recall, confusion_matrix[2,2]/sum(confusion_matrix[,2]))
}
auc_svm; precision; recall

#BEST SVM NU: nu = 0.5 ; AUC = 0.5 ; precision = 0.146; recall = 0.97

# OCSVM paper with k=7 and k=Best knn value:
#K=7:
distanze<-kNNdist(X,k=7)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK)); nu
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.5
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.146
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.364

#K=BEST knn:
distanze<-kNNdist(X,k=6)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.468
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.11
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.167

#BEST SVM K: k = 7 ; AUC = 0.5 ; precision = 0.146 ; recall = 0.364

#Arrhythmia SCALATO -----
X <- scale(X)
dati <- cbind(X, is_out)
togliere<-which(nearZeroVar(dati, saveMetrics = T)$zeroVar==TRUE)
dati=dati[,-togliere]
X=dati[,-ncol(dati)]
p=dim(X)[2]; n=dim(X)[1]

### LOF:
outlier.scores <- lof(X, k=mean(c(log(n), p+1, 2*p)))

rfind <- rpart_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=18)
abline(h=rfind, col=4)
abline(h=1.5, col=2)
rbest_taglio <- 1.5 #buon senso
out_previsti <- rep(0,n)
out_previsti <- ifelse(outlier.scores>rbest_taglio, 1, 0)

#Confusion Matrix & AUC:
table(LOF=out_previsti, Actual=is_out)
#Valutazione performance:
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.6428
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.41
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.379

cfind <- ctree_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=19)
abline(h=cfind, col=cfind+1)
cbest_taglio <- sort(cfind)[1]
out_previsti=rep(0,n)
out_previsti <- ifelse(outlier.scores>cbest_taglio, 1, 0)
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.7087
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.368
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.591

#BEST LOF: taglio = 1.14221 ; AUC = 0.7087 ; precision = 0.368 ; recall = 0.591

### SVM:

# nu with 4 values:
nu_val <- c(0.01, 0.1, 0.9, 0.5)
auc_svm <- c()
precision <- c()
recall <- c()
for (el in nu_val){
  svm.model<-svm(X, y=NULL,
                 type='one-classification',
                 nu=el,
                 kernel="radial")
  
  svm.predtrain<-predict(svm.model, X)
  out <- rep(0, n)
  out[svm.predtrain==FALSE] <- 1
  roc_lof<- roc(response = is_out, predictor = out)
  auc_svm <- c(auc_svm ,auc(roc_lof))
  confusion_matrix<-table(SVM=out, Actual=is_out)
  precision <- c(precision, confusion_matrix[2,2]/sum(confusion_matrix[2,]))
  recall <- c(recall, confusion_matrix[2,2]/sum(confusion_matrix[,2]))
}
auc_svm; precision; recall

#BEST SVM NU: nu = 0.5 ; AUC = 0.695 ; precision = 0.243; recall = 0.833

# OCSVM paper with k=7 and k=Best knn value:
#K=7:
distanze<-kNNdist(X,k=7)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-(sum(SK)-x.max_cf)/sum(SK); nu
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.5365
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.156
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.985

#K=BEST knn:
distanze<-kNNdist(X,k=132)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-(sum(SK)-x.max_cf)/sum(SK)
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.5237
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.152
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.97

#BEST SVM K: k = 7 ; AUC = 0.5365 ; precision = 0.156 ; recall = 0.985

### Arrythmia COMPLETO -----
#Su dataset completo:
dati<-read.csv("arrhythmia.csv", sep=",", header=F)
fac<-c(2,22:27)
X <- dati
p <- dim(X)[2]; n <- dim(X)[1]
for (i in 1:length(fac)){
  dati[,i]<-as.factor(dati[,i])
}
#One hot encoding:
dati<-cbind(as.data.frame(model.matrix(~.-1,data=dati[,-(p+1)])),is_out)
X=dati[,1:(ncol(dati)-1)]
p <- dim(X)[2]; n <- dim(X)[1]

### LOF:
outlier.scores <- lof(X, k=log(n)) #solo log(n) perchè n<p

rfind <- rpart_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=18)
abline(h=rfind, col=4)
rbest_taglio <- sort(rfind)[5]
out_previsti <- rep(0,n)
out_previsti <- ifelse(outlier.scores>rbest_taglio, 1, 0)
#Confusion Matrix & AUC:
table(LOF=out_previsti, Actual=is_out)
#Valutazione performance:
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.6952
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.333
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.015

cfind <- ctree_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=19)
abline(h=cfind, col=cfind+1) 
cbest_taglio <- sort(cfind)[1]
out_previsti=rep(0,n)
out_previsti <- ifelse(outlier.scores>cbest_taglio, 1, 0)
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.5219
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.226
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.106

#BEST LOF: taglio = 2 ; AUC = 0.6952 ; precision = 0.333 ; recall = 0.015

### SVM:

# nu with 4 values:
nu_val <- c(0.01, 0.1, 0.9, 0.5)
auc_svm <- c()
precision <- c()
recall <- c()
for (el in nu_val){
  svm.model<-svm(X, y=NULL,
                 type='one-classification',
                 nu=el,
                 kernel="radial")
  
  svm.predtrain<-predict(svm.model, X)
  out <- rep(0, n)
  out[svm.predtrain==FALSE] <- 1
  roc_lof<- roc(response = is_out, predictor = out)
  auc_svm <- c(auc_svm ,auc(roc_lof))
  confusion_matrix<-table(SVM=out, Actual=is_out)
  precision <- c(precision, confusion_matrix[2,2]/sum(confusion_matrix[2,]))
  recall <- c(recall, confusion_matrix[2,2]/sum(confusion_matrix[,2]))
}
auc_svm; precision; recall

#BEST SVM NU: nu = 0.1 ; AUC = 0.553 ; precision = 0.174 ; recall = 0.576

# OCSVM paper with k=7 and k=Best knn value:
#K=7:
distanze<-kNNdist(X,k=7)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK)); nu
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.4558
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.132
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.712

#K=BEST knn:
distanze<-kNNdist(X,k=6)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.5338
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.183
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.288

#BEST SVM K: k = 6 ; AUC = 0.5338 ; precision = 0.183 ; recall = 0.288

#Arrhythmia COMPLETO SCALATO -----
X <- scale(X)
dati <- cbind(X, is_out)
togliere<-which(nearZeroVar(dati, saveMetrics = T)$zeroVar==TRUE)
dati=dati[,-togliere]
X=dati[,-ncol(dati)]
p=dim(X)[2]; n=dim(X)[1]

### LOF:
outlier.scores <- lof(X, k=log(n))

rfind <- rpart_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=18)
abline(h=rfind, col=4)
abline(h=1.51, col=2) #buon senso
rbest_taglio <- 1.51
out_previsti <- rep(0,n)
out_previsti <- ifelse(outlier.scores>rbest_taglio, 1, 0)
#Confusion Matrix & AUC:
table(LOF=out_previsti, Actual=is_out)
#Valutazione performance:
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.6298
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.352
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.379

cfind <- ctree_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=19)
abline(h=cfind, col=cfind+1) 
cbest_taglio <- max(cfind)
out_previsti=rep(0,n)
out_previsti <- ifelse(outlier.scores>cbest_taglio, 1, 0)
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.7028
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.274
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.742

#BEST LOF: taglio = 1.2408 ; AUC = 0.7028 ; precision = 0.274 ; recall = 0.742

### SVM:

# nu with 4 values:
nu_val <- c(0.01, 0.1, 0.9, 0.5)
auc_svm <- c()
precision <- c()
recall <- c()
for (el in nu_val){
  svm.model<-svm(X, y=NULL,
                 type='one-classification',
                 nu=el,
                 kernel="radial")
  
  svm.predtrain<-predict(svm.model, X)
  out <- rep(0, n)
  out[svm.predtrain==FALSE] <- 1
  roc_lof<- roc(response = is_out, predictor = out)
  auc_svm <- c(auc_svm ,auc(roc_lof))
  confusion_matrix<-table(SVM=out, Actual=is_out)
  precision <- c(precision, confusion_matrix[2,2]/sum(confusion_matrix[2,]))
  recall <- c(recall, confusion_matrix[2,2]/sum(confusion_matrix[,2]))
}
auc_svm; precision; recall

#BEST SVM NU: nu = 0.5 ; AUC = 0.684 ; precision = 0.237 ; recall = 0.818

# OCSVM paper with k=7 and k=Best knn value:
#K=7:
distanze<-kNNdist(X,k=7)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-(sum(SK)-x.max_cf)/sum(SK)
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.4488
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.093
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.151

#K=BEST knn:
distanze<-kNNdist(X,k=451)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-(sum(SK)-x.max_cf)/sum(SK)
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.5803
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.115
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.5

#BEST SVM K: k = 451 ; AUC = 0.5803 ; precision = 0.115 ; recall = 0.5

# Ionosphere - 126 outliers --------------------
path="ionosphere.mat"
data=readMat(path)
X <- data$X
p=dim(X)[2]; n=dim(X)[1]
is_out <- data$y
dati<-data.frame(cbind(X, is_out)) 

### LOF:
outlier.scores <- lof(X, k=mean(c(log(n), p+1, 2*p)))

rfind <- rpart_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=18)
abline(h=rfind, col=4)
rbest_taglio <- max(rfind)
out_previsti <- rep(0,n)
out_previsti <- ifelse(outlier.scores>rbest_taglio, 1, 0)
#Confusion Matrix & AUC:
table(LOF=out_previsti, Actual=is_out)
#Valutazione performance:
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.7952
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.729
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.746

cfind <- ctree_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=19)
abline(h=cfind, col=cfind+1) 
cbest_taglio <- max(cfind)
out_previsti=rep(0,n)
out_previsti <- ifelse(outlier.scores>cbest_taglio, 1, 0)
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #07695
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.609
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.841

#BEST LOF: taglio = 1.48 ; AUC = 0.7952 ; precision = 0.729 ; recall = 0.746

### SVM:

# nu with 4 values:
nu_val <- c(0.01, 0.1, 0.9, 0.5)
auc_svm <- c()
precision <- c()
recall <- c()
for (el in nu_val){
  svm.model<-svm(X, y=NULL,
                 type='one-classification',
                 nu=el,
                 kernel="radial")
  
  svm.predtrain<-predict(svm.model, X)
  out <- rep(0, n)
  out[svm.predtrain==FALSE] <- 1
  roc_lof<- roc(response = is_out, predictor = out)
  auc_svm <- c(auc_svm ,auc(roc_lof))
  confusion_matrix<-table(SVM=out, Actual=is_out)
  precision <- c(precision, confusion_matrix[2,2]/sum(confusion_matrix[2,]))
  recall <- c(recall, confusion_matrix[2,2]/sum(confusion_matrix[,2]))
}
auc_svm; precision; recall

#BEST SVM NU: nu = 0.7448 ; AUC = 0.776 ; precision = 0.586 ; recall = 0.809

# OCSVM paper with k=7 and k=Best knn value:
#K=7:
distanze<-kNNdist(X,k=7)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))

svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.8059
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.748
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.754

#K=BEST knn:
distanze<-kNNdist(X,k=6)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.8086
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.764
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.746

#BEST SVM K: k = 6 ; AUC = 0.8086 ; precision = 0.764 ; recall = 0.746


# Mnist - 700 outliers -------------------------
path="mnist.mat"
#inserire path
data=readMat(path)
X <- data$X
p=dim(X)[2]; n=dim(X)[1]
is_out <- data$y
dati<-data.frame(cbind(X, is_out)) 
togliere<-which(nearZeroVar(dati, saveMetrics = T)$zeroVar==TRUE)
dati=dati[,-togliere]
X=dati[,-ncol(dati)]
p=dim(X)[2]; n=dim(X)[1]

### LOF:
outlier.scores <- lof(X, k=mean(c(log(n), p+1, 2*p)))

rfind <- rpart_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=18)
abline(h=rfind, col=4)
rbest_taglio <- max(rfind)
out_previsti <- rep(0,n)
out_previsti <- ifelse(outlier.scores>rbest_taglio, 1, 0)

#Confusion Matrix & AUC:
table(LOF=out_previsti, Actual=is_out)
#Valutazione performance:
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.6078
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.475
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.243

cfind <- ctree_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=19)
abline(h=cfind, col=cfind+1)
cbest_taglio <- max(cfind)
out_previsti=rep(0,n)
out_previsti <- ifelse(outlier.scores>cbest_taglio, 1, 0)
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.5927
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.507
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.206

#BEST LOF: taglio = 1.2 ; AUC = 0.6078 ; precision = 0.475 ; recall = 0.243

### SVM:

# nu with 4 values:
nu_val <- c(0.01, 0.1, 0.9, 0.5)
auc_svm <- c()
precision <- c()
recall <- c()
for (el in nu_val){
  svm.model<-svm(X, y=NULL,
                 type='one-classification',
                 nu=el,
                 kernel="radial")
  
  svm.predtrain<-predict(svm.model, X)
  out <- rep(0, n)
  out[svm.predtrain==FALSE] <- 1
  roc_lof<- roc(response = is_out, predictor = out)
  auc_svm <- c(auc_svm ,auc(roc_lof))
  confusion_matrix<-table(SVM=out, Actual=is_out)
  precision <- c(precision, confusion_matrix[2,2]/sum(confusion_matrix[2,]))
  recall <- c(recall, confusion_matrix[2,2]/sum(confusion_matrix[,2]))
}
auc_svm; precision; recall

#BEST SVM NU: nu = 0.5 ; AUC = 0.734 ; precision = 0.17 ; recall = 0.924

# OCSVM paper with k=7 and k=Best knn value:
#K=7:
distanze<-kNNdist(X,k=7)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))

svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.5069
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.094
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.571

#K=BEST knn:
distanze<-kNNdist(X,k=44)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.5091
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.102
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.171

#BEST SVM K: k = 7 ; AUC = 0.5069 ; precision = 0.094 ; recall = 0.571

#Mnist SCALATO ----
X <- scale(X)

### LOF:
outlier.scores <- lof(X, k=mean(c(log(n), p+1, 2*p)))

rfind <- rpart_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=18)
abline(h=rfind, col=4)
rbest_taglio <- sort(rfind)[length(rfind)-2] 
out_previsti <- rep(0,n)
out_previsti <- ifelse(outlier.scores>rbest_taglio, 1, 0)

#Confusion Matrix & AUC:
table(LOF=out_previsti, Actual=is_out)
#Valutazione performance:
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.5732
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.667
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.154

cfind <- ctree_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=19)
abline(h=cfind, col=cfind+1) 
cbest_taglio <- sort(cfind)[length(cfind)-1]
out_previsti=rep(0,n)
out_previsti <- ifelse(outlier.scores>cbest_taglio, 1, 0)
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.5874
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.606
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.187

#BEST LOF: taglio = 1.60127 ; AUC = 0.5874 ; precision = 0.606 ; recall = 0.187

### SVM:

# nu with 4 values:
nu_val <- c(0.01, 0.1, 0.9, 0.5)
auc_svm <- c()
precision <- c()
recall <- c()
for (el in nu_val){
  svm.model<-svm(X, y=NULL,
                 type='one-classification',
                 nu=el,
                 kernel="radial")
  
  svm.predtrain<-predict(svm.model, X)
  out <- rep(0, n)
  out[svm.predtrain==FALSE] <- 1
  roc_lof<- roc(response = is_out, predictor = out)
  auc_svm <- c(auc_svm ,auc(roc_lof))
  confusion_matrix<-table(SVM=out, Actual=is_out)
  precision <- c(precision, confusion_matrix[2,2]/sum(confusion_matrix[2,]))
  recall <- c(recall, confusion_matrix[2,2]/sum(confusion_matrix[,2]))
}
auc_svm; precision; recall

#BEST SVM NU: nu = 0.5 ; AUC = 0.7336 ; precision = 0.17 ; recall = 0.924

# OCSVM paper with k=7 and k=Best knn value:
#K=7:
distanze<-kNNdist(X,k=7)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))

svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.6208
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.118
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.999

#K=BEST knn:
distanze<-kNNdist(X,k=79)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.5929
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.111
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.998

#BEST SVM K: k = 7 ; AUC = 0.6208 ; precision = 0.117 ; recall = 0.999

# Optdigits - 150 outliers ---------------------
path="optdigits.mat"
data=readMat(path)
X <- data$X
p=dim(X)[2]; n=dim(X)[1]
is_out <- data$y
dati<-data.frame(cbind(X, is_out)) 
togliere<-which(nearZeroVar(dati, saveMetrics = T)$zeroVar==TRUE)
dati=dati[,-togliere]
X=dati[,-ncol(dati)]
p=dim(X)[2]; n=dim(X)[1]

### LOF:
outlier.scores <- lof(X, k=mean(c(log(n), p+1, 2*p)))

rfind <- rpart_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=18)
abline(h=rfind, col=4)
rbest_taglio <- max(rfind)
out_previsti <- rep(0,n)
out_previsti <- ifelse(outlier.scores>rbest_taglio, 1, 0)

#Confusion Matrix & AUC:
table(LOF=out_previsti, Actual=is_out)
#Valutazione performance:
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.5123
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.286
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.027

cfind <- ctree_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=19)
abline(h=cfind, col=cfind+1) #nessun taglio
# cbest_taglio <- sort(cfind)[1] 
# out_previsti=rep(0,n)
# out_previsti <- ifelse(outlier.scores>cbest_taglio, 1, 0)
# roc_lof<- roc(response = is_out, predictor = out_previsti)
# auc(roc_lof) #0.5867
# confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
# precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.167
# recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.22

#BEST LOF: taglio = 1.35 ; AUC = 0.5123 ; precision = 0.286 ; recall = 0.027

### SVM:

# nu with 4 values:
nu_val <- c(0.01, 0.1, 0.9, 0.5)
auc_svm <- c()
precision <- c()
recall <- c()
for (el in nu_val){
  svm.model<-svm(X, y=NULL,
                 type='one-classification',
                 nu=el,
                 kernel="radial")
  
  svm.predtrain<-predict(svm.model, X)
  out <- rep(0, n)
  out[svm.predtrain==FALSE] <- 1
  roc_lof<- roc(response = is_out, predictor = out)
  auc_svm <- c(auc_svm ,auc(roc_lof))
  confusion_matrix<-table(SVM=out, Actual=is_out)
  precision <- c(precision, confusion_matrix[2,2]/sum(confusion_matrix[2,]))
  recall <- c(recall, confusion_matrix[2,2]/sum(confusion_matrix[,2]))
}
auc_svm; precision; recall

#BEST SVM NU: nu = 0.9 ; AUC = 0.5516 ; precision = 0.032 ; recall = 1.00

# OCSVM paper with k=7 and k=Best knn value:
#K=7:
distanze<-kNNdist(X,k=7)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))

svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.526
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.033
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.387

#K=BEST knn:
distanze<-kNNdist(X,k=63)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.5248
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.034
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.32

#BEST SVM K: k = 7 ; AUC = 0.526 ; precision = 0.033 ; recall = 0.387

#Opdigits SCALATO ----
X <- scale(X)

### LOF:
outlier.scores <- lof(X, k=mean(c(log(n), p+1, 2*p)))

rfind <- rpart_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=18)
abline(h=rfind, col=4)
rbest_taglio <- max(rfind)
out_previsti <- rep(0,n)
out_previsti <- ifelse(outlier.scores>rbest_taglio, 1, 0)

#Confusion Matrix & AUC:
table(LOF=out_previsti, Actual=is_out)
#Valutazione performance:
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.4827
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.018
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.06

cfind <- ctree_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=19)
abline(h=cfind, col=cfind+1) #nessun taglio
abline(h=3.1, col=2) #buon senso
cbest_taglio <- 3
out_previsti=rep(0,n)
out_previsti <- ifelse(outlier.scores>cbest_taglio, 1, 0)
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.4974
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0

#BEST LOF: taglio = 3.1 ; AUC = 0.4974; precision = 0 ; recall = 0

### SVM:

# nu with 4 values:
nu_val <- c(0.01, 0.1, 0.9, 0.5)
auc_svm <- c()
precision <- c()
recall <- c()
for (el in nu_val){
  svm.model<-svm(X, y=NULL,
                 type='one-classification',
                 nu=el,
                 kernel="radial")
  
  svm.predtrain<-predict(svm.model, X)
  out <- rep(0, n)
  out[svm.predtrain==FALSE] <- 1
  roc_lof<- roc(response = is_out, predictor = out)
  auc_svm <- c(auc_svm ,auc(roc_lof))
  confusion_matrix<-table(SVM=out, Actual=is_out)
  precision <- c(precision, confusion_matrix[2,2]/sum(confusion_matrix[2,]))
  recall <- c(recall, confusion_matrix[2,2]/sum(confusion_matrix[,2]))
}
auc_svm; precision; recall

#BEST SVM NU: nu = 0.9 ; AUC = 0.5516 ; precision = 0.031 ; recall = 1.00

# OCSVM paper with k=7 and k=Best knn value:
#K=7:
distanze<-kNNdist(X,k=7)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))

svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.5668
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.034
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.88

#K=BEST knn:
#Nessun k trovato

#BEST SVM K: k = 7 ; AUC = 0.5668 ; precision = 0.034 ; recall = 0.88

#Http - 2211 outliers --------------------------
path <- "X_http.mat"
data=readMat(path)
X=data$X
p <- dim(X)[2]; n <- dim(X)[1]
ydata <- readMat("y_http.mat")
y <- ydata$y
is_out<-as.vector(y)
dati<-data.frame(cbind(X, is_out))
X <- dati[,-is_out]
is_out <- dati[,4]

### LOF:
#Alta dimensionalità

### SVM:

# nu with 4 values:
nu_val <- c(0.01, 0.1, 0.9, 0.5)
auc_svm <- c()
precision <- c()
recall <- c()
for (el in nu_val){
  svm.model<-svm(X, y=NULL,
                 type='one-classification',
                 nu=el,
                 kernel="radial")
  
  svm.predtrain<-predict(svm.model, X)
  out <- rep(0, n)
  out[svm.predtrain==FALSE] <- 1
  roc_lof<- roc(response = is_out, predictor = out)
  auc_svm <- c(auc_svm ,auc(roc_lof))
  confusion_matrix<-table(SVM=out, Actual=is_out)
  precision <- c(precision, confusion_matrix[2,2]/sum(confusion_matrix[2,]))
  recall <- c(recall, confusion_matrix[2,2]/sum(confusion_matrix[,2]))
}
auc_svm; precision; recall

#BEST SVM NU: nu =  ; AUC =  ; precision =  ; recall = 

# OCSVM paper with k=7 and k=Best knn value:
#K=7:
distanze<-kNNdist(X,k=7)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))
if (nu>1){
  nu <- 0.5
}
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #

#K=BEST knn:
#k=13
distanze<-kNNdist(X,k=13)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))
if (nu>1){
  nu <- 0.5
}
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #


#BEST SVM K: k =  ; AUC = ; precision =  ; recall = 

#Http SCALATO -----
X <- scale(X)
### LOF:
#Alta dimensionalità

### SVM:

# nu with 4 values:
nu_val <- c(0.01, 0.1, 0.9, 0.5)
auc_svm <- c()
precision <- c()
recall <- c()
for (el in nu_val){
  svm.model<-svm(X, y=NULL,
                 type='one-classification',
                 nu=el,
                 kernel="radial")
  
  svm.predtrain<-predict(svm.model, X)
  out <- rep(0, n)
  out[svm.predtrain==FALSE] <- 1
  roc_lof<- roc(response = is_out, predictor = out)
  auc_svm <- c(auc_svm ,auc(roc_lof))
  confusion_matrix<-table(SVM=out, Actual=is_out)
  precision <- c(precision, confusion_matrix[2,2]/sum(confusion_matrix[2,]))
  recall <- c(recall, confusion_matrix[2,2]/sum(confusion_matrix[,2]))
}
auc_svm; precision; recall

#BEST SVM NU: nu =  ; AUC =  ; precision =  ; recall = 

# OCSVM paper with k=7 and k=Best knn value:
#K=7:
distanze<-kNNdist(X,k=7)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))
if (nu>1){
  nu <- 0.5
}
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #

#K=BEST knn:
#k=9
distanze<-kNNdist(X,k=9)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))
if (nu>1){
  nu <- 0.5
}
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #


#BEST SVM K: k =  ; AUC =  ; precision =  ; recall = 


# ForestCover - 2747 outliers ------------------
path="cover.mat"
data=readMat(path)
X <- data$X
p=dim(X)[2]; n=dim(X)[1]
is_out <- data$y
dati<-data.frame(cbind(X, is_out)) 
togliere<-which(nearZeroVar(dati, saveMetrics = T)$zeroVar==TRUE)
dati=dati[,-togliere]
X=dati[,-ncol(dati)]
p=dim(X)[2]; n=dim(X)[1]

### LOF:
#Alta dimensionalità

### SVM:

# nu with 4 values:
nu_val <- c(0.01, 0.1, 0.9, 0.5)
auc_svm <- c()
precision <- c()
recall <- c()
for (el in nu_val){
  svm.model<-svm(X, y=NULL,
                 type='one-classification',
                 nu=el,
                 kernel="radial")
  
  svm.predtrain<-predict(svm.model, X)
  out <- rep(0, n)
  out[svm.predtrain==FALSE] <- 1
  roc_lof<- roc(response = is_out, predictor = out)
  auc_svm <- c(auc_svm ,auc(roc_lof))
  confusion_matrix<-table(SVM=out, Actual=is_out)
  precision <- c(precision, confusion_matrix[2,2]/sum(confusion_matrix[2,]))
  recall <- c(recall, confusion_matrix[2,2]/sum(confusion_matrix[,2]))
}
auc_svm; precision; recall

#BEST SVM NU: nu =  ; AUC =  ; precision =  ; recall = 

# OCSVM paper with k=7 and k=Best knn value:
#K=7:
distanze<-kNNdist(X,k=7)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))
if (nu>1){
  nu <- 0.5
}
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.526
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.033
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.387

#K=BEST knn:
distanze<-kNNdist(X,k=13)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))
if (nu>1){
  nu <- 0.5
}
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #

#BEST SVM K: k =  ; AUC =  ; precision =  ; recall = 

#ForestCover SCALATO ----
X <- scale(X)

### LOF:
#Alta dimensionalità

### SVM:

# nu with 4 values:
nu_val <- c(0.01, 0.1, 0.9, 0.5)
auc_svm <- c()
precision <- c()
recall <- c()
for (el in nu_val){
  svm.model<-svm(X, y=NULL,
                 type='one-classification',
                 nu=el,
                 kernel="radial")
  
  svm.predtrain<-predict(svm.model, X)
  out <- rep(0, n)
  out[svm.predtrain==FALSE] <- 1
  roc_lof<- roc(response = is_out, predictor = out)
  auc_svm <- c(auc_svm ,auc(roc_lof))
  confusion_matrix<-table(SVM=out, Actual=is_out)
  precision <- c(precision, confusion_matrix[2,2]/sum(confusion_matrix[2,]))
  recall <- c(recall, confusion_matrix[2,2]/sum(confusion_matrix[,2]))
}
auc_svm; precision; recall

#BEST SVM NU: nu =  ; AUC =  ; precision =  ; recall = 

# OCSVM paper with k=7 and k=Best knn value:
#K=7:
distanze<-kNNdist(X,k=7)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))
if (nu>1){
  nu <- 0.5
}
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc_fcscal<- roc(response = is_out, predictor = out)
auc(roc_svmoc_fcscal) #
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #

#K=BEST knn:
distanze<-kNNdist(X,k=13)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))
if (nu>1){
  nu <- 0.5
}
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc_fcscal2<- roc(response = is_out, predictor = out)
auc(roc_svmoc_fcscal2) #
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #


#BEST SVM K: k = 7 ; AUC =  ; precision =  ; recall = 

#ForestCover COMPLETO ----
#Su dataset completo:
dati=read.table("covtype.data", sep=",", header=F)
dati<-dati[which(dati$V55==2|dati$V55==4),]
is_out<-ifelse(dati$V55==4,1,0)
dati<-dati[,-55]
dati$is_out<-is_out
head(dati)

### LOF:
#Alta dimensionalità

### SVM:

# nu with 4 values:
nu_val <- c(0.01, 0.1, 0.9, 0.5)
auc_svm <- c()
precision <- c()
recall <- c()
for (el in nu_val){
  svm.model<-svm(X, y=NULL,
                 type='one-classification',
                 nu=el,
                 kernel="radial")
  
  svm.predtrain<-predict(svm.model, X)
  out <- rep(0, n)
  out[svm.predtrain==FALSE] <- 1
  roc_lof<- roc(response = is_out, predictor = out)
  auc_svm <- c(auc_svm ,auc(roc_lof))
  confusion_matrix<-table(SVM=out, Actual=is_out)
  precision <- c(precision, confusion_matrix[2,2]/sum(confusion_matrix[2,]))
  recall <- c(recall, confusion_matrix[2,2]/sum(confusion_matrix[,2]))
}
auc_svm; precision; recall

#BEST SVM NU: nu =  ; AUC =  ; precision =  ; recall = 

# OCSVM paper with k=7 and k=Best knn value:
#K=7:
distanze<-kNNdist(X,k=7)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))
if(nu>1){
  nu <- 0.5
}
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #

#K=BEST knn:
distanze<-kNNdist(X,k=53)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))
if(nu>1){
  nu <- 0.5
}
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #

#BEST SVM K: k =  ; AUC =  ; precision =  ; recall = 

#ForestCover COMPLETO SCALATO ----
X <- scale(X)

### LOF:
#Alta dimensionalità

### SVM:

# nu with 4 values:
nu_val <- c(0.01, 0.1, 0.9, 0.5)
auc_svm <- c()
precision <- c()
recall <- c()
for (el in nu_val){
  svm.model<-svm(X, y=NULL,
                 type='one-classification',
                 nu=el,
                 kernel="radial")
  
  svm.predtrain<-predict(svm.model, X)
  out <- rep(0, n)
  out[svm.predtrain==FALSE] <- 1
  roc_lof<- roc(response = is_out, predictor = out)
  auc_svm <- c(auc_svm ,auc(roc_lof))
  confusion_matrix<-table(SVM=out, Actual=is_out)
  precision <- c(precision, confusion_matrix[2,2]/sum(confusion_matrix[2,]))
  recall <- c(recall, confusion_matrix[2,2]/sum(confusion_matrix[,2]))
}
auc_svm; precision; recall

#BEST SVM NU: nu =  ; AUC =  ; precision =  ; recall = 

# OCSVM paper with k=7 and k=Best knn value:
#K=7:
distanze<-kNNdist(X,k=7)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))
if(nu>1){
  nu <- 0.5
}
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #

#K=BEST knn:
distanze<-kNNdist(X,k=53)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))
if(nu>1){
  nu <- 0.5
}
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #

#BEST SVM K: k =  ; AUC =  ; precision =  ; recall = 

# Smtp (KDDCUP99) - 30 outliers ----------------
path="smtp1.mat"
data=readMat(path)
X <- data$X
p=dim(X)[2]; n=dim(X)[1]
is_out <- data$y

### LOF:
#Alta dimensionalità

### SVM:

# nu with 4 values:
nu_val <- c(0.01, 0.1, 0.9, 0.5)
auc_svm <- c()
precision <- c()
recall <- c()
for (el in nu_val){
  svm.model<-svm(X, y=NULL,
                 type='one-classification',
                 nu=el,
                 kernel="radial")
  
  svm.predtrain<-predict(svm.model, X)
  out <- rep(0, n)
  out[svm.predtrain==FALSE] <- 1
  roc_lof<- roc(response = is_out, predictor = out)
  auc_svm <- c(auc_svm ,auc(roc_lof))
  confusion_matrix<-table(SVM=out, Actual=is_out)
  precision <- c(precision, confusion_matrix[2,2]/sum(confusion_matrix[2,]))
  recall <- c(recall, confusion_matrix[2,2]/sum(confusion_matrix[,2]))
}
auc_svm; precision; recall

#BEST SVM NU: nu = 0.1 ; AUC = 0.8334 ; precision = 0.0024 ; recall = 0.767

# OCSVM paper with k=7 and k=Best knn value:
#K=7:
distanze<-kNNdist(X,k=7)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))
if(nu>1){
  nu <- 0.5
}
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.6334
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.0004
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.767

#K=BEST knn:
distanze<-kNNdist(X,k=8)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))
if(nu>1){
  nu <- 0.5
}
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.6334
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.0004
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.767

#BEST SVM K: k = 7 ; AUC = 0.6334 ; precision = 0.0004 ; recall = 0.767

#Smtp SCALATO ----
X <- scale(X)

### LOF:
#Alta dimensionalità

### SVM:

# nu with 4 values:
nu_val <- c(0.01, 0.1, 0.9, 0.5)
auc_svm <- c()
precision <- c()
recall <- c()
for (el in nu_val){
  svm.model<-svm(X, y=NULL,
                 type='one-classification',
                 nu=el,
                 kernel="radial")
  
  svm.predtrain<-predict(svm.model, X)
  out <- rep(0, n)
  out[svm.predtrain==FALSE] <- 1
  roc_lof<- roc(response = is_out, predictor = out)
  auc_svm <- c(auc_svm ,auc(roc_lof))
  confusion_matrix<-table(SVM=out, Actual=is_out)
  precision <- c(precision, confusion_matrix[2,2]/sum(confusion_matrix[2,]))
  recall <- c(recall, confusion_matrix[2,2]/sum(confusion_matrix[,2]))
}
auc_svm; precision; recall

#BEST SVM NU: nu = 0.1 ; AUC = 0.8334 ; precision = 0.002 ; recall = 0.767

# OCSVM paper with k=7 and k=Best knn value:
#K=7:
distanze<-kNNdist(X,k=7)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))
if(nu>1){
  nu <- 0.5
}
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.7001
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.001
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.9

#K=BEST knn:
distanze<-kNNdist(X,k=8)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))
if(nu>1){
  nu <- 0.5
}
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.7001
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.001
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.9

#BEST SVM K: k = 7 ; AUC = 0.7001 ; precision = 0.001 ; recall = 0.9

# Mammography - 260 outliers -------------------
path="mammography.mat"
data=readMat(path)
X <- data$X
p=dim(X)[2]; n=dim(X)[1]
is_out <- data$y
dati<-data.frame(cbind(X, is_out)) 

### LOF:
outlier.scores <- lof(X, k=mean(c(log(n), p+1, 2*p)))

rfind <- rpart_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=18)
abline(h=rfind, col=4)
rbest_taglio <- max(rfind)
out_previsti <- rep(0,n)
out_previsti <- ifelse(outlier.scores>rbest_taglio, 1, 0)

#Confusion Matrix & AUC:
table(LOF=out_previsti, Actual=is_out)
#Valutazione performance:
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.513
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.16
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.032

# cfind <- ctree_split_finder(outlier.scores, n)
# plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=19)
# abline(h=cfind, col=cfind+1) 
# cbest_taglio <- sort(cfind)[1]
# out_previsti=rep(0,n)
# out_previsti <- ifelse(outlier.scores>cbest_taglio, 1, 0)
# roc_lof<- roc(response = is_out, predictor = out_previsti)
# auc(roc_lof) #
# confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
# precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #
# recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #

#BEST LOF: taglio = 1.9 ; AUC = 0.513 ; precision = 0.16 ; recall = 0.032

### SVM:

# nu with 4 values:
nu_val <- c(0.01, 0.1, 0.9, 0.5)
auc_svm <- c()
precision <- c()
recall <- c()
for (el in nu_val){
  svm.model<-svm(X, y=NULL,
                 type='one-classification',
                 nu=el,
                 kernel="radial")
  
  svm.predtrain<-predict(svm.model, X)
  out <- rep(0, n)
  out[svm.predtrain==FALSE] <- 1
  roc_lof<- roc(response = is_out, predictor = out)
  auc_svm <- c(auc_svm ,auc(roc_lof))
  confusion_matrix<-table(SVM=out, Actual=is_out)
  precision <- c(precision, confusion_matrix[2,2]/sum(confusion_matrix[2,]))
  recall <- c(recall, confusion_matrix[2,2]/sum(confusion_matrix[,2]))
}
auc_svm; precision; recall

#BEST SVM NU: nu = 0.5 ; AUC = 0.7756 ; precision = 0.054 ; recall = 0.938

# OCSVM paper with k=7 and k=Best knn value:
#K=7:
distanze<-kNNdist(X,k=7)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))
nu <- 0.5
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.8006
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.062
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.938

#K=BEST knn:
distanze<-kNNdist(X,k=8)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))
nu <- 0.5
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.8011
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.062
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.938

#BEST SVM K: k = 8 ; AUC = 0.801 ; precision = 0.062 ; recall = 0.938

#Mammography SCALATO ----
X <- scale(X)

### LOF:
outlier.scores <- lof(X, k=mean(c(log(n), p+1, 2*p)))

rfind <- rpart_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=18)
abline(h=rfind, col=4)
rbest_taglio <- sort(rfind)[length(rfind)-1]
out_previsti <- rep(0,n)
out_previsti <- ifelse(outlier.scores>rbest_taglio, 1, 0)

#Confusion Matrix & AUC:
table(LOF=out_previsti, Actual=is_out)
#Valutazione performance:
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.5278
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.162
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.067

# cfind <- ctree_split_finder(outlier.scores, n)
# plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=19)
# abline(h=cfind, col=cfind+1)  
# cbest_taglio <- sort(cfind)
# out_previsti=rep(0,n)
# out_previsti <- ifelse(outlier.scores>cbest_taglio, 1, 0)
# roc_lof<- roc(response = is_out, predictor = out_previsti)
# auc(roc_lof) #
# confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
# precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0
# recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0

#BEST LOF: taglio = 1.6 ; AUC = 0.5278 ; precision = 0.162 ; recall = 0.067

### SVM:

# nu with 4 values:
nu_val <- c(0.01, 0.1, 0.9, 0.5)
auc_svm <- c()
precision <- c()
recall <- c()
for (el in nu_val){
  svm.model<-svm(X, y=NULL,
                 type='one-classification',
                 nu=el,
                 kernel="radial")
  
  svm.predtrain<-predict(svm.model, X)
  out <- rep(0, n)
  out[svm.predtrain==FALSE] <- 1
  roc_lof<- roc(response = is_out, predictor = out)
  auc_svm <- c(auc_svm ,auc(roc_lof))
  confusion_matrix<-table(SVM=out, Actual=is_out)
  precision <- c(precision, confusion_matrix[2,2]/sum(confusion_matrix[2,]))
  recall <- c(recall, confusion_matrix[2,2]/sum(confusion_matrix[,2]))
}
auc_svm; precision; recall

#BEST SVM NU: nu = 0.5 ; AUC = 0.7756 ; precision = 0.054 ; recall = 0.938
#( nu=0.1, prec=0.136, rec=0.546, auc=0.7319)

# OCSVM paper with k=7 and k=Best knn value:
#K=7:
distanze<-kNNdist(X,k=7)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))
if(nu>1){
  nu <- 0.5
}
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.8006
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.062
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.938

#K=BEST knn:
distanze<-kNNdist(X,k=8)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))
if(nu>1){
  nu <- 0.5
}
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.8011
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.062
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.938

#BEST SVM K: k = 8 ; AUC =  0.8011 ; precision =0.062  ; recall = 0.938


# Annthyroid - 534 outliers --------------------
path <-"annthyroid.mat"
data=readMat(path)
X=data$X
p <- dim(X)[2]; n <- dim(X)[1]
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))

### LOF:
outlier.scores <- lof(X, k=mean(c(log(n), p+1, 2*p)))

rfind <- rpart_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=18)
abline(h=rfind, col=4) #non sensati, tagliamo al ginocchio
abline(h=1.5, col=2)
rbest_taglio <- 1.5
out_previsti <- rep(0,n)
out_previsti <- ifelse(outlier.scores>rbest_taglio, 1, 0)

#Confusion Matrix & AUC:
table(LOF=out_previsti, Actual=is_out)
#Valutazione performance:
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.5439
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.247
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.116

# cfind <- ctree_split_finder(outlier.scores, n)
# plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=19)
# abline(h=cfind, col=cfind+1) 
# cbest_taglio <- sort(cfind)[1]
# out_previsti=rep(0,n)
# out_previsti <- ifelse(outlier.scores>cbest_taglio, 1, 0)
# roc_lof<- roc(response = is_out, predictor = out_previsti)
# auc(roc_lof) #
# confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
# precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #
# recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #

#BEST LOF: taglio = 1.5 ; AUC = 0.5439 ; precision = 0.247 ; recall = 0.116

### SVM:

# nu with 4 values:
nu_val <- c(0.01, 0.1, 0.9, 0.5)
auc_svm <- c()
precision <- c()
recall <- c()
for (el in nu_val){
  svm.model<-svm(X, y=NULL,
                 type='one-classification',
                 nu=el,
                 kernel="radial")
  
  svm.predtrain<-predict(svm.model, X)
  out <- rep(0, n)
  out[svm.predtrain==FALSE] <- 1
  roc_lof<- roc(response = is_out, predictor = out)
  auc_svm <- c(auc_svm ,auc(roc_lof))
  confusion_matrix<-table(SVM=out, Actual=is_out)
  precision <- c(precision, confusion_matrix[2,2]/sum(confusion_matrix[2,]))
  recall <- c(recall, confusion_matrix[2,2]/sum(confusion_matrix[,2]))
}
auc_svm; precision; recall

#BEST SVM NU: nu = 0.5 ; AUC = 0.6193 ; precision = 0.107 ; recall = 0.721

# OCSVM paper with k=7 and k=Best knn value:
#K=7:
distanze<-kNNdist(X,k=7)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK)); nu
if(nu>1){
  nu <- 0.5
}
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.5405
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.085
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.575

#K=BEST knn:
distanze<-kNNdist(X,k=9)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))
if(abs(nu)>1){
  nu <- 0.5
}
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.5414
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.0855
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.577

#BEST SVM K: k = 9 ; AUC =  0.5414 ; precision = 0.0855 ; recall = 0.577

#Annthyroid SCALATO -----
X <- scale(X)
### LOF:
outlier.scores <- lof(X, k=mean(c(log(n), p+1, 2*p)))

rfind <- rpart_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=18)
abline(h=rfind, col=4) #non sensate, ginocchio
abline(h=1.5, col=2)
rbest_taglio <- 1.5
out_previsti <- rep(0,n)
out_previsti <- ifelse(outlier.scores>rbest_taglio, 1, 0)

#Confusion Matrix & AUC:
table(LOF=out_previsti, Actual=is_out)
#Valutazione performance:
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.5278
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.185
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.086

# cfind <- ctree_split_finder(outlier.scores, n)
# plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=19)
# abline(h=cfind, col=cfind+1)
# cbest_taglio <- max(cfind)
# out_previsti=rep(0,n)
# out_previsti <- ifelse(outlier.scores>cbest_taglio, 1, 0)
# roc_lof<- roc(response = is_out, predictor = out_previsti)
# auc(roc_lof) #
# confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
# precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #
# recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #

#BEST LOF: taglio = 1.5 ; AUC = 0.5278 ; precision = 0.185 ; recall = 0.086

### SVM:

# nu with 4 values:
nu_val <- c(0.01, 0.1, 0.9, 0.5)
auc_svm <- c()
precision <- c()
recall <- c()
for (el in nu_val){
  svm.model<-svm(X, y=NULL,
                 type='one-classification',
                 nu=el,
                 kernel="radial")
  
  svm.predtrain<-predict(svm.model, X)
  out <- rep(0, n)
  out[svm.predtrain==FALSE] <- 1
  roc_lof<- roc(response = is_out, predictor = out)
  auc_svm <- c(auc_svm ,auc(roc_lof))
  confusion_matrix<-table(SVM=out, Actual=is_out)
  precision <- c(precision, confusion_matrix[2,2]/sum(confusion_matrix[2,]))
  recall <- c(recall, confusion_matrix[2,2]/sum(confusion_matrix[,2]))
}
auc_svm; precision; recall

#BEST SVM NU: nu = 0.5 ; AUC = 0.6193 ; precision = 0.107 ; recall = 0.721

# OCSVM paper with k=7 and k=Best knn value:
#K=7:
distanze<-kNNdist(X,k=7)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-(sum(SK)-x.max_cf)/sum(SK)
if(abs(nu)>1){
  nu <- 0.5
}
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.6173
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.106
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.717

#K=BEST knn:
distanze<-kNNdist(X,k=9)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-(sum(SK)-x.max_cf)/sum(SK)
if(abs(nu)>1){
  nu <- 0.5
}
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.6173
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.106
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.717

#BEST SVM K: k = 7/9 ; AUC = 0.6173 ; precision = 0.106 ; recall = 0.717

### Annthyroid COMPLETO ----
train <- read.csv("ann_train.csv", sep=" ", header=F)
test <- read.csv("ann_test.csv", sep=" ", header=F)
dati <- rbind(train, test)[-c(23,24)]
p <- dim(dati)[2]-1; n <- dim(dati)[1]
fac <- c(2:16)
is_out <- dati$V22
is_out <- ifelse(is_out==3, 0, 1)
for (i in 1:length(fac)){
  dati[,i]<-as.factor(dati[,i])
}
#One hot encoding:
dati<-model.matrix(~.-1,dati[,-(p+1)])
X <- dati[,1:(ncol(dati)-1)]
p=dim(X)[2]; n=dim(X)[1]

### LOF:
outlier.scores <- lof(X, k=mean(c(log(n), p+1, 2*p)))

rfind <- rpart_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=18)
abline(h=rfind, col=4)
abline(h=1.07, col=2)
rbest_taglio <- 1.07
out_previsti <- rep(0,n)
out_previsti <- ifelse(outlier.scores>rbest_taglio, 1, 0)

#Confusion Matrix & AUC:
table(LOF=out_previsti, Actual=is_out)
#Valutazione performance:
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.4845
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.044
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.05

cfind <- ctree_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=19)
abline(h=cfind, col=cfind+1) #nessun taglio
# cbest_taglio <- sort(cfind)[1]
# out_previsti=rep(0,n)
# out_previsti <- ifelse(outlier.scores>cbest_taglio, 1, 0)
# roc_lof<- roc(response = is_out, predictor = out_previsti)
# auc(roc_lof) #
# confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
# precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #
# recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #

#BEST LOF: taglio = 1.07 ; AUC = 0.4845 ; precision = 0.047 ; recall = 0.049

### SVM:

# nu with 4 values:
nu_val <- c(0.01, 0.1, 0.9, 0.5)
auc_svm <- c()
precision <- c()
recall <- c()
for (el in nu_val){
  svm.model<-svm(X, y=NULL,
                 type='one-classification',
                 nu=el,
                 kernel="radial")
  
  svm.predtrain<-predict(svm.model, X)
  out <- rep(0, n)
  out[svm.predtrain==FALSE] <- 1
  roc_lof<- roc(response = is_out, predictor = out)
  auc_svm <- c(auc_svm ,auc(roc_lof))
  confusion_matrix<-table(SVM=out, Actual=is_out)
  precision <- c(precision, confusion_matrix[2,2]/sum(confusion_matrix[2,]))
  recall <- c(recall, confusion_matrix[2,2]/sum(confusion_matrix[,2]))
}
auc_svm; precision; recall

#BEST SVM NU: nu = 0.5 ; AUC =0.531  ; precision = 0.082 ; recall = 0.558

# OCSVM paper with k=7 and k=Best knn value:
#K=7:
distanze<-kNNdist(X,k=7)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))
nu <- 0.5
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.5141
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.07
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.474

#K=BEST knn:
distanze<-kNNdist(X,k=64)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.5141
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.07
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.326

#BEST SVM K: k = 7 ; AUC = 0.5141 ; precision = 0.07 ; recall = 0.474

#Annthyroid COMPLETO SCALATO -----
X <- scale(X)
### LOF:
outlier.scores <- lof(X, k=mean(c(log(n), p+1, 2*p)))

rfind <- rpart_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=18)
abline(h=rfind, col=4)
rbest_taglio <- sort(rfind)[2] 
out_previsti <- rep(0,n)
out_previsti <- ifelse(outlier.scores>rbest_taglio, 1, 0)

#Confusion Matrix & AUC:
table(LOF=out_previsti, Actual=is_out)
#Valutazione performance:
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.5112
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.0877
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.135

cfind <- ctree_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=19)
abline(h=cfind, col=cfind+1) #nessuno split
# cbest_taglio <- max(cfind)
# out_previsti=rep(0,n)
# out_previsti <- ifelse(outlier.scores>cbest_taglio, 1, 0)
# roc_lof<- roc(response = is_out, predictor = out_previsti)
# auc(roc_lof) #
# confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
# precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #
# recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #

#BEST LOF: taglio = 1.6 ; AUC = 0.5112 ; precision = 0.0877 ; recall = 0.135

### SVM:

# nu with 4 values:
nu_val <- c(0.01, 0.1, 0.9, 0.5)
auc_svm <- c()
precision <- c()
recall <- c()
for (el in nu_val){
  svm.model<-svm(X, y=NULL,
                 type='one-classification',
                 nu=el,
                 kernel="radial")
  
  svm.predtrain<-predict(svm.model, X)
  out <- rep(0, n)
  out[svm.predtrain==FALSE] <- 1
  roc_lof<- roc(response = is_out, predictor = out)
  auc_svm <- c(auc_svm ,auc(roc_lof))
  confusion_matrix<-table(SVM=out, Actual=is_out)
  precision <- c(precision, confusion_matrix[2,2]/sum(confusion_matrix[2,]))
  recall <- c(recall, confusion_matrix[2,2]/sum(confusion_matrix[,2]))
}
auc_svm; precision; recall

#BEST SVM NU: nu = 0.5 ; AUC = 0.522 ; precision =0.08  ; recall = 0.541

# OCSVM paper with k=7 and k=Best knn value:
#K=7:
distanze<-kNNdist(X,k=7)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-(sum(SK)-x.max_cf)/sum(SK)
if(abs(nu>1)){
  nu <- 0.5
}
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.5371
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.083
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.637

#K=BEST knn:
distanze<-kNNdist(X,k=119)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-(sum(SK)-x.max_cf)/sum(SK)
if(abs(nu)>1){
  nu <- 0.5
}
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.5024
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.074
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.848

#BEST SVM K: k = 7 ; AUC = 0.5371 ; precision = 0.083 ; recall = 0.637

# Pendigits - 156 outliers ---------------------
path <-"pendigits.mat"
data <- readMat(path)
X=as.data.frame(data$X)
p=dim(X)[2]; n=dim(X)[1]
is_out=data$y
dati<-data.frame(cbind(X, is_out)) 

### LOF:
outlier.scores <- lof(X, k=mean(c(log(n), p+1, 2*p)))

rfind <- rpart_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=18)
abline(h=rfind, col=4)
rbest_taglio <- sort(rfind)[2] 
out_previsti <- rep(0,n)
out_previsti <- ifelse(outlier.scores>rbest_taglio, 1, 0)

#Confusion Matrix & AUC:
table(LOF=out_previsti, Actual=is_out)
#Valutazione performance:
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.5089
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.25
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.19

cfind <- ctree_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=19)
abline(h=cfind, col=cfind+1) #nessun taglio
# cbest_taglio <- sort(cfind) 
# out_previsti=rep(0,n)
# out_previsti[out_previsti>numero]=1
# confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
# precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision
# recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall

#BEST LOF: taglio = 2 ; AUC = 0.5089 ; precision = 0.25 ; recall = 0.019

### SVM:

# nu with 4 values:
nu_val <- c(0.01, 0.1, 0.9, 0.5)
auc_svm <- c()
precision <- c()
recall <- c()
for (el in nu_val){
  svm.model<-svm(X, y=NULL,
                 type='one-classification',
                 nu=el,
                 kernel="radial")
  
  svm.predtrain<-predict(svm.model, X)
  out <- rep(0, n)
  out[svm.predtrain==FALSE] <- 1
  roc_lof<- roc(response = is_out, predictor = out)
  auc_svm <- c(auc_svm ,auc(roc_lof))
  confusion_matrix<-table(SVM=out, Actual=is_out)
  precision <- c(precision, confusion_matrix[2,2]/sum(confusion_matrix[2,]))
  recall <- c(recall, confusion_matrix[2,2]/sum(confusion_matrix[,2]))
  
}
auc_svm; precision; recall

#BEST SVM NU: nu = 0.5 ; AUC = 0.7557 ; precision = 0.045 ; recall = 1.00

# OCSVM paper with k=7 and k=Best knn value:
#K=7:
distanze<-kNNdist(X,k=7)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))
if(nu>1){nu <- 0.5}
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,  
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.7557
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.045
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #1.00


#K=BEST knn:
distanze<-kNNdist(X,k=17)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))
if(nu>1){nu <- 0.5}
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE, 
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.7556
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.045
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #1.00

#BEST SVM K: k = 7 ; AUC = 0.7557 ; precision = 0.045 ; recall = 1.00

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
p=dim(X)[2]; n=dim(X)[1]

### LOF:
outlier.scores <- lof(X, k=mean(c(log(n), p+1, 2*p)))

rfind <- rpart_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=18)
abline(h=rfind, col=4) #proponiamo taglio più sensato
abline(h=1.63, col=2)
rbest_taglio <- 1.63
out_previsti <- rep(0,n)
out_previsti <- ifelse(outlier.scores>rbest_taglio, 1, 0)

#Confusion Matrix & AUC:
table(LOF=out_previsti, Actual=is_out)
#Valutazione performance:
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.6483
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.2
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.333

cfind <- ctree_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=19)
abline(h=cfind, col=cfind+1)
cbest_taglio <- sort(cfind) #nessun taglio con ctree
# out_previsti=rep(0,n)
# out_previsti[out_previsti>numero]=1
# confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
# precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); 

#BEST LOF: taglio = 1.63 ; AUC = 0.6483 ; precision = 0.2 ; recall = 0.333

### SVM:

# nu with 4 values:
nu_val <- c(0.01, 0.1, 0.9, 0.5)
auc_svm <- c()
precision <- c()
recall <- c()
for (el in nu_val){
  svm.model<-svm(X, y=NULL,
                 type='one-classification',
                 nu=el,
                 kernel="radial")
  
  svm.predtrain<-predict(svm.model, X)
  out <- rep(0, n)
  out[svm.predtrain==FALSE] <- 1
  roc_lof<- roc(response = is_out, predictor = out)
  auc_svm <- c(auc_svm ,auc(roc_lof))
  confusion_matrix<-table(SVM=out, Actual=is_out)
  precision <- c(precision, confusion_matrix[2,2]/sum(confusion_matrix[2,]))
  recall <- c(recall, confusion_matrix[2,2]/sum(confusion_matrix[,2])
)
}
auc_svm; precision; recall

#BEST SVM NU: nu = 0.1 ; AUC = 0.7936 ; precision = 0.187 ; recall = 0.667

# OCSVM paper with k=7 and k=Best knn value:
#K=7:
distanze<-kNNdist(X,k=7)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))
if(nu>1){nu <- 0.5}
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,  
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.6998
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.047
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.889

#K=BEST knn:
distanze<-kNNdist(X,k=5)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))
if(nu>1){nu <- 0.5}
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE, 
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.6998
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.048
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.889

#BEST SVM K: k = 5/7 ; AUC = 0.6998 ; precision = 0.048 ; recall = 0.889

# Wine - 10 outliers ---------------------------
path <- "wine.mat"
data=readMat(path)
X <- data$X
p=dim(X)[2]; n=dim(X)[1]
is_out <- data$y

### LOF:
outlier.scores <- lof(X, k=mean(c(log(n), p+1, 2*p)))
rfind <- rpart_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=18)
abline(h=rfind, col=4)
rbest_taglio <- sort(rfind)[3]
out_previsti <- rep(0,n)
out_previsti <- ifelse(outlier.scores>rbest_taglio, 1, 0)
#Confusion Matrix & AUC:
table(LOF=out_previsti, Actual=is_out)
#Valutazione performance:
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.829
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.583
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.7

cfind <- ctree_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=19)
abline(h=cfind, col=cfind+1) #nessun taglio
cbest_taglio <- sort(cfind)[1]
out_previsti=rep(0,n)
out_previsti <- ifelse(outlier.scores>cbest_taglio, 1, 0)
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.8706
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.533
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.8

#BEST LOF: taglio = 1.3676 ; AUC = 0.8706 ; precision = 0.533 ; recall = 0.8

### SVM:

# nu with 4 values:
nu_val <- c(0.01, 0.1, 0.9, 0.5)
auc_svm <- c()
precision <- c()
recall <- c()
for (el in nu_val){
  svm.model<-svm(X, y=NULL,
                 type='one-classification',
                 nu=el,
                 kernel="radial")
  
  svm.predtrain<-predict(svm.model, X)
  out <- rep(0, n)
  out[svm.predtrain==FALSE] <- 1
  roc_lof<- roc(response = is_out, predictor = out)
  auc_svm <- c(auc_svm ,auc(roc_lof))
  confusion_matrix<-table(SVM=out, Actual=is_out)
  precision <- c(precision, confusion_matrix[2,2]/sum(confusion_matrix[2,]))
  recall <- c(recall, confusion_matrix[2,2]/sum(confusion_matrix[,2]))
}
auc_svm; precision; recall

#BEST SVM NU: nu = 0.5 ; AUC = 0.6605 ; precision = 0.123 ; recall = 0.8

# OCSVM paper with k=7 and k=Best knn value:
#K=7:
distanze<-kNNdist(X,k=7)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))

svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.5504
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.085
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #1.00

#K=BEST knn:
distanze<-kNNdist(X,k=14)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.5294
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.081
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #1.00

#BEST SVM K: k = 7 ; AUC = 0.5504 ; precision = 0.085 ; recall = 1.00

#Wine SCALATO ----
X <- scale(X)

### LOF:
outlier.scores <- lof(X, k=mean(c(log(n), p+1, 2*p)))

rfind <- rpart_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=18)
abline(h=rfind, col=4)
rbest_taglio <- max(rfind)
out_previsti <- rep(0,n)
out_previsti <- ifelse(outlier.scores>rbest_taglio, 1, 0)

#Confusion Matrix & AUC:
table(LOF=out_previsti, Actual=is_out)
#Valutazione performance:
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.4412
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0

cfind <- ctree_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=19)
abline(h=cfind, col=cfind+1) #nessun senso
# cbest_taglio <- 3
# out_previsti=rep(0,n)
# out_previsti <- ifelse(outlier.scores>cbest_taglio, 1, 0)
# roc_lof<- roc(response = is_out, predictor = out_previsti)
# auc(roc_lof) #0.4974
# confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
# precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0
# recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0

#BEST LOF: taglio = 1.23 ; AUC = 0.4412 ; precision = 0 ; recall = 0

### SVM:

# nu with 4 values:
nu_val <- c(0.01, 0.1, 0.9, 0.5)
auc_svm <- c()
precision <- c()
recall <- c()
for (el in nu_val){
  svm.model<-svm(X, y=NULL,
                 type='one-classification',
                 nu=el,
                 kernel="radial")
  
  svm.predtrain<-predict(svm.model, X)
  out <- rep(0, n)
  out[svm.predtrain==FALSE] <- 1
  roc_lof<- roc(response = is_out, predictor = out)
  auc_svm <- c(auc_svm ,auc(roc_lof))
  confusion_matrix<-table(SVM=out, Actual=is_out)
  precision <- c(precision, confusion_matrix[2,2]/sum(confusion_matrix[2,]))
  recall <- c(recall, confusion_matrix[2,2]/sum(confusion_matrix[,2]))
}
auc_svm; precision; recall

#BEST SVM NU: nu = 0.5 ; AUC = 0.6605 ; precision = 0.123 ; recall = 0.8

# OCSVM paper with k=7 and k=Best knn value:
#K=7:
distanze<-kNNdist(X,k=7)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))
if(nu>1){nu <- 0.5}
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.5609
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.063
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.5

#K=BEST knn:
distanze<-kNNdist(X,k=14)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))
if(nu>1){nu <- 0.5}
svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.5769
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.097
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.7

#BEST SVM K: k = 14 ; AUC = 0.5769 ; precision = 0.097 ; recall = 0.7

# Vertebral - 30 outliers -----------------------
path <- "vertebral.mat"
data=readMat(path)
X <- data$X
p=dim(X)[2]; n=dim(X)[1]
is_out <- data$y
dati<-data.frame(cbind(X, is_out)) 

### LOF:
outlier.scores <- lof(X, k=mean(c(log(n), p+1, 2*p)))
rfind <- rpart_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=18)
abline(h=rfind, col=4)
rbest_taglio <- max(rfind)
out_previsti <- rep(0,n)
out_previsti <- ifelse(outlier.scores>rbest_taglio, 1, 0)
#Confusion Matrix & AUC:
table(LOF=out_previsti, Actual=is_out)
#Valutazione performance:
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.4714
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.1
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.2

cfind <- ctree_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=19)
abline(h=cfind, col=cfind+1) #nessun taglio
# cbest_taglio <- sort(cfind)[1]
# out_previsti=rep(0,n)
# out_previsti <- ifelse(outlier.scores>cbest_taglio, 1, 0)
# roc_lof<- roc(response = is_out, predictor = out_previsti)
# auc(roc_lof) #
# confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
# precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #
# recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #

#BEST LOF: taglio = 1.17 ; AUC = 0.4717 ; precision = 0.1 ; recall = 0.2

### SVM:

# nu with 4 values:
nu_val <- c(0.01, 0.1, 0.9, 0.5)
auc_svm <- c()
precision <- c()
recall <- c()
for (el in nu_val){
  svm.model<-svm(X, y=NULL,
                 type='one-classification',
                 nu=el,
                 kernel="radial")
  
  svm.predtrain<-predict(svm.model, X)
  out <- rep(0, n)
  out[svm.predtrain==FALSE] <- 1
  roc_lof<- roc(response = is_out, predictor = out)
  auc_svm <- c(auc_svm ,auc(roc_lof))
  confusion_matrix<-table(SVM=out, Actual=is_out)
  precision <- c(precision, confusion_matrix[2,2]/sum(confusion_matrix[2,]))
  recall <- c(recall, confusion_matrix[2,2]/sum(confusion_matrix[,2]))
}
auc_svm; precision; recall

#BEST SVM NU: nu = 0.5  ; AUC = 0.5762 ; precision = 0.092 ; recall = 0.367

# OCSVM paper with k=7 and k=Best knn value:
#K=7:
distanze<-kNNdist(X,k=7)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))

svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #

#K=BEST knn:
#Nessun k trovato

#BEST SVM K: k = 7 ; AUC = 0.4929 ; precision = 0.123 ; recall = 0.867

#vertebral SCALATO ----
X <- scale(X)

### LOF:
outlier.scores <- lof(X, k=mean(c(log(n), p+1, 2*p)))

rfind <- rpart_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=18)
abline(h=rfind, col=4)
rbest_taglio <- max(rfind)
out_previsti <- rep(0,n)
out_previsti <- ifelse(outlier.scores>rbest_taglio, 1, 0)

#Confusion Matrix & AUC:
table(LOF=out_previsti, Actual=is_out)
#Valutazione performance:
roc_lof<- roc(response = is_out, predictor = out_previsti)
auc(roc_lof) #0.5333
confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.144
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.4333

cfind <- ctree_split_finder(outlier.scores, n)
plot(sort(outlier.scores), col=is_out[order(outlier.scores)]+1, pch=19)
abline(h=cfind, col=cfind+1) #nessun taglio
# out_previsti=rep(0,n)
# out_previsti <- ifelse(outlier.scores>cbest_taglio, 1, 0)
# roc_lof<- roc(response = is_out, predictor = out_previsti)
# auc(roc_lof) #
# confusion_matrix<-table(LOF=out_previsti, Actual=is_out)
# precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #
# recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #

#BEST LOF: taglio = 1.1 ; AUC = 0.5333 ; precision = 0.144 ; recall = 0.433

### SVM:

# nu with 4 values:
nu_val <- c(0.01, 0.1, 0.9, 0.5)
auc_svm <- c()
precision <- c()
recall <- c()
for (el in nu_val){
  svm.model<-svm(X, y=NULL,
                 type='one-classification',
                 nu=el,
                 kernel="radial")
  
  svm.predtrain<-predict(svm.model, X)
  out <- rep(0, n)
  out[svm.predtrain==FALSE] <- 1
  roc_lof<- roc(response = is_out, predictor = out)
  auc_svm <- c(auc_svm ,auc(roc_lof))
  confusion_matrix<-table(SVM=out, Actual=is_out)
  precision <- c(precision, confusion_matrix[2,2]/sum(confusion_matrix[2,]))
  recall <- c(recall, confusion_matrix[2,2]/sum(confusion_matrix[,2]))
}
auc_svm; precision; recall

#BEST SVM NU: nu = 0.5 ; AUC =  0.5762 ; precision = 0.092 ; recall = 0.367

# OCSVM paper with k=7 and k=Best knn value:
#K=7:
distanze<-kNNdist(X,k=7)
si_K<-rowMeans(distanze)
plot(sort(si_K), type="l", lwd=2)
x<-1:n
si_K<-sort(si_K)
fit = smooth.spline(x, si_K, cv=TRUE)
fit$lambda
fit_opt<-smooth.spline(x, si_K, lambda = fit$lambda)
plot(x,si_K ,col="lightgray")
lines(x, predict(fit_opt)$y)
d2<-predict(fit_opt,x,deriv=2)$y
d1<-predict(fit_opt,x,deriv=1)$y
CF<-d2/sqrt(1+d1^2)
x.max_cf<-which.max(CF)
gamma<-1/si_K[x.max_cf]
SK<-si_K[1:x.max_cf]
nu<-abs((sum(SK)-x.max_cf)/sum(SK))

svm.model<-svm(X, y=NULL,
               type='one-classification',
               nu=nu, gamma=gamma,
               scale=FALSE,
               kernel="radial")

svm.predtrain<-predict(svm.model, X)
out <- rep(0, n)
out[svm.predtrain==FALSE] <- 1
#Confusion Matrix & AUC:
table(SVMOneClass=out, Actual=is_out)
#Valutazione performance:
roc_svmoc<- roc(response = is_out, predictor = out)
auc(roc_svmoc) #0.469
confusion_matrix<-table(SVM=out, Actual=is_out)
precision <- confusion_matrix[2,2]/sum(confusion_matrix[2,]); precision #0.069
recall <- confusion_matrix[2,2]/sum(confusion_matrix[,2]); recall #0.067

#K=BEST knn:
#Nessun k trovato

#BEST SVM K: k = 7 ; AUC = 0.469 ; precision = 0.069 ; recall = 0.067

### FINE ###################################################################################################