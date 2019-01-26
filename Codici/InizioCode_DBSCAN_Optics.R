
# Lympho ------------------------------------------------------------------



rm(list=ls())
setwd("C:/Users/federico/Desktop/Progetto_sl")


data=readMat("lympho.mat")
X=data$X
Y=data$y
X


scaled.dat <- scale(X)

# check that we get mean of 0 and sd of 1
round(colMeans(scaled.dat),2)  # faster version of apply(scaled.dat, 2, mean)
apply(scaled.dat, 2, sd)

p=dim(X)[2]
n=dim(X)[1]
k=max(p+1,log(n))
kNNdistplot(X,k=max(p+1,log(n)))
d=dist(X)
abline(h=2.9,col=2)
eps=2.9

res=dbscan(X,eps=eps,minPts = k)
indicatore=res$cluster==0
out=rep(0,n)
out[indicatore]=1
sum(out)
##outlier reali
sum(Y)
t3=table(Y,out)
t3

##normalizzati

kNNdistplot(scaled.dat,k=max(p+1,log(n)))
d=dist(scaled.dat)
abline(h=4.4,col=2)
eps=4.4

res=dbscan(scaled.dat,eps=eps,minPts = k)
indicatore=res$cluster==0
out=rep(0,n)
out[indicatore]=1
sum(out)
##outlier reali
sum(Y)
t3=table(Y,out)
t3




##OPTICS

res<-optics(X,eps=10,minPts = 20)
res
head(res$order,n=15)
plot(res)
plot(X,col="green")

polygon(X[res$order,],)

#2 MODI PER ESTRARLI
res<-extractDBSCAN(res,eps_cl=3)
plot(res)
hullplot(X,res)
sum(res$cluster==0)
#altro modo per estrarlo

res=extractXi(res,xi=0.01)
sum(res$cluster==0)


# WBC ---------------------------------------------------------------------


###devo cambiare distnze per togliere gli zeri
#devo inserire albero nel dbscan
#install.packages("pROC")
library(pROC)
library(rpart)
library(rpart.plot)
library(R.matlab)
library(dbscan)
library(partykit)





###codice operativo


rm(list=ls())
###FUNZIONI
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




setwd("C:/Users/federico/Desktop/Progetto_sl")

path="WBC.mat"

data=readMat(path)
X=data$X
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))

p=dim(X)[2]; n=dim(X)[1]


### DBSCAN 


#primo k, media tra i tre
k=mean(c(p+1,log(n),2*p))
#secondo k max tra i due 
#k2=max(p+1,log(n))
#3)
#k3=max(2*p,log(n))
a=kNNdist(X,k=k)
head(a)
kNNdistplot(X,k=k)
distanze=as.vector(a)
####inserire qui il metodo di taglio con "Alberelli"
head(distanze)


(valori=rpart_split_finder(distanze,length(distanze)))
#####
plot(sort(distanze),type="l")
##################

abline(h=valori,col=valori+1)
abline(h=0.7)
###scelta di buonsenso: nesusno di questi, 1


eps=0.7

res=dbscan(X,eps=eps,minPts = k)
indicatore=res$cluster==0
out_previsti=rep(0,n);out_previsti[indicatore]=1
sum(out_previsti)
##outlier reali
sum(is_out)
t3=table(is_out,out_previsti)
t3

hullplot(X,res)



#Confusion Matrix & AUC:
table(DBSCAN=out_previsti, Actual=is_out)
#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
(precision=t3[2,2]/sum(t3[2,]))
#[1] 0.4761905
# auc(roc_dbscan)
# Area under the curve: 0.7227
(recall=t3[2,2]/sum(t3[,2]))
#[1] 0.4761905
####OPTICS




res<-optics(scale(X),eps=30,minPts = k)
a=kNN(X,k)
head(a$id,3)
numerid=a$id
# numeridi_esimo=[i,]
axx<-kNNdist(X,k)
head(res$order,n=15)
plot(res)

#altro modo per estra1lo
distanzerech=res$reachdist
head(distanzerech)
distanzerech[1]=distanzerech[2]
#dist=mean(distanzerech[numerid[1,]])
dist=vector()
lrd=vector()
#for (i in 1:n){
#  minptsvicini=numerid[i,]
#  dist[i]=mean(distanzerech[numerid[i,]])
#  lrd[i]=1/dist[i]

#}

###modifica per evitare problemi dati uguali (dati ripetuti)
for (i in 1:n){
  minptsvicini=numerid[i,]
  dist[i]=mean(distanzerech[numerid[i,]])
  
  
}
distmagg=dist[dist>0]
valore=min(distmagg)
dist[dist==0]=valore/2
lrd=1/dist
#lrd[i]=1/dist[i]
summary(dist)
min(dist)
hist(dist)
dist
head(lrd,20)
numeratore=vector()
OF=vector()
for (i in 1:n){
  minptsvicini=numerid[i,]
  lrd_numero=lrd[i]
  lrd_minpts=lrd[numerid[i,]]
  numeratore[i]=sum(lrd_minpts/lrd_numero)
  OF[i]=numeratore[i]/k
}
summary(lrd)
lrd[max(lrd)]
sum(is.na(OF))
str(OF)
(sum=summary(OF))
str(sum)




###QUI INSERIRE ALBERELLI
(val_opt=rpart_split_finder(OF,length(OF)))
####

#OF RISCALATO
plot(exp(sort(OF)))
plot(exp(sort(OF)),type="l")
par(mfrow=c(1,1))
plot(sort(OF),ylim=c(sum[1],sum[6]))

plot(sort(OF),type="l",ylim=c(sum[1],sum[6]))

##alternativa
plot(sort(OF), col=is_out[order(OF)]+1)
abline(h=val_opt)
sort(val_opt)
##la soglia è il terzo valore piu alto
soglia=sort(val_opt)[3]
##taglio va effettuato 
out_previsti=rep(0,n)
out_previsti=ifelse(OF>soglia,1,0)
table(out_previsti)
(toptics=table(out_previsti,is_out))



#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)

#auc(roc_dbscan)
#Area under the curve: 0.6877



(precision=toptics[2,2]/sum(toptics[2,]))
#[1] 0.8

(recall=toptics[2,2]/sum(toptics[,2]))
#[1] 0.3809524



# GLASS and GLASS N -------------------------------------------------------





##GLASS

###devo cambiare distnze per togliere gli zeri
#devo inserire albero nel dbscan
#install.packages("pROC")
library(pROC)
library(rpart)
library(rpart.plot)
library(R.matlab)
library(dbscan)
library(partykit)





###codice operativo


rm(list=ls())
###FUNZIONI
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




setwd("C:/Users/federico/Desktop/Progetto_sl")

path="Glass.mat"

data=readMat(path)
X=data$X
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))

p=dim(X)[2]; n=dim(X)[1]


### DBSCAN non Norm ---------------------------


#primo k, media tra i tre
k=mean(c(p+1,log(n),2*p))
#secondo k max tra i due 
#k2=max(p+1,log(n))
#3)
#k3=max(2*p,log(n))
a=kNNdist(X,k=k)
head(a)
kNNdistplot(X,k=k)
distanze=as.vector(a)
####inserire qui il metodo di taglio con "Alberelli"
head(distanze)


(valori=rpart_split_finder(distanze,length(distanze)))
#
plot(sort(distanze),type="l")
#

abline(h=valori,col=valori+1)
abline(h=0.7)
###scelta di buonsenso: il massimo


eps=max(valori)

res=dbscan(X,eps=eps,minPts = k)
indicatore=res$cluster==0
out_previsti=rep(0,n);out_previsti[indicatore]=1
sum(out_previsti)
##outlier reali
sum(is_out)
t3=table(is_out,out_previsti)
t3

hullplot(X,res)



#Confusion Matrix & AUC:
table(DBSCAN=out_previsti, Actual=is_out)
#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
(precision=t3[2,2]/sum(t3[2,]))
#[1] 0.6666667

# auc(roc_dbscan)
#Area under the curve: 0.7431

(recall=t3[2,2]/sum(t3[,2]))
#[1] 0.1395349


##Dbscan norm ------------------------------

a=kNNdist(scale(X),k=k)
head(a)
kNNdistplot(scale(X),k=k)
distanze=as.vector(a)
####inserire qui il metodo di taglio con "Alberelli"
head(distanze)


(valori=rpart_split_finder(distanze,length(distanze)))
#
plot(sort(distanze),type="l")
#

abline(h=valori,col=valori+1)

plot(sort(distanze),type="l")
##

abline(h=2.1)
###scelta di buonsenso: nessuna di questa


eps=2

res=dbscan(scale(X),eps=eps,minPts = k)
indicatore=res$cluster==0
out_previsti=rep(0,n);out_previsti[indicatore]=1
sum(out_previsti)
##outlier reali
sum(is_out)
t3=table(is_out,out_previsti)
t3

hullplot(scale(X),res)



#Confusion Matrix & AUC:
table(DBSCAN=out_previsti, Actual=is_out)
#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
(precision=t3[2,2]/sum(t3[2,]))

(recall=t3[2,2]/sum(t3[,2]))


#Area under the curve: 0.6588
#> (precision=t3[2,2]/sum(t3[2,]))
#[1] 0.4444444
##> (recall=t3[2,2]/sum(t3[,2]))
#[1] 0.1333333

####OPTICS

###OF NON scalato --------------


res<-optics(X,eps=30,minPts = k)
a=kNN(X,k)
head(a$id,3)
numerid=a$id
# numeridi_esimo=[i,]
axx<-kNNdist(X,k)
head(res$order,n=15)
plot(res)

#altro modo per estra1lo
distanzerech=res$reachdist
head(distanzerech)
distanzerech[1]=distanzerech[2]
#dist=mean(distanzerech[numerid[1,]])
dist=vector()
lrd=vector()

###modifica per evitare problemi dati uguali (dati ripetuti)
for (i in 1:n){
  minptsvicini=numerid[i,]
  dist[i]=mean(distanzerech[numerid[i,]])
  
  
}
distmagg=dist[dist>0]
valore=min(distmagg)
dist[dist==0]=valore/2
lrd=1/dist
#lrd[i]=1/dist[i]
summary(dist)
min(dist)
hist(dist)
dist
head(lrd,20)
numeratore=vector()
OF=vector()
for (i in 1:n){
  minptsvicini=numerid[i,]
  lrd_numero=lrd[i]
  lrd_minpts=lrd[numerid[i,]]
  numeratore[i]=sum(lrd_minpts/lrd_numero)
  OF[i]=numeratore[i]/k
}
summary(lrd)
lrd[max(lrd)]
sum(is.na(OF))
str(OF)
(sum=summary(OF))
str(sum)




###QUI INSERIRE ALBERELLI
(val_opt=rpart_split_finder(OF,length(OF)))
(val_opt2=ctree_split_finder(OF,length(OF)))
####

#OF RISCALATO
plot(sort(OF),type="l",ylim=c(sum[1],sum[6]))

##alternativa
plot(sort(OF), col=is_out[order(OF)]+1)
abline(h=val_opt)
sort(val_opt)
##la soglia è il quarto valore piu alto
soglia=sort(val_opt)[4]
##taglio va effettuato 
out_previsti=rep(0,n)
out_previsti=ifelse(OF>soglia,1,0)
table(out_previsti)
(toptics=table(out_previsti,is_out))



#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
#auc(roc_dbscan)
#Area under the curve: 0.6301



(precision=toptics[2,2]/sum(toptics[2,]))
#[1] 0.1666667

(recall=toptics[2,2]/sum(toptics[,2]))
#[1] 0.3333333





###OPTICS RISCALATO -------------------------


X=scale(X)
res<-optics(X,eps=30,minPts = k)
a=kNN(X,k)
head(a$id,3)
numerid=a$id
# numeridi_esimo=[i,]
axx<-kNNdist(X,k)
head(res$order,n=15)
plot(res)

#altro modo per estra1lo
distanzerech=res$reachdist
head(distanzerech)
distanzerech[1]=distanzerech[2]
#dist=mean(distanzerech[numerid[1,]])
dist=vector();lrd=vector()

###modifica per evitare problemi dati uguali (dati ripetuti)
for (i in 1:n){
  minptsvicini=numerid[i,]
  dist[i]=mean(distanzerech[numerid[i,]])
  
  
}
distmagg=dist[dist>0]
valore=min(distmagg)
dist[dist==0]=valore/2
lrd=1/dist
#lrd[i]=1/dist[i]
summary(dist)
min(dist)
hist(dist)
dist
head(lrd,20)
numeratore=vector()
OF=vector()
for (i in 1:n){
  minptsvicini=numerid[i,]
  lrd_numero=lrd[i]
  lrd_minpts=lrd[numerid[i,]]
  numeratore[i]=sum(lrd_minpts/lrd_numero)
  OF[i]=numeratore[i]/k
}
summary(lrd)
lrd[max(lrd)]
sum(is.na(OF))
str(OF)
(sum=summary(OF))
str(sum)




###QUI INSERIRE ALBERELLI
(val_opt=rpart_split_finder(OF,length(OF)))
(val_opt2=ctree_split_finder(OF,length(OF)))
####

plot(sort(OF),type="l",ylim=c(sum[1],sum[6]))
##alternativa
plot(sort(OF), col=is_out[order(OF)]+1)

abline(h=val_opt)
abline(h=val_opt2)
##la soglia è secondo piu grande
soglia=sort(val_opt)[3]
##taglio va effettuato 
out_previsti=rep(0,n);out_previsti=ifelse(OF>soglia,1,0)
table(out_previsti)
(toptics=table(out_previsti,is_out))

#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
#auc(roc_dbscan)
#Area under the curve: 0.8591
(precision=toptics[2,2]/sum(toptics[2,]))
#[1] 0.1860465

(recall=toptics[2,2]/sum(toptics[,2]))
#[1] 0.8888889




# VOWELS ------------------------------------------------------------------



###VOWELS


rm(list=ls())
###devo cambiare distnze per togliere gli zeri
#devo inserire albero nel dbscan
#install.packages("pROC")
library(pROC)
library(rpart)
library(rpart.plot)
library(R.matlab)
library(dbscan)
library(partykit)

##gia scalato



###codice operativo


rm(list=ls())
###FUNZIONI
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




setwd("C:/Users/federico/Desktop/Progetto_sl")

path="Vowels.mat"

data=readMat(path)
X=data$X
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))

p=dim(X)[2]; n=dim(X)[1]


### DBSCAN non Norm ---------------------------


#primo k, media tra i tre
k=mean(c(p+1,log(n),2*p))
#secondo k max tra i due 
#k2=max(p+1,log(n))
#3)
#k3=max(2*p,log(n))
a=kNNdist(X,k=k)
head(a)
kNNdistplot(X,k=k)
distanze=as.vector(a)
####inserire qui il metodo di taglio con "Alberelli"
head(distanze)

###SPLIT
(valori=rpart_split_finder(distanze,length(distanze)))
(valori2=ctree_split_finder(distanze,length(distanze)))

#


plot(sort(distanze),type="l")
abline(h=valori,col=valori+1)


plot(sort(distanze),type="l")
abline(h=valori2,col=valori+1)
###scelta di buonsenso: nessuno di questi
abline(h=2.3)


eps=2.3

res=dbscan(X,eps=eps,minPts = k)
indicatore=res$cluster==0
out_previsti=rep(0,n);out_previsti[indicatore]=1
sum(out_previsti)
##outlier reali
sum(is_out)
t3=table(is_out,out_previsti)
t3

hullplot(X,res)

#Confusion Matrix & AUC:
table(DBSCAN=out_previsti, Actual=is_out)
#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
(precision=t3[2,2]/sum(t3[2,]))
#[1] [1] 0.5

# auc(roc_dbscan)
#Area under the curve: 0.7418

(recall=t3[2,2]/sum(t3[,2]))
#[[1] 0.5208333


####OPTICS

###OF NON scalato --------------


res<-optics(X,eps=30,minPts = k)
a=kNN(X,k)
head(a$id,3)
numerid=a$id
# numeridi_esimo=[i,]
axx<-kNNdist(X,k)
head(res$order,n=15)
plot(res)

#altro modo per estra1lo
distanzerech=res$reachdist
head(distanzerech)
distanzerech[1]=distanzerech[2]
#dist=mean(distanzerech[numerid[1,]])
dist=vector()
lrd=vector()

###modifica per evitare problemi dati uguali (dati ripetuti)
for (i in 1:n){
  minptsvicini=numerid[i,]
  dist[i]=mean(distanzerech[numerid[i,]])
  
  
}
distmagg=dist[dist>0]
valore=min(distmagg)
dist[dist==0]=valore/2
lrd=1/dist
#lrd[i]=1/dist[i]
summary(dist)
min(dist)
hist(dist)
dist
head(lrd,20)
numeratore=vector()
OF=vector()
for (i in 1:n){
  minptsvicini=numerid[i,]
  lrd_numero=lrd[i]
  lrd_minpts=lrd[numerid[i,]]
  numeratore[i]=sum(lrd_minpts/lrd_numero)
  OF[i]=numeratore[i]/k
}
summary(lrd)
lrd[max(lrd)]
sum(is.na(OF))
str(OF)
(sum=summary(OF))
str(sum)




###QUI INSERIRE ALBERELLI
(val_opt=rpart_split_finder(OF,length(OF)))
(val_opt2=ctree_split_finder(OF,length(OF)))
####

plot(sort(OF),type="l",ylim=c(sum[1],sum[6]))

##alternativa
plot(sort(OF), col=is_out[order(OF)]+1)
abline(h=val_opt)
sort(val_opt)
##la soglia è il secondo valore piu alto
soglia=sort(val_opt)[2]
##taglio va effettuato 
out_previsti=rep(0,n)
out_previsti=ifelse(OF>soglia,1,0)
table(out_previsti)
(toptics=table(out_previsti,is_out))



#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
#auc(roc_OPTICS)

#Area under the curve: 0.7025
(precision=toptics[2,2]/sum(toptics[2,]))
#[1] 0.5

(recall=toptics[2,2]/sum(toptics[,2]))
#[1] 0.42








# CARDIO ------------------------------------------------------------------




rm(list=ls())
###devo cambiare distnze per togliere gli zeri
#devo inserire albero nel dbscan
#install.packages("pROC")
library(pROC)
library(rpart)
library(rpart.plot)
library(R.matlab)
library(dbscan)
library(partykit)

##gia scalato



###codice operativo


rm(list=ls())
###FUNZIONI
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




setwd("C:/Users/federico/Desktop/Progetto_sl")

path="cardio.mat"

data=readMat(path)
X=data$X
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))

p=dim(X)[2]; n=dim(X)[1]


### DBSCAN non Norm ---------------------------


#primo k, media tra i tre
k=mean(c(p+1,log(n),2*p))
#secondo k max tra i due 
#k2=max(p+1,log(n))
#3)
#k3=max(2*p,log(n))
a=kNNdist(X,k=k)
head(a)
kNNdistplot(X,k=k)
distanze=as.vector(a)
####inserire qui il metodo di taglio con "Alberelli"
head(distanze)

###SPLIT
(valori=rpart_split_finder(distanze,length(distanze)))
(valori2=ctree_split_finder(distanze,length(distanze)))

#


plot(sort(distanze),type="l")
abline(h=valori,col=valori+1)


plot(sort(distanze),type="l")
abline(h=valori2,col=valori+1)
###scelta di buonsenso: nessuno di questi
abline(h=3)

#3tre trois tarari
eps=3

res=dbscan(X,eps=eps,minPts = k)
indicatore=res$cluster==0
out_previsti=rep(0,n);out_previsti[indicatore]=1
sum(out_previsti)
##outlier reali
sum(is_out)
t3=table(is_out,out_previsti)
t3

hullplot(X,res)

#Confusion Matrix & AUC:
table(DBSCAN=out_previsti, Actual=is_out)
#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
(precision=t3[2,2]/sum(t3[2,]))
#0.6022727

auc(roc_dbscan)
#Area under the curve: 0.7573

(recall=t3[2,2]/sum(t3[,2]))
#[[1] 0.4223108


####OPTICS

###OF NON scalato --------------


res<-optics(X,eps=30,minPts = k)
a=kNN(X,k)
head(a$id,3)
numerid=a$id
# numeridi_esimo=[i,]
axx<-kNNdist(X,k)
head(res$order,n=15)
plot(res)

#altro modo per estra1lo
distanzerech=res$reachdist
head(distanzerech)
distanzerech[1]=mean(distanzerech)
#dist=mean(distanzerech[numerid[1,]])
dist=vector()
lrd=vector()

###modifica per evitare problemi dati uguali (dati ripetuti)
for (i in 1:n){
  minptsvicini=numerid[i,]
  dist[i]=mean(distanzerech[numerid[i,]])
  
  
}
distmagg=dist[dist>0]
valore=min(distmagg)
dist[dist==0]=valore/2
lrd=1/dist
#lrd[i]=1/dist[i]
summary(dist)
min(dist)
hist(dist)
dist
head(lrd,20)
numeratore=vector()
OF=vector()
for (i in 1:n){
  minptsvicini=numerid[i,]
  lrd_numero=lrd[i]
  lrd_minpts=lrd[numerid[i,]]
  numeratore[i]=sum(lrd_minpts/lrd_numero)
  OF[i]=numeratore[i]/k
}
summary(lrd)
lrd[max(lrd)]
sum(is.na(OF))
str(OF)
(sum=summary(OF))
str(sum)




###QUI INSERIRE ALBERELLI
(val_opt=rpart_split_finder(OF,length(OF)))
(val_opt2=ctree_split_finder(OF,length(OF)))
####

plot(sort(OF),type="l",ylim=c(sum[1],sum[6]))

##alternativa
plot(sort(OF), col=is_out[order(OF)]+1)
abline(h=val_opt)
sort(val_opt)
##la soglia di buonsenso è 1.15
soglia=1.15
##taglio va effettuato 
out_previsti=rep(0,n)
out_previsti=ifelse(OF>soglia,1,0)
table(out_previsti)
(toptics=table(out_previsti,is_out))



#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
#auc(roc_dbscan)

#Area under the curve: 5461
(precision=toptics[2,2]/sum(toptics[2,]))
#[1] 0.422

(recall=toptics[2,2]/sum(toptics[,2]))
#[1] 0.1079



# THYROID -----------------------------------------------------------------


##THYROID
library(R.matlab)
library(dbscan)
rm(list=ls())
setwd("C:/Users/federico/Desktop/Progetto_sl")


data=readMat("thyroid.mat")
X=data$X
Y=data$y
X=scale(X)

p=dim(X)[2]
n=dim(X)[1]
k=max(p+1,log(n))
kNNdistplot(X,k=max(p+1,log(n)))
kNNdistplot(X,k=2*p)

abline(h=0.7,col=2)
eps=0.6

res=dbscan(X,eps=eps,minPts = 2*p)
indicatore=res$cluster==0
out=rep(0,n)
out[indicatore]=1
sum(out)
##outlier reali
sum(Y)
t3=table(Y,out)
t3
t2
t1
t2=table(Y,out)
t2 #minPts=330
t1 #minPts=p+1
t1=table(Y,out)



####Optics
x1=c(rep(rnorm(mean=0,1,1),100),rep(rnorm(mean=10,sd=3,1),100),100,-100)
a=rnorm(mean=0,sd=1,202)
x=as.matrix(x1)
plot(x1,x2,xlim=c(-1,1))
res<-optics(X,eps=10,minPts = 2*p)
res1
head(res$order,n=15)
plot(res)
plot(X,col="dark grey",pch=3)

polygon(X[res$order,],)

#2 MODI PER ESTRARLI
res<-extractDBSCAN(res,eps_cl=.)
plot(res)
hullplot(X,res)
sum(res$cluster==0)
#altro modo per estrarlo
re=vector()
a=seq(0.001,0.05,by=0.0005)
a[100]=0.06
for (i in 1:100){
  
  re[i]<-sum(extractXi(res,xi=a[i])$cluster==0)
  
}

res=extractXi(res,xi=0.000001)
sum(res$cluster==0)
table(res$cluster)






##COMPLETO












##GIA SCALATO



#####Completo --------


ann_train <- read.table("C:/Users/federico/Desktop/Progetto_sl/ann_train.csv", quote="\"", comment.char="")
dati=ann_train
summary(dati)
y=dati[,22]
is_out=ifelse(y>1,0,1)
X=dati[,-22]
dati<-data.frame(cbind(X, is_out))

p=dim(X)[2]; n=dim(X)[1]
summary(X)#one hot encoding

fac<-c(2:16)
for (i in 1:length(fac)){
  dati[,i]<-as.factor(dati[,i])
}
#One hot encoding:
dati<-model.matrix(~.-1,dati[,-n])
X=dati[,1:(ncol(dati)-1)]
p=dim(X)[2]; n=dim(X)[1]



#####RICOMINCIAMO DA CAPO --------
#dbscan 

X=scale(X)


#primo k, media tra i tre
k=mean(c(p+1,log(n),2*p))
k=100
#secondo k max tra i due 
#k2=max(p+1,log(n))
#3)
#k3=max(2*p,log(n))
a=kNNdist(X,k=k)
head(a)
kNNdistplot(X,k=k)
distanze=as.vector(a)
####inserire qui il metodo di taglio con "Alberelli"
head(distanze)


(valori=rpart_split_finder(distanze,length(distanze)))
#
plot(sort(distanze),type="l")
#

abline(h=valori,col=valori+1)
abline(h=1)
###scelta di buonsenso: il prima del massimo

valori
eps=sort(valori)[7]
res=dbscan(X,eps=eps,minPts = k)
indicatore=res$cluster==0
out_previsti=rep(0,n);out_previsti[indicatore]=1
sum(out_previsti)
##outlier reali
sum()
t3=table(is_out,out_previsti)
t3

hullplot(X,res)



#Confusion Matrix & AUC:
table(DBSCAN=out_previsti, Actual=is_out)
#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
(precision=t3[2,2]/sum(t3[2,]))
#[1] 0.03846154

# auc(roc_dbscan)
#Area under the curve: 0.5059

(recall=t3[2,2]/sum(t3[,2]))
#[1] 0.03225806




###optics completo ---------
###OF NON scalato --------------


res<-optics(X,eps=100,minPts = k)
a=kNN(X,k)
head(a$id,3)
numerid=a$id
# numeridi_esimo=[i,]
axx<-kNNdist(X,k)
head(res$order,n=15)
plot(res)

#altro modo per estra1lo
distanzerech=res$reachdist
head(distanzerech)
distanzerech[1]=distanzerech[2]
#dist=mean(distanzerech[numerid[1,]])
dist=vector()
lrd=vector()

###modifica per evitare problemi dati uguali (dati ripetuti)
for (i in 1:n){
  minptsvicini=numerid[i,]
  dist[i]=mean(distanzerech[numerid[i,]])
  
  
}
distmagg=dist[dist>0]
valore=min(distmagg)
dist[dist==0]=valore/2
lrd=1/dist
#lrd[i]=1/dist[i]
summary(dist)
min(dist)
hist(dist)
head(lrd,20)
numeratore=vector()
OF=vector()
for (i in 1:n){
  minptsvicini=numerid[i,]
  lrd_numero=lrd[i]
  lrd_minpts=lrd[numerid[i,]]
  numeratore[i]=sum(lrd_minpts/lrd_numero)
  OF[i]=numeratore[i]/k
}
summary(lrd)
lrd[max(lrd)]
sum(is.na(OF))
str(OF)
(sum=summary(OF))
str(sum)




###QUI INSERIRE ALBERELLI
(val_opt=rpart_split_finder(OF,length(OF)))
(val_opt2=ctree_split_finder(OF,length(OF)))
####

#OF RISCALATO
plot(sort(OF),type="l",ylim=c(sum[1],sum[6]))

##alternativa
plot(sort(OF), col=is_out[order(OF)]+1)
abline(h=val_opt)
abline(h=val_opt2)
sort(val_opt)
##la soglia è il valore piu alto
soglia=max(val_opt)
##taglio va effettuato 
out_previsti=rep(0,n)
out_previsti=ifelse(OF>soglia,1,0)
table(out_previsti)
(toptics=table(out_previsti,is_out))



#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
#auc(roc_dbscan)
#Area under the curve: 0.5057



(precision=toptics[2,2]/sum(toptics[2,]))
#[1] 0.03333333

(recall=toptics[2,2]/sum(toptics[,2]))
#[1] 0.04301075














# MUSK --------------------------------------------------------------------





###devo cambiare distnze per togliere gli zeri
#devo inserire albero nel dbscan
#install.packages("pROC")
library(pROC)
library(rpart)
library(rpart.plot)
library(R.matlab)
library(dbscan)
library(partykit)





###codice operativo


rm(list=ls())
###FUNZIONI
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




setwd("C:/Users/federico/Desktop/Progetto_sl")

path="musk.mat"

data=readMat(path)
X=data$X
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))

p=dim(X)[2]; n=dim(X)[1]


### DBSCAN non Norm ---------------------------


#primo k, media tra i tre
k=mean(c(p+1,log(n),2*p))
#secondo k max tra i due 
#k2=max(p+1,log(n))
#3)
#k3=max(2*p,log(n))
a=kNNdist(X,k=k)
head(a)
kNNdistplot(X,k=k)
distanze=as.vector(a)
####inserire qui il metodo di taglio con "Alberelli"
head(distanze)


(valori=rpart_split_finder(distanze,length(distanze)))

#
plot(sort(distanze),type="l")
#

abline(h=valori,col=valori+1)
abline(h=0.7)
###scelta di buonsenso, il max

eps=max(valori)
res=dbscan(X,eps=eps,minPts = k)
indicatore=res$cluster==0
out_previsti=rep(0,n);out_previsti[indicatore]=1
sum(out_previsti)
##outlier reali
sum(is_out)
t3=table(is_out,out_previsti)
t3

hullplot(X,res)



#Confusion Matrix & AUC:
table(DBSCAN=out_previsti, Actual=is_out)
#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
(precision=t3[2,2]/sum(t3[2,]))
#[1] 1

# auc(roc_dbscan)
#Area under the curve: 0.8136

(recall=t3[2,2]/sum(t3[,2]))
#[1] 1




#####DBSCAN scalato ------------
X=scale(X)
a=kNNdist(X,k=k)
head(a)
kNNdistplot(X,k=k)
distanze=as.vector(a)
####inserire qui il metodo di taglio con "Alberelli"
head(distanze)


(valori=rpart_split_finder(distanze,length(distanze)))

#
plot(sort(distanze),type="l")
#

abline(h=valori,col=valori+1)
###scelta di buonsenso, il max

eps=max(valori)
res=dbscan(X,eps=eps,minPts = k)
indicatore=res$cluster==0
out_previsti=rep(0,n);out_previsti[indicatore]=1
sum(out_previsti)
##outlier reali
sum(is_out)
t3=table(is_out,out_previsti)
t3

hullplot(X,res)



#Confusion Matrix & AUC:
table(DBSCAN=out_previsti, Actual=is_out)
#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
(precision=t3[2,2]/sum(t3[2,]))
#[1] 1

# auc(roc_dbscan)
#Area under the curve: 1

(recall=t3[2,2]/sum(t3[,2]))
#1




###OF NON scalato --------------
##--> impossibile

###OF SCALATO ------------------
X=scale(X)
res<-optics(X,eps=30,minPts = k)
a=kNN(X,k)
head(a$id,3)
numerid=a$id
# numeridi_esimo=[i,]
axx<-kNNdist(X,k)
head(res$order,n=15)
plot(res)

#altro modo per estra1lo
distanzerech=res$reachdist
head(distanzerech)
distanzerech[1]=distanzerech[2]
#dist=mean(distanzerech[numerid[1,]])
dist=vector()
lrd=vector()

###modifica per evitare problemi dati uguali (dati ripetuti)
for (i in 1:n){
  minptsvicini=numerid[i,]
  dist[i]=mean(distanzerech[numerid[i,]])
  
  
}
distmagg=dist[dist>0]
valore=min(distmagg)
dist[dist==0]=valore/2
lrd=1/dist
#lrd[i]=1/dist[i]
summary(dist)
min(dist)
hist(dist)
dist
head(lrd,20)
numeratore=vector()
OF=vector()
for (i in 1:n){
  minptsvicini=numerid[i,]
  lrd_numero=lrd[i]
  lrd_minpts=lrd[numerid[i,]]
  numeratore[i]=sum(lrd_minpts/lrd_numero)
  OF[i]=numeratore[i]/k
}
summary(lrd)
lrd[max(lrd)]
sum(is.na(OF))
str(OF)
(sum=summary(OF))
str(sum)




###QUI INSERIRE ALBERELLI
(val_opt=rpart_split_finder(OF,length(OF)))
(val_opt2=ctree_split_finder(OF,length(OF)))
####

#OF RISCALATO
plot(sort(OF),type="l",ylim=c(sum[1],sum[6]))

##alternativa
plot(sort(OF), col=is_out[order(OF)]+1)
abline(h=val_opt)
sort(val_opt)
##la soglia è il secondo valore piu alto
soglia=sort(val_opt)[6]
##taglio va effettuato 
out_previsti=rep(0,n)
out_previsti=ifelse(OF>soglia,1,0)
table(out_previsti)
(toptics=table(out_previsti,is_out))



#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
#auc(roc_dbscan)
#Area under the curve: 1



(precision=toptics[2,2]/sum(toptics[2,]))
#[1] 1

(recall=toptics[2,2]/sum(toptics[,2]))
#[1] 1




# SATIMAGE ----------------------------------------------------------------




###SATIMAGE

rm(list=ls())
###devo cambiare distnze per togliere gli zeri
#devo inserire albero nel dbscan
#install.packages("pROC")
library(pROC)
library(rpart)
library(rpart.plot)
library(R.matlab)
library(dbscan)
library(partykit)





###codice operativo


rm(list=ls())
###FUNZIONI
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




setwd("C:/Users/federico/Desktop/Progetto_sl")

path="satimage-2.mat"

data=readMat(path)
X=data$X
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))

p=dim(X)[2]; n=dim(X)[1]


### DBSCAN non Norm ---------------------------


#primo k, media tra i tre
k=mean(c(p+1,log(n),2*p))
#secondo k max tra i due 
#k2=max(p+1,log(n))
#3)
#k3=max(2*p,log(n))
a=kNNdist(X,k=k)
head(a)
kNNdistplot(X,k=k)
distanze=as.vector(a)
####inserire qui il metodo di taglio con "Alberelli"
head(distanze)


(valori=rpart_split_finder(distanze,length(distanze)))
(valori2=ctree_split_finder(distanze,length(distanze)))
#
plot(sort(distanze),type="l")
#

abline(h=valori,col=valori+1)

###scelta di buonsenso, il max

eps=max(valori)
res=dbscan(X,eps=eps,minPts = k)
indicatore=res$cluster==0
out_previsti=rep(0,n);out_previsti[indicatore]=1
sum(out_previsti)
##outlier reali
sum(is_out)
t3=table(is_out,out_previsti)
t3

hullplot(X,res)



#Confusion Matrix & AUC:
table(DBSCAN=out_previsti, Actual=is_out)
#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
(precision=t3[2,2]/sum(t3[2,]))
#[1] 0.971831

# auc(roc_dbscan)
#Area under the curve: 0.9803

(recall=t3[2,2]/sum(t3[,2]))
#[1] 0.518797




#####DBSCAN scalato ------------
X=scale(X)
a=kNNdist(X,k=k)
head(a)
kNNdistplot(X,k=k)
distanze=as.vector(a)
####inserire qui il metodo di taglio con "Alberelli"
head(distanze)


(valori=rpart_split_finder(distanze,length(distanze)))
(valori2=ctree_split_finder(distanze,length(distanze)) )
#
plot(sort(distanze),type="l")
#

abline(h=valori,col=valori+1)
abline(h=valori2)
###scelta di buonsenso, inessuno di questi, 3.6
abline(h=3.2)

eps=3.6
res=dbscan(X,eps=eps,minPts = k)
indicatore=res$cluster==0
out_previsti=rep(0,n);out_previsti[indicatore]=1
sum(out_previsti)
##outlier reali
sum(is_out)
t3=table(is_out,out_previsti)
t3

hullplot(X,res)



#Confusion Matrix & AUC:
table(DBSCAN=out_previsti, Actual=is_out)
#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
(precision=t3[2,2]/sum(t3[2,]))
#[1] Area under the curve: 0.9818
#[1] 0.971831


# auc(roc_dbscan)
#Area under the curve: 1

(recall=t3[2,2]/sum(t3[,2]))
#0.5948276




###OF NON scalato --------------
X=data$X


res<-optics(X,eps=200,minPts = k)
a=kNN(X,k)
head(a$id,3)
numerid=a$id
# numeridi_esimo=[i,]
axx<-kNNdist(X,k)
head(res$order,n=15)
plot(res)

#altro modo per estra1lo
distanzerech=res$reachdist
head(distanzerech)
distanzerech[1]=distanzerech[2]
#dist=mean(distanzerech[numerid[1,]])
dist=vector()
lrd=vector()

###modifica per evitare problemi dati uguali (dati ripetuti)
for (i in 1:n){
  minptsvicini=numerid[i,]
  dist[i]=mean(distanzerech[numerid[i,]])
  
  
}
distmagg=dist[dist>0]
valore=min(distmagg)
dist[dist==0]=valore/2
lrd=1/dist
#lrd[i]=1/dist[i]
summary(dist)
min(dist)
hist(dist)
dist
head(lrd,20)
numeratore=vector()
OF=vector()
for (i in 1:n){
  minptsvicini=numerid[i,]
  lrd_numero=lrd[i]
  lrd_minpts=lrd[numerid[i,]]
  numeratore[i]=sum(lrd_minpts/lrd_numero)
  OF[i]=numeratore[i]/k
}
summary(lrd)
lrd[max(lrd)]
sum(is.na(OF))
str(OF)
(sum=summary(OF))
str(sum)




###QUI INSERIRE ALBERELLI
(val_opt=rpart_split_finder(OF,length(OF)))
(val_opt2=ctree_split_finder(OF,length(OF)))
####

#OF RISCALATO
plot(sort(OF),type="l",ylim=c(sum[1],sum[6]))

##alternativa
plot(sort(OF), col=is_out[order(OF)]+1)
abline(h=val_opt)
sort(val_opt)

##la soglia non  nessuno di questi 
soglia=1.3
##taglio va effettuato 
out_previsti=rep(0,n)
out_previsti=ifelse(OF>soglia,1,0)
table(out_previsti)
(toptics=table(out_previsti,is_out))



#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
#auc(roc_dbscan)
#Area under the curve 0.5395



(precision=toptics[2,2]/sum(toptics[2,]))
#[1] 0.157

(recall=toptics[2,2]/sum(toptics[,2]))
#[1] 0.0845





###OF SCALATO ------------------
X=scale(X)
res<-optics(X,eps=30,minPts = k)
a=kNN(X,k)
head(a$id,3)
numerid=a$id
# numeridi_esimo=[i,]
axx<-kNNdist(X,k)
head(res$order,n=15)
plot(res)

#altro modo per estra1lo
distanzerech=res$reachdist
head(distanzerech)
distanzerech[1]=distanzerech[2]
#dist=mean(distanzerech[numerid[1,]])
dist=vector()
lrd=vector()

###modifica per evitare problemi dati uguali (dati ripetuti)
for (i in 1:n){
  minptsvicini=numerid[i,]
  dist[i]=mean(distanzerech[numerid[i,]])
  
  
}
distmagg=dist[dist>0]
valore=min(distmagg)
dist[dist==0]=valore/2
lrd=1/dist
#lrd[i]=1/dist[i]
summary(dist)
min(dist)
hist(dist)
dist
head(lrd,20)
numeratore=vector()
OF=vector()
for (i in 1:n){
  minptsvicini=numerid[i,]
  lrd_numero=lrd[i]
  lrd_minpts=lrd[numerid[i,]]
  numeratore[i]=sum(lrd_minpts/lrd_numero)
  OF[i]=numeratore[i]/k
}
summary(lrd)
lrd[max(lrd)]
sum(is.na(OF))
str(OF)
(sum=summary(OF))
str(sum)




###QUI INSERIRE ALBERELLI
(val_opt=rpart_split_finder(OF,length(OF)))
(val_opt2=ctree_split_finder(OF,length(OF)))
####

#OF RISCALATO
plot(sort(OF),type="l",ylim=c(sum[1],sum[6]))

##alternativa
plot(sort(OF), col=is_out[order(OF)]+1)
abline(h=val_opt)
sort(val_opt)
##la soglia è il  valore piu alto
soglia=max(val_opt)
##taglio va effettuato 
out_previsti=rep(0,n)
out_previsti=ifelse(OF>soglia,1,0)
table(out_previsti)
(toptics=table(out_previsti,is_out))



#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
#auc(roc_dbscan)
#Area under the curve: Area under the curve: 0.5555



(precision=toptics[2,2]/sum(toptics[2,]))
#[1] 0.1066667

(recall=toptics[2,2]/sum(toptics[,2]))
#[1] 0.1126761





# LETTER RECOGNIZION ------------------------------------------------------



##LETTER RECONGN

rm(list=ls())
###devo cambiare distnze per togliere gli zeri
#devo inserire albero nel dbscan
#install.packages("pROC")
library(pROC)
library(rpart)
library(rpart.plot)
library(R.matlab)
library(dbscan)
library(partykit)





###codice operativo


rm(list=ls())
###FUNZIONI
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




setwd("C:/Users/federico/Desktop/Progetto_sl")

path="letter.mat"

data=readMat(path)
X=data$X
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))

p=dim(X)[2]; n=dim(X)[1]


### DBSCAN non Norm ---------------------------


#primo k, media tra i tre
k=mean(c(p+1,log(n),2*p))
#secondo k max tra i due 
#k2=max(p+1,log(n))
#3)
#k3=max(2*p,log(n))
a=kNNdist(X,k=k)
head(a)
kNNdistplot(X,k=k)
distanze=as.vector(a)
####inserire qui il metodo di taglio con "Alberelli"
head(distanze)


(valori=rpart_split_finder(distanze,length(distanze)))
(valori2=ctree_split_finder(distanze,length(distanze)))
#
plot(sort(distanze),type="l")
#

abline(h=valori,col=valori+1)
abline(h=valori2,col=valori2+1)
###scelta di buonsenso, il max
abline(h=13.3)
eps=max(valori)
res=dbscan(X,eps=eps,minPts = k)
indicatore=res$cluster==0
out_previsti=rep(0,n);out_previsti[indicatore]=1
sum(out_previsti)
##outlier reali
sum(is_out)
t3=table(is_out,out_previsti)
t3

hullplot(X,res)



#Confusion Matrix & AUC:
table(DBSCAN=out_previsti, Actual=is_out)
#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
(precision=t3[2,2]/sum(t3[2,]))
#[1] 0.25

# auc(roc_dbscan)
#Area under the curve: 0.604

(recall=t3[2,2]/sum(t3[,2]))
#[1] 0.2840909




#####DBSCAN scalato ------------
X=scale(X)
a=kNNdist(X,k=k)
head(a)
kNNdistplot(X,k=k)
distanze=as.vector(a)
####inserire qui il metodo di taglio con "Alberelli"
head(distanze)


(valori=rpart_split_finder(distanze,length(distanze)))
(valori2=ctree_split_finder(distanze,length(distanze)) )
#
plot(sort(distanze),type="l")
#

abline(h=valori,col=valori+1)
abline(h=valori2)
###scelta di buonsenso, il max
abline(h=3.2)

eps=max(valori)
res=dbscan(X,eps=eps,minPts = k)
indicatore=res$cluster==0
out_previsti=rep(0,n);out_previsti[indicatore]=1
sum(out_previsti)
##outlier reali
sum(is_out)
t3=table(is_out,out_previsti)
t3

hullplot(X,res)



#Confusion Matrix & AUC:
table(DBSCAN=out_previsti, Actual=is_out)
#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
(precision=t3[2,2]/sum(t3[2,]))
#[1] Area under the curve: 0.61
#[1] 0.28


# auc(roc_dbscan)
#Area under the curve: 1

(recall=t3[2,2]/sum(t3[,2]))
#0.2372881




###OF NON scalato --------------
X=data$X


res<-optics(X,eps=200,minPts = k)
a=kNN(X,k)
head(a$id,3)
numerid=a$id
# numeridi_esimo=[i,]
axx<-kNNdist(X,k)
head(res$order,n=15)
plot(res)

#altro modo per estra1lo
distanzerech=res$reachdist
head(distanzerech)
distanzerech[1]=distanzerech[2]
#dist=mean(distanzerech[numerid[1,]])
dist=vector()
lrd=vector()

###modifica per evitare problemi dati uguali (dati ripetuti)
for (i in 1:n){
  minptsvicini=numerid[i,]
  dist[i]=mean(distanzerech[numerid[i,]])
  
  
}
distmagg=dist[dist>0]
valore=min(distmagg)
dist[dist==0]=valore/2
lrd=1/dist
#lrd[i]=1/dist[i]
summary(dist)
min(dist)
hist(dist)
dist
head(lrd,20)
numeratore=vector()
OF=vector()
for (i in 1:n){
  minptsvicini=numerid[i,]
  lrd_numero=lrd[i]
  lrd_minpts=lrd[numerid[i,]]
  numeratore[i]=sum(lrd_minpts/lrd_numero)
  OF[i]=numeratore[i]/k
}
summary(lrd)
lrd[max(lrd)]
sum(is.na(OF))
str(OF)
(sum=summary(OF))
str(sum)




###QUI INSERIRE ALBERELLI
(val_opt=rpart_split_finder(OF,length(OF)))
(val_opt2=ctree_split_finder(OF,length(OF)))
####

#OF RISCALATO
plot(sort(OF),type="l",ylim=c(sum[1],sum[6]))

##alternativa
plot(sort(OF), col=is_out[order(OF)]+1)
abline(h=val_opt)
sort(val_opt)

##la soglia è il secondo piu alto  
soglia=sort(val_opt)[3]
##taglio va effettuato 
out_previsti=rep(0,n)
out_previsti=ifelse(OF>soglia,1,0)
table(out_previsti)
(toptics=table(out_previsti,is_out))



#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
#auc(roc_dbscan
#Area under the curve 0.6083



(precision=toptics[2,2]/sum(toptics[2,]))
#[1] 0.1423

(recall=toptics[2,2]/sum(toptics[,2]))
#[1] 0.36





###OF SCALATO ------------------
X=scale(X)
res<-optics(X,eps=30,minPts = k)
a=kNN(X,k)
head(a$id,3)
numerid=a$id
# numeridi_esimo=[i,]
axx<-kNNdist(X,k)
head(res$order,n=15)
plot(res)

#altro modo per estra1lo
distanzerech=res$reachdist
head(distanzerech)
distanzerech[1]=distanzerech[2]
#dist=mean(distanzerech[numerid[1,]])
dist=vector()
lrd=vector()

###modifica per evitare problemi dati uguali (dati ripetuti)
for (i in 1:n){
  minptsvicini=numerid[i,]
  dist[i]=mean(distanzerech[numerid[i,]])
  
  
}
distmagg=dist[dist>0]
valore=min(distmagg)
dist[dist==0]=valore/2
lrd=1/dist
#lrd[i]=1/dist[i]
summary(dist)
min(dist)
hist(dist)
dist
head(lrd,20)
numeratore=vector()
OF=vector()
for (i in 1:n){
  minptsvicini=numerid[i,]
  lrd_numero=lrd[i]
  lrd_minpts=lrd[numerid[i,]]
  numeratore[i]=sum(lrd_minpts/lrd_numero)
  OF[i]=numeratore[i]/k
}
summary(lrd)
lrd[max(lrd)]
sum(is.na(OF))
str(OF)
(sum=summary(OF))
str(sum)




###QUI INSERIRE ALBERELLI
(val_opt=rpart_split_finder(OF,length(OF)))
(val_opt2=ctree_split_finder(OF,length(OF)))
####

#OF RISCALATO
plot(sort(OF),type="l",ylim=c(sum[1],sum[6]))

##alternativa
plot(sort(OF), col=is_out[order(OF)]+1)
abline(h=val_opt)
sort(val_opt)
##la soglia è nessuno di questi, è 1.025
soglia=1.025
##taglio va effettuato 
out_previsti=rep(0,n)
out_previsti=ifelse(OF>soglia,1,0)
table(out_previsti)
(toptics=table(out_previsti,is_out))



#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
#auc(roc_dbscan)
#Area under the curve: Area under the curve: 0.62



(precision=toptics[2,2]/sum(toptics[2,]))
#[1] 0.1346154

(recall=toptics[2,2]/sum(toptics[,2]))
#[1] 0.42




# SPEECH ------------------------------------------------------------------

##speech

rm(list=ls())
###devo cambiare distnze per togliere gli zeri
#devo inserire albero nel dbscan
#install.packages("pROC")
library(pROC)
library(rpart)
library(rpart.plot)
library(R.matlab)
library(dbscan)
library(partykit)





###codice operativo


rm(list=ls())
###FUNZIONI
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




setwd("C:/Users/federico/Desktop/Progetto_sl")

path="speech.mat"

data=readMat(path)
X=data$X
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))

p=dim(X)[2]; n=dim(X)[1]


### DBSCAN non Norm ---------------------------


#primo k, media tra i tre
k=mean(c(p+1,log(n),2*p))
#secondo k max tra i due 
#k2=max(p+1,log(n))
#3)
#k3=max(2*p,log(n))
a=kNNdist(X,k=k)
head(a)
kNNdistplot(X,k=k)
distanze=as.vector(a)
####inserire qui il metodo di taglio con "Alberelli"
head(distanze)


(valori=rpart_split_finder(distanze,length(distanze)))
(valori2=ctree_split_finder(distanze,length(distanze)))
#
plot(sort(distanze),type="l")
#

abline(h=valori,col=valori+1)
abline(h=valori2,col=valori2+1)
###scelta di buonsenso, il max
abline(h=24)
eps=23.8
res=dbscan(X,eps=eps,minPts = k)
indicatore=res$cluster==0
out_previsti=rep(0,n);out_previsti[indicatore]=1
sum(out_previsti)
##outlier reali
sum(is_out)
t3=table(is_out,out_previsti)
t3

hullplot(X,res)



#Confusion Matrix & AUC:
table(DBSCAN=out_previsti, Actual=is_out)
#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
(precision=t3[2,2]/sum(t3[2,]))
#[1] 0.04918033

# auc(roc_dbscan)
#Area under the curve: 0.5198

(recall=t3[2,2]/sum(t3[,2]))
#[1] 0.07894737




#####DBSCAN scalato ------------
X=scale(X)
a=kNNdist(X,k=k)
head(a)
kNNdistplot(X,k=k)
distanze=as.vector(a)
####inserire qui il metodo di taglio con "Alberelli"
head(distanze)


(valori=rpart_split_finder(distanze,length(distanze)))
(valori2=ctree_split_finder(distanze,length(distanze)) )
#
plot(sort(distanze),type="l")
#

abline(h=valori,col=valori+1)
abline(h=valori2)
###scelta di buonsenso, il max


eps=max(valori)
res=dbscan(X,eps=eps,minPts = k)
indicatore=res$cluster==0
out_previsti=rep(0,n);out_previsti[indicatore]=1
sum(out_previsti)
##outlier reali
sum(is_out)
t3=table(is_out,out_previsti)
t3

hullplot(X,res)



#Confusion Matrix & AUC:
table(DBSCAN=out_previsti, Actual=is_out)
#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
(precision=t3[2,2]/sum(t3[2,]))
#[1] Area under the curve: 0.4982
#[1] 0


# auc(roc_dbscan)

(recall=t3[2,2]/sum(t3[,2]))
#0



###OF NON scalato --------------
X=data$X


res<-optics(X,eps=200,minPts = k)
a=kNN(X,k)
head(a$id,3)
numerid=a$id
# numeridi_esimo=[i,]
axx<-kNNdist(X,k)
head(res$order,n=15)
plot(res)

#altro modo per estra1lo
distanzerech=res$reachdist
head(distanzerech)
distanzerech[1]=distanzerech[2]
#dist=mean(distanzerech[numerid[1,]])
dist=vector()
lrd=vector()

###modifica per evitare problemi dati uguali (dati ripetuti)
for (i in 1:n){
  minptsvicini=numerid[i,]
  dist[i]=mean(distanzerech[numerid[i,]])
  
  
}
distmagg=dist[dist>0]
valore=min(distmagg)
dist[dist==0]=valore/2
lrd=1/dist
#lrd[i]=1/dist[i]
summary(dist)
min(dist)
hist(dist)
dist
head(lrd,20)
numeratore=vector()
OF=vector()
for (i in 1:n){
  minptsvicini=numerid[i,]
  lrd_numero=lrd[i]
  lrd_minpts=lrd[numerid[i,]]
  numeratore[i]=sum(lrd_minpts/lrd_numero)
  OF[i]=numeratore[i]/k
}
summary(lrd)
lrd[max(lrd)]
sum(is.na(OF))
str(OF)
(sum=summary(OF))
str(sum)




###QUI INSERIRE ALBERELLI
(val_opt=rpart_split_finder(OF,length(OF)))
(val_opt2=ctree_split_finder(OF,length(OF)))
####

#OF RISCALATO
plot(sort(OF),type="l",ylim=c(sum[1],sum[6]))

##alternativa
plot(sort(OF), col=is_out[order(OF)]+1)
abline(h=val_opt)
sort(val_opt)

##la soglia è il secondo piu alto  
soglia=1.0055
##taglio va effettuato 
out_previsti=rep(0,n)
out_previsti=ifelse(OF>soglia,1,0)
table(out_previsti)
(toptics=table(out_previsti,is_out))



#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
#auc(roc_dbscan
#Area under the curve 0.4966



(precision=toptics[2,2]/sum(toptics[2,]))
#[1] 0.01176471

(recall=toptics[2,2]/sum(toptics[,2]))
#[1] 0.01639344





###OF SCALATO ------------------
X=scale(X)
res<-optics(X,eps=30,minPts = k)
a=kNN(X,k)
head(a$id,3)
numerid=a$id
# numeridi_esimo=[i,]
axx<-kNNdist(X,k)
head(res$order,n=15)
plot(res)

#altro modo per estra1lo
distanzerech=res$reachdist
head(distanzerech)
distanzerech[1]=distanzerech[2]
#dist=mean(distanzerech[numerid[1,]])
dist=vector()
lrd=vector()

###modifica per evitare problemi dati uguali (dati ripetuti)
for (i in 1:n){
  minptsvicini=numerid[i,]
  dist[i]=mean(distanzerech[numerid[i,]])
  
  
}
distmagg=dist[dist>0]
valore=min(distmagg)
dist[dist==0]=valore/2
lrd=1/dist
#lrd[i]=1/dist[i]
summary(dist)
min(dist)
hist(dist)
dist
head(lrd,20)
numeratore=vector()
OF=vector()
for (i in 1:n){
  minptsvicini=numerid[i,]
  lrd_numero=lrd[i]
  lrd_minpts=lrd[numerid[i,]]
  numeratore[i]=sum(lrd_minpts/lrd_numero)
  OF[i]=numeratore[i]/k
}
summary(lrd)
lrd[max(lrd)]
sum(is.na(OF))
str(OF)
(sum=summary(OF))
str(sum)




###QUI INSERIRE ALBERELLI
(val_opt=rpart_split_finder(OF,length(OF)))
(val_opt2=ctree_split_finder(OF,length(OF)))
####

#OF RISCALATO
plot(sort(OF),type="l",ylim=c(sum[1],sum[6]))

##alternativa
plot(sort(OF), col=is_out[order(OF)]+1)
abline(h=val_opt)
abline(h=val_opt2)
sort(val_opt)
##la soglia è max opt 2
soglia=max(val_opt2)
##taglio va effettuato 
out_previsti=rep(0,n)
out_previsti=ifelse(OF>soglia,1,0)
table(out_previsti)
(toptics=table(out_previsti,is_out))



#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
#auc(roc_dbscan)
#Area under the curve: Area under the curve: 0.5142



(precision=toptics[2,2]/sum(toptics[2,]))
#[1] 0.01957295

(recall=toptics[2,2]/sum(toptics[,2]))
#[1] 0.1803279






# PIMA --------------------------------------------------------------------



##PIMA

rm(list=ls())
###devo cambiare distnze per togliere gli zeri
#devo inserire albero nel dbscan
#install.packages("pROC")
library(pROC)
library(rpart)
library(rpart.plot)
library(R.matlab)
library(dbscan)
library(partykit)





###codice operativo


rm(list=ls())
###FUNZIONI
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




setwd("C:/Users/federico/Desktop/Progetto_sl")

path="pima.mat"

data=readMat(path)
X=data$X
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))

p=dim(X)[2]; n=dim(X)[1]


### DBSCAN non Norm ---------------------------


#primo k, media tra i tre
k=mean(c(p+1,log(n),2*p))
#secondo k max tra i due 
#k2=max(p+1,log(n))
#3)
#k3=max(2*p,log(n))
a=kNNdist(X,k=k)
head(a)
kNNdistplot(X,k=k)
distanze=as.vector(a)
####inserire qui il metodo di taglio con "Alberelli"
head(distanze)


(valori=rpart_split_finder(distanze,length(distanze)))
(valori2=ctree_split_finder(distanze,length(distanze)))
#
plot(sort(distanze),type="l")
#

abline(h=valori,col=valori+1)
abline(h=valori2,col=valori2+1)
###scelta di buonsenso, secondo piu alto
abline(h=24)
eps=sort(valori)[7]
res=dbscan(X,eps=eps,minPts = k)
indicatore=res$cluster==0
out_previsti=rep(0,n);out_previsti[indicatore]=1
sum(out_previsti)
##outlier reali
sum(is_out)
t3=table(is_out,out_previsti)
t3

hullplot(X,res)



#Confusion Matrix & AUC:
table(DBSCAN=out_previsti, Actual=is_out)
#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
(precision=t3[2,2]/sum(t3[2,]))
#[1] 0.5410448

# auc(roc_dbscan)
#Area under the curve: 0.5925

(recall=t3[2,2]/sum(t3[,2]))
#[1] 0.4489164




#####DBSCAN scalato ------------
X=scale(X)
a=kNNdist(X,k=k)
head(a)
kNNdistplot(X,k=k)
distanze=as.vector(a)
####inserire qui il metodo di taglio con "Alberelli"
head(distanze)


(valori=rpart_split_finder(distanze,length(distanze)))
(valori2=ctree_split_finder(distanze,length(distanze)) )
#
plot(sort(distanze),type="l")
#

abline(h=valori,col=valori+1)
abline(h=valori2)
###scelta di buonsenso, il secondo piualto


eps=sort(valori)[7]
res=dbscan(X,eps=eps,minPts = k)
indicatore=res$cluster==0
out_previsti=rep(0,n);out_previsti[indicatore]=1
sum(out_previsti)
##outlier reali
sum(is_out)
t3=table(is_out,out_previsti)
t3

hullplot(X,res)



#Confusion Matrix & AUC:
table(DBSCAN=out_previsti, Actual=is_out)
#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
(precision=t3[2,2]/sum(t3[2,]))
#[1] Area under the curve: 0.5533
#[1] 0.1865672


# auc(roc_dbscan)

(recall=t3[2,2]/sum(t3[,2]))
#0.5555556



###OF NON scalato --------------
X=data$X


res<-optics(X,eps=200,minPts = k)
a=kNN(X,k)
head(a$id,3)
numerid=a$id
# numeridi_esimo=[i,]
axx<-kNNdist(X,k)
head(res$order,n=15)
plot(res)

#altro modo per estra1lo
distanzerech=res$reachdist
head(distanzerech)
distanzerech[1]=distanzerech[2]
#dist=mean(distanzerech[numerid[1,]])
dist=vector()
lrd=vector()

###modifica per evitare problemi dati uguali (dati ripetuti)
for (i in 1:n){
  minptsvicini=numerid[i,]
  dist[i]=mean(distanzerech[numerid[i,]])
  
  
}
distmagg=dist[dist>0]
valore=min(distmagg)
dist[dist==0]=valore/2
lrd=1/dist
#lrd[i]=1/dist[i]
summary(dist)
min(dist)
hist(dist)
dist
head(lrd,20)
numeratore=vector()
OF=vector()
for (i in 1:n){
  minptsvicini=numerid[i,]
  lrd_numero=lrd[i]
  lrd_minpts=lrd[numerid[i,]]
  numeratore[i]=sum(lrd_minpts/lrd_numero)
  OF[i]=numeratore[i]/k
}
summary(lrd)
lrd[max(lrd)]
sum(is.na(OF))
str(OF)
(sum=summary(OF))
str(sum)




###QUI INSERIRE ALBERELLI
(val_opt=rpart_split_finder(OF,length(OF)))
(val_opt2=ctree_split_finder(OF,length(OF)))
####

#OF RISCALATO
plot(sort(OF),type="l",ylim=c(sum[1],sum[6]))

##alternativa
plot(sort(OF), col=is_out[order(OF)]+1)
abline(h=val_opt)
sort(val_opt)

##la soglia è il minore
soglia=1
##taglio va effettuato 
out_previsti=rep(0,n)
out_previsti=ifelse(OF>soglia,1,0)
table(out_previsti)
(toptics=table(out_previsti,is_out))



#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
#auc(roc_dbscan
#Area under the curve 0.507



(precision=toptics[2,2]/sum(toptics[2,]))
#[1] 0.3621622

(recall=toptics[2,2]/sum(toptics[,2]))
#[1] 0.25





###OF SCALATO ------------------
X=scale(X)
res<-optics(X,eps=30,minPts = k)
a=kNN(X,k)
head(a$id,3)
numerid=a$id
# numeridi_esimo=[i,]
axx<-kNNdist(X,k)
head(res$order,n=15)
plot(res)

#altro modo per estra1lo
distanzerech=res$reachdist
head(distanzerech)
distanzerech[1]=distanzerech[2]
#dist=mean(distanzerech[numerid[1,]])
dist=vector()
lrd=vector()

###modifica per evitare problemi dati uguali (dati ripetuti)
for (i in 1:n){
  minptsvicini=numerid[i,]
  dist[i]=mean(distanzerech[numerid[i,]])
  
  
}
distmagg=dist[dist>0]
valore=min(distmagg)
dist[dist==0]=valore/2
lrd=1/dist
#lrd[i]=1/dist[i]
summary(dist)
min(dist)
hist(dist)
dist
head(lrd,20)
numeratore=vector()
OF=vector()
for (i in 1:n){
  minptsvicini=numerid[i,]
  lrd_numero=lrd[i]
  lrd_minpts=lrd[numerid[i,]]
  numeratore[i]=sum(lrd_minpts/lrd_numero)
  OF[i]=numeratore[i]/k
}
summary(lrd)
lrd[max(lrd)]
sum(is.na(OF))
str(OF)
(sum=summary(OF))
str(sum)




###QUI INSERIRE ALBERELLI
(val_opt=rpart_split_finder(OF,length(OF)))
(val_opt2=ctree_split_finder(OF,length(OF)))
####

#OF RISCALATO
plot(sort(OF),type="l",ylim=c(sum[1],sum[6]))

##alternativa
plot(sort(OF), col=is_out[order(OF)]+1)
abline(h=val_opt)
abline(h=val_opt2)
sort(val_opt)
##la soglia è il terzo minore
soglia=1.05
##taglio va effettuato 
out_previsti=rep(0,n)
out_previsti=ifelse(OF>soglia,1,0)
table(out_previsti)
(toptics=table(out_previsti,is_out))



#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
#auc(roc_dbscan)
#Area under the curve: Area under the curve: 0.5396



(precision=toptics[2,2]/sum(toptics[2,]))
#[1] 0.4897959

(recall=toptics[2,2]/sum(toptics[,2]))
#[1] 0.1791045



##PIMA

rm(list=ls())
###devo cambiare distnze per togliere gli zeri
#devo inserire albero nel dbscan
#install.packages("pROC")
library(pROC)
library(rpart)
library(rpart.plot)
library(R.matlab)
library(dbscan)
library(partykit)





###codice operativo


rm(list=ls())
###FUNZIONI
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




setwd("C:/Users/federico/Desktop/Progetto_sl")

path="satellite.mat"

data=readMat(path)
X=data$X
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))

p=dim(X)[2]; n=dim(X)[1]


### DBSCAN non Norm ---------------------------


#primo k, media tra i tre
k=mean(c(p+1,log(n),2*p))
#secondo k max tra i due 
#k2=max(p+1,log(n))
#3)
#k3=max(2*p,log(n))
a=kNNdist(X,k=k)
head(a)
kNNdistplot(X,k=k)
distanze=as.vector(a)
####inserire qui il metodo di taglio con "Alberelli"
head(distanze)


(valori=rpart_split_finder(distanze,length(distanze)))
(valori2=ctree_split_finder(distanze,length(distanze)))
#
plot(sort(distanze),type="l")
#

abline(h=valori,col=valori+1)
abline(h=valori2,col=valori2+1)
###scelta di buonsenso, nessuna di queste
abline(h=40)
eps=40
res=dbscan(X,eps=eps,minPts = k)
indicatore=res$cluster==0
out_previsti=rep(0,n);out_previsti[indicatore]=1
sum(out_previsti)
##outlier reali
sum(is_out)
t3=table(is_out,out_previsti)
t3

hullplot(X,res)



#Confusion Matrix & AUC:
table(DBSCAN=out_previsti, Actual=is_out)
#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
(precision=t3[2,2]/sum(t3[2,]))
#[1] 0.2539

# auc(roc_dbscan)
#Area under the curve: 0.6083

(recall=t3[2,2]/sum(t3[,2]))
#[1] 0.7591777




#####DBSCAN scalato ------------
X=scale(X)
a=kNNdist(X,k=k)
head(a)
kNNdistplot(X,k=k)
distanze=as.vector(a)
####inserire qui il metodo di taglio con "Alberelli"
head(distanze)


(valori=rpart_split_finder(distanze,length(distanze)))
(valori2=ctree_split_finder(distanze,length(distanze)) )
#
plot(sort(distanze),type="l")
#

abline(h=valori,col=valori+1)
abline(h=valori2)
###scelta di buonsenso, nessuno di questi 
abline(h=2.5)

eps=2.4
res=dbscan(X,eps=eps,minPts = k)
indicatore=res$cluster==0
out_previsti=rep(0,n);out_previsti[indicatore]=1
sum(out_previsti)
##outlier reali
sum(is_out)
t3=table(is_out,out_previsti)
t3

hullplot(X,res)



#Confusion Matrix & AUC:
table(DBSCAN=out_previsti, Actual=is_out)
#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
(precision=t3[2,2]/sum(t3[2,]))
#[1] Area under the curve: 0.587
#[1] 0.19980


# auc(roc_dbscan)

(recall=t3[2,2]/sum(t3[,2]))
#0.78119



###OF NON scalato --------------
X=data$X


res<-optics(X,eps=200,minPts = k)
a=kNN(X,k)
head(a$id,3)
numerid=a$id
# numeridi_esimo=[i,]
axx<-kNNdist(X,k)
head(res$order,n=15)
plot(res)

#altro modo per estra1lo
distanzerech=res$reachdist
head(distanzerech)
distanzerech[1]=distanzerech[2]
#dist=mean(distanzerech[numerid[1,]])
dist=vector()
lrd=vector()

###modifica per evitare problemi dati uguali (dati ripetuti)
for (i in 1:n){
  minptsvicini=numerid[i,]
  dist[i]=mean(distanzerech[numerid[i,]])
  
  
}
distmagg=dist[dist>0]
valore=min(distmagg)
dist[dist==0]=valore/2
lrd=1/dist
#lrd[i]=1/dist[i]
summary(dist)
min(dist)
hist(dist)
dist
head(lrd,20)
numeratore=vector()
OF=vector()
for (i in 1:n){
  minptsvicini=numerid[i,]
  lrd_numero=lrd[i]
  lrd_minpts=lrd[numerid[i,]]
  numeratore[i]=sum(lrd_minpts/lrd_numero)
  OF[i]=numeratore[i]/k
}
summary(lrd)
lrd[max(lrd)]
sum(is.na(OF))
str(OF)
(sum=summary(OF))
str(sum)




###QUI INSERIRE ALBERELLI
(val_opt=rpart_split_finder(OF,length(OF)))
(val_opt2=ctree_split_finder(OF,length(OF)))
####

#OF RISCALATO
plot(sort(OF),type="l",ylim=c(sum[1],sum[6]))

##alternativa
plot(sort(OF), col=is_out[order(OF)]+1)
abline(h=val_opt)
sort(val_opt)

##la soglia è secondo piu alto
soglia=sort(val_opt)[5]
##taglio va effettuato 
out_previsti=rep(0,n)
out_previsti=ifelse(OF>soglia,1,0)
table(out_previsti)
(toptics=table(out_previsti,is_out))

#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
#auc(roc_dbscan
#Area under the curve 0.5791



(precision=toptics[2,2]/sum(toptics[2,]))
#[1] 0.51390

(recall=toptics[2,2]/sum(toptics[,2]))
#[1] 0.2841423





###OF SCALATO ------------------
X=scale(X)
res<-optics(X,eps=30,minPts = k)
a=kNN(X,k)
head(a$id,3)
numerid=a$id
# numeridi_esimo=[i,]
axx<-kNNdist(X,k)
head(res$order,n=15)
plot(res)

#altro modo per estra1lo
distanzerech=res$reachdist
head(distanzerech)
distanzerech[1]=distanzerech[2]
#dist=mean(distanzerech[numerid[1,]])
dist=vector()
lrd=vector()

###modifica per evitare problemi dati uguali (dati ripetuti)
for (i in 1:n){
  minptsvicini=numerid[i,]
  dist[i]=mean(distanzerech[numerid[i,]])
  
  
}
distmagg=dist[dist>0]
valore=min(distmagg)
dist[dist==0]=valore/2
lrd=1/dist
#lrd[i]=1/dist[i]
summary(dist)
min(dist)
hist(dist)
dist
head(lrd,20)
numeratore=vector()
OF=vector()
for (i in 1:n){
  minptsvicini=numerid[i,]
  lrd_numero=lrd[i]
  lrd_minpts=lrd[numerid[i,]]
  numeratore[i]=sum(lrd_minpts/lrd_numero)
  OF[i]=numeratore[i]/k
}
summary(lrd)
lrd[max(lrd)]
sum(is.na(OF))
str(OF)
(sum=summary(OF))
str(sum)




###QUI INSERIRE ALBERELLI
(val_opt=rpart_split_finder(OF,length(OF)))
(val_opt2=ctree_split_finder(OF,length(OF)))
####

#OF RISCALATO
plot(sort(OF),type="l",ylim=c(sum[1],sum[6]))

##alternativa
plot(sort(OF), col=is_out[order(OF)]+1)
abline(h=val_opt)
abline(h=val_opt2)
sort(val_opt)
##la soglia è il quarto/quinto
soglia=sort(val_opt)[4]
##taglio va effettuato 
out_previsti=rep(0,n)
out_previsti=ifelse(OF>soglia,1,0)
table(out_previsti)
(toptics=table(out_previsti,is_out))



#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
#auc(roc_dbscan)
#Area under the curve: Area under the curve: 0.575



(precision=toptics[2,2]/sum(toptics[2,]))
#[1] 0.5178399

(recall=toptics[2,2]/sum(toptics[,2]))
#[1] 0.2637525







# BREASTW -----------------------------------------------------------------




###BREASTW


rm(list=ls())
###devo cambiare distnze per togliere gli zeri
#devo inserire albero nel dbscan
#install.packages("pROC")
library(pROC)
library(rpart)
library(rpart.plot)
library(R.matlab)
library(dbscan)
library(partykit)





###codice operativo


rm(list=ls())
###FUNZIONI
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




setwd("C:/Users/federico/Desktop/Progetto_sl")

path="breastw.mat"

data=readMat(path)
X=data$X
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))

p=dim(X)[2]; n=dim(X)[1]

summary(X)
#one hot encoding

fac<-c(1:9)
for (i in 1:length(fac)){
  dati[,i]<-as.factor(dati[,i])
}
#One hot encoding:
dati<-model.matrix(~.-1,dati[,-n])
X=dati[,1:(ncol(dati)-1)]
p=dim(X)[2]; n=dim(X)[1]

### DBSCAN non Norm ---------------------------


#primo k, media tra i tre
k=mean(c(p+1,log(n),2*p))
#secondo k max tra i due 
#k2=max(p+1,log(n))
#3)
#k3=max(2*p,log(n))
a=kNNdist(X,k=k)
head(a)
kNNdistplot(X,k=k)
distanze=as.vector(a)
####inserire qui il metodo di taglio con "Alberelli"
head(distanze)


(valori=rpart_split_finder(distanze,length(distanze)))
(valori2=ctree_split_finder(distanze,length(distanze)))
#
plot(sort(distanze),type="l")
max(distanze)
#

abline(h=valori,col=valori+1)
abline(h=valori2,col=valori2+1)
abline(h=5)
###scelta di buonsenso, secondo piu alto
abline(h=40)
eps=sort(valori)[4]
res=dbscan(X,eps=eps,minPts = k)
indicatore=res$cluster==0
out_previsti=rep(0,n);out_previsti[indicatore]=1
sum(out_previsti)
##outlier reali
sum(is_out)
t3=table(is_out,out_previsti)
t3

hullplot(X,res)



#Confusion Matrix & AUC:
table(DBSCAN=out_previsti, Actual=is_out)
#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
(precision=t3[2,2]/sum(t3[2,]))
#[1] 0.9748954

# auc(roc_dbscan)
#Area under the curve: 0.9706

(recall=t3[2,2]/sum(t3[,2]))
#[1] 0.9395161




#####DBSCAN scalato ------------
head(X)
X=scale(X)
head(X)
a=kNNdist(X,k=k)
head(a)
kNNdistplot(X,k=k)
distanze=as.vector(a)
####inserire qui il metodo di taglio con "Alberelli"
head(distanze)


(valori=rpart_split_finder(distanze,length(distanze)))
(valori2=ctree_split_finder(distanze,length(distanze)) )
#
plot(sort(distanze),type="l")
#

abline(h=valori,col=valori+1)
abline(h=valori2)
###scelta di buonsenso, nessuno di questi 
abline(h=0.15)

eps=max(valori)
res=dbscan(X,eps=eps,minPts = k)
indicatore=res$cluster==0
out_previsti=rep(0,n);out_previsti[indicatore]=1
sum(out_previsti)
##outlier reali
sum(is_out)
t3=table(is_out,out_previsti)
t3

hullplot(X,res)



#Confusion Matrix & AUC:
table(DBSCAN=out_previsti, Actual=is_out)
#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
(precision=t3[2,2]/sum(t3[2,]))
#[1] Area under the curve: 0.9691
#0.9832636


# auc(roc_dbscan)

(recall=t3[2,2]/sum(t3[,2]))
#0.9215686



###OF NON scalato --------------
X=data$X


res<-optics(X,eps=20,minPts = k)
a=kNN(X,k)
head(a$id,3)
numerid=a$id
# numeridi_esimo=[i,]
axx<-kNNdist(X,k)
head(res$order,n=15)
plot(res)

#altro modo per estra1lo
distanzerech=res$reachdist
head(distanzerech)
distanzerech[1]=distanzerech[2]
#dist=mean(distanzerech[numerid[1,]])
dist=vector()
lrd=vector()

###modifica per evitare problemi dati uguali (dati ripetuti)
for (i in 1:n){
  minptsvicini=numerid[i,]
  dist[i]=mean(distanzerech[numerid[i,]])
  
  
}
distmagg=dist[dist>0]
valore=min(distmagg)
dist[dist==0]=valore/2
lrd=1/dist
#lrd[i]=1/dist[i]
summary(dist)
min(dist)
hist(dist)
dist
head(lrd,20)
numeratore=vector()
OF=vector()
for (i in 1:n){
  minptsvicini=numerid[i,]
  lrd_numero=lrd[i]
  lrd_minpts=lrd[numerid[i,]]
  numeratore[i]=sum(lrd_minpts/lrd_numero)
  OF[i]=numeratore[i]/k
}
summary(lrd)
lrd[max(lrd)]
sum(is.na(OF))
str(OF)
(sum=summary(OF))
str(sum)




###QUI INSERIRE ALBERELLI
(val_opt=rpart_split_finder(OF,length(OF)))
(val_opt2=ctree_split_finder(OF,length(OF)))
####

#OF RISCALATO
plot(sort(OF),type="l",ylim=c(sum[1],sum[6]))

##alternativa
plot(sort(OF), col=is_out[order(OF)]+1)
abline(h=val_opt)
sort(val_opt)

##la soglia è sesto dal basso
soglia=sort(val_opt)[6]
##taglio va effettuato 
out_previsti=rep(0,n)
out_previsti=ifelse(OF>soglia,1,0)
table(out_previsti)
(toptics=table(out_previsti,is_out))

#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
#auc(roc_dbscan
#Area under the curve 0.6053



(precision=toptics[2,2]/sum(toptics[2,]))
#[1] 0.5869565

(recall=toptics[2,2]/sum(toptics[,2]))
#[1] 0.3389121





###OF SCALATO ------------------
X=scale(X)
res<-optics(X,eps=30,minPts = k)
a=kNN(X,k)
head(a$id,3)
numerid=a$id
# numeridi_esimo=[i,]
axx<-kNNdist(X,k)
head(res$order,n=15)
plot(res)

#altro modo per estra1lo
distanzerech=res$reachdist
head(distanzerech)
distanzerech[1]=distanzerech[2]
#dist=mean(distanzerech[numerid[1,]])
dist=vector()
lrd=vector()

###modifica per evitare problemi dati uguali (dati ripetuti)
for (i in 1:n){
  minptsvicini=numerid[i,]
  dist[i]=mean(distanzerech[numerid[i,]])
  
  
}
distmagg=dist[dist>0]
valore=min(distmagg)
dist[dist==0]=valore/2
lrd=1/dist
#lrd[i]=1/dist[i]
summary(dist)
min(dist)
hist(dist)
dist
head(lrd,20)
numeratore=vector()
OF=vector()
for (i in 1:n){
  minptsvicini=numerid[i,]
  lrd_numero=lrd[i]
  lrd_minpts=lrd[numerid[i,]]
  numeratore[i]=sum(lrd_minpts/lrd_numero)
  OF[i]=numeratore[i]/k
}
summary(lrd)
lrd[max(lrd)]
sum(is.na(OF))
str(OF)
(sum=summary(OF))
str(sum)




###QUI INSERIRE ALBERELLI
(val_opt=rpart_split_finder(OF,length(OF)))
(val_opt2=ctree_split_finder(OF,length(OF)))
####

#OF RISCALATO
plot(sort(OF),type="l",ylim=c(sum[1],sum[6]))

##alternativa
plot(sort(OF), col=is_out[order(OF)]+1)
abline(h=val_opt)
abline(h=val_opt2)
sort(val_opt)
##la soglia è il quarto/quinto
soglia=sort(val_opt)[4]
##taglio va effettuato 
out_previsti=rep(0,n)
out_previsti=ifelse(OF>soglia,1,0)
table(out_previsti)
(toptics=table(out_previsti,is_out))



#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
#auc(roc_dbscan)
#Area under the curve: Area under the curve: 0.7337



(precision=toptics[2,2]/sum(toptics[2,]))
#[1] 0.5610465

(recall=toptics[2,2]/sum(toptics[,2]))
#[1] 0.8075314




rm(list=ls())
###devo cambiare distnze per togliere gli zeri
#devo inserire albero nel dbscan
#install.packages("pROC")
library(pROC)
library(rpart)
library(rpart.plot)
library(R.matlab)
library(dbscan)
library(partykit)





###codice operativo


rm(list=ls())
###FUNZIONI
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




setwd("C:/Users/federico/Desktop/Progetto_sl")

path="arrhythmia.mat"

data=readMat(path)
X=data$X
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))

p=dim(X)[2]; n=dim(X)[1]

summary(X)
### DBSCAN non Norm ---------------------------


#primo k, media tra i tre
k=mean(c(p+1,log(n),2*p))
#secondo k max tra i due 
#k2=max(p+1,log(n))
#3)
#k3=max(2*p,log(n))
a=kNNdist(X,k=k)
head(a)
kNNdistplot(X,k=k)
distanze=as.vector(a)
####inserire qui il metodo di taglio con "Alberelli"
head(distanze)


(valori=rpart_split_finder(distanze,length(distanze)))
(valori2=ctree_split_finder(distanze,length(distanze)))
#
plot(sort(distanze),type="l")
#

abline(h=valori,col=valori+1)
abline(h=valori2,col=valori2+1)
###scelta di buonsenso, nessuna di queste
abline(h=40)
eps=300
res=dbscan(X,eps=eps,minPts = k)
indicatore=res$cluster==0
out_previsti=rep(0,n);out_previsti[indicatore]=1
sum(out_previsti)
##outlier reali
sum(is_out)
t3=table(is_out,out_previsti)
t3

hullplot(X,res)



#Confusion Matrix & AUC:
table(DBSCAN=out_previsti, Actual=is_out)
#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
(precision=t3[2,2]/sum(t3[2,]))
#[1] 0.2539

# auc(roc_dbscan)
#Area under the curve: 0.6083

(recall=t3[2,2]/sum(t3[,2]))
#[1] 0.7591777



summary(X)
sum(is.na(X))
#####DBSCAN scalato ------------
togliere<-which(nearZeroVar(dati, saveMetrics = T)$zeroVar==TRUE)
dati=dati[,-togliere]
X=dati[,-ncol(dati)]
p=dim(X)[2]; n=dim(X)[1]
summary(X)
X=scale(X)
sum(is.na(X))
a=kNNdist(X,k=k)
head(a)
kNNdistplot(X,k=k)
distanze=as.vector(a)
####inserire qui il metodo di taglio con "Alberelli"
head(distanze)


(valori=rpart_split_finder(distanze,length(distanze)))
(valori2=ctree_split_finder(distanze,length(distanze)) )
#
plot(sort(distanze),type="l")
#

abline(h=valori,col=valori+1)
abline(h=valori2)
###scelta di buonsenso, nessuno di questi 
abline(h=2.5)

eps=20
res=dbscan(X,eps=eps,minPts = k)
indicatore=res$cluster==0
out_previsti=rep(0,n);out_previsti[indicatore]=1
sum(out_previsti)
##outlier reali
sum(is_out)
t3=table(is_out,out_previsti)
t3

hullplot(X,res)



#Confusion Matrix & AUC:
table(DBSCAN=out_previsti, Actual=is_out)
#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
(precision=t3[2,2]/sum(t3[2,]))
#[1] Area under the curve: 0.582
#[1] 0.19980


# auc(roc_dbscan)

(recall=t3[2,2]/sum(t3[,2]))
#0.78119



###OF NON scalato --------------
X=data$X


res<-optics(X,eps=400,minPts = k)
a=kNN(X,k)
head(a$id,3)
numerid=a$id
# numeridi_esimo=[i,]
axx<-kNNdist(X,k)
head(res$order,n=15)
plot(res)

#altro modo per estra1lo
distanzerech=res$reachdist
head(distanzerech)
distanzerech[1]=distanzerech[2]
#dist=mean(distanzerech[numerid[1,]])
dist=vector()
lrd=vector()

###modifica per evitare problemi dati uguali (dati ripetuti)
for (i in 1:n){
  minptsvicini=numerid[i,]
  dist[i]=mean(distanzerech[numerid[i,]])
  
  
}
distmagg=dist[dist>0]
valore=min(distmagg)
dist[dist==0]=valore/2
lrd=1/dist
#lrd[i]=1/dist[i]
summary(dist)
min(dist)
hist(dist)
dist
head(lrd,20)
numeratore=vector()
OF=vector()
for (i in 1:n){
  minptsvicini=numerid[i,]
  lrd_numero=lrd[i]
  lrd_minpts=lrd[numerid[i,]]
  numeratore[i]=sum(lrd_minpts/lrd_numero)
  OF[i]=numeratore[i]/k
}
summary(lrd)
lrd[max(lrd)]
sum(is.na(OF))
str(OF)
(sum=summary(OF))
str(sum)




###QUI INSERIRE ALBERELLI
(val_opt=rpart_split_finder(OF,length(OF)))
(val_opt2=ctree_split_finder(OF,length(OF)))
####

#OF RISCALATO
plot(sort(OF),type="l",ylim=c(sum[1],sum[6]))

##alternativa
plot(sort(OF), col=is_out[order(OF)]+1)
abline(h=val_opt)
sort(val_opt)

##la soglia è secondo piu alto
soglia=1.01
##taglio va effettuato 
out_previsti=rep(0,n)
out_previsti=ifelse(OF>soglia,1,0)
table(out_previsti)
(toptics=table(out_previsti,is_out))

#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
#auc(roc_dbscan
#Area under the curve 0.675



(precision=toptics[2,2]/sum(toptics[2,]))
#[1] 0.51390

(recall=toptics[2,2]/sum(toptics[,2]))
#[1] 0.2841423





###OF SCALATO ------------------
data=readMat(path)
X=data$X
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
togliere<-which(nearZeroVar(dati, saveMetrics = T)$zeroVar==TRUE)
dati=dati[,-togliere]
X=dati[,-ncol(dati)]
p=dim(X)[2]; n=dim(X)[1]
summary(X)
X=scale(X)
X=scale(X)
X=sum(is.na(X))
res<-optics(X,eps=300,minPts = k)
a=kNN(X,k)
head(a$id,3)
numerid=a$id
# numeridi_esimo=[i,]
axx<-kNNdist(X,k)
head(res$order,n=15)
plot(res)

#altro modo per estra1lo
distanzerech=res$reachdist
head(distanzerech)
distanzerech[1]=distanzerech[2]
#dist=mean(distanzerech[numerid[1,]])
dist=vector()
lrd=vector()

###modifica per evitare problemi dati uguali (dati ripetuti)
for (i in 1:n){
  minptsvicini=numerid[i,]
  dist[i]=mean(distanzerech[numerid[i,]])
  
  
}
distmagg=dist[dist>0]
valore=min(distmagg)
dist[dist==0]=valore/2
lrd=1/dist
#lrd[i]=1/dist[i]
summary(dist)
min(dist)
hist(dist)
dist
head(lrd,20)
numeratore=vector()
OF=vector()
for (i in 1:n){
  minptsvicini=numerid[i,]
  lrd_numero=lrd[i]
  lrd_minpts=lrd[numerid[i,]]
  numeratore[i]=sum(lrd_minpts/lrd_numero)
  OF[i]=numeratore[i]/k
}
summary(lrd)
lrd[max(lrd)]
sum(is.na(OF))
str(OF)
(sum=summary(OF))
str(sum)




###QUI INSERIRE ALBERELLI
(val_opt=rpart_split_finder(OF,length(OF)))
(val_opt2=ctree_split_finder(OF,length(OF)))
####

#OF RISCALATO
plot(sort(OF),type="l",ylim=c(sum[1],sum[6]))

##alternativa
plot(sort(OF), col=is_out[order(OF)]+1)
abline(h=val_opt)
abline(h=val_opt2)
sort(val_opt)
##la soglia è il quarto/quinto
soglia=1.008
##taglio va effettuato 
out_previsti=rep(0,n)
out_previsti=ifelse(OF>soglia,1,0)
table(out_previsti)
(toptics=table(out_previsti,is_out))



#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
#auc(roc_dbscan)
#Area under the curve: Area under the curve: 0.645



(precision=toptics[2,2]/sum(toptics[2,]))
#[1] 0.5178399

(recall=toptics[2,2]/sum(toptics[,2]))
#[1] 0.2637525






##IONO

#IONOSHPERE

rm(list=ls())
###devo cambiare distnze per togliere gli zeri
#devo inserire albero nel dbscan
#install.packages("pROC")
library(pROC)
library(rpart)
library(rpart.plot)
library(R.matlab)
library(dbscan)
library(partykit)





###codice operativo


rm(list=ls())
###FUNZIONI
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




setwd("C:/Users/federico/Desktop/Progetto_sl")

path="ionosphere.mat"

data=readMat(path)
X=data$X
summary(X)
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))

p=dim(X)[2]; n=dim(X)[1]

summary(X)

### DBSCAN non Norm ---------------------------


#primo k, media tra i tre
k=mean(c(p+1,log(n),2*p))
#secondo k max tra i due 
#k2=max(p+1,log(n))
#3)
#k3=max(2*p,log(n))
a=kNNdist(X,k=k)
head(a)
kNNdistplot(X,k=k)
distanze=as.vector(a)
####inserire qui il metodo di taglio con "Alberelli"
head(distanze)


(valori=rpart_split_finder(distanze,length(distanze)))
(valori2=ctree_split_finder(distanze,length(distanze)))
#
plot(sort(distanze),type="l")
max(distanze)
#

abline(h=valori,col=valori+1)
abline(h=valori2,col=valori2+1)
abline(h=5)
###scelta di buonsenso, max
abline(h=40)
eps=max(valori)
res=dbscan(X,eps=eps,minPts = k)
indicatore=res$cluster==0
out_previsti=rep(0,n);out_previsti[indicatore]=1
sum(out_previsti)
##outlier reali
sum(is_out)
t3=table(is_out,out_previsti)
t3

hullplot(X,res)



#Confusion Matrix & AUC:
table(DBSCAN=out_previsti, Actual=is_out)
#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
(precision=t3[2,2]/sum(t3[2,]))
#[1] 0.4047619
#Area under the curve: 0.7024
(recall=t3[2,2]/sum(t3[,2]))
#[1] 1

###ALTERNATIVA: 
#EPS=2
#AUC=0.7416
#PRECISION=0.71428
#RECALL=0.6338208



###OF NON scalato --------------
X=data$X


res<-optics(X,eps=20,minPts = k)
a=kNN(X,k)
head(a$id,3)
numerid=a$id
# numeridi_esimo=[i,]
axx<-kNNdist(X,k)
head(res$order,n=15)
plot(res)

#altro modo per estra1lo
distanzerech=res$reachdist
head(distanzerech)
distanzerech[1]=distanzerech[2]
#dist=mean(distanzerech[numerid[1,]])
dist=vector()
lrd=vector()

###modifica per evitare problemi dati uguali (dati ripetuti)
for (i in 1:n){
  minptsvicini=numerid[i,]
  dist[i]=mean(distanzerech[numerid[i,]])
  
  
}
distmagg=dist[dist>0]
valore=min(distmagg)
dist[dist==0]=valore/2
lrd=1/dist
#lrd[i]=1/dist[i]
summary(dist)
min(dist)
hist(dist)
dist
head(lrd,20)
numeratore=vector()
OF=vector()
for (i in 1:n){
  minptsvicini=numerid[i,]
  lrd_numero=lrd[i]
  lrd_minpts=lrd[numerid[i,]]
  numeratore[i]=sum(lrd_minpts/lrd_numero)
  OF[i]=numeratore[i]/k
}
summary(lrd)
lrd[max(lrd)]
sum(is.na(OF))
str(OF)
(sum=summary(OF))
str(sum)




###QUI INSERIRE ALBERELLI
(val_opt=rpart_split_finder(OF,length(OF)))
(val_opt2=ctree_split_finder(OF,length(OF)))
####

#OF RISCALATO
plot(sort(OF),type="l",ylim=c(sum[1],sum[6]))

##alternativa
plot(sort(OF), col=is_out[order(OF)]+1)
abline(h=val_opt)
sort(val_opt)
##piglio il 4
##la soglia è il quarto
soglia=sort(val_opt)[4]
##taglio va effettuato 
out_previsti=rep(0,n)
out_previsti=ifelse(OF>soglia,1,0)
table(out_previsti)
(toptics=table(out_previsti,is_out))

#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
#auc(roc_dbscan
#Area under the curve: 0.8227



(precision=toptics[2,2]/sum(toptics[2,]))
#[1] 0.6768293

(recall=toptics[2,2]/sum(toptics[,2]))
#[1] 0.8809524







# MNIST -------------------------------------------------------------------


##MNIST

rm(list=ls())
###devo cambiare distnze per togliere gli zeri
#devo inserire albero nel dbscan
#install.packages("pROC")
library(pROC)
library(rpart)
library(rpart.plot)
library(R.matlab)
library(dbscan)
library(partykit)





###codice operativo


rm(list=ls())
###FUNZIONI
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

###DATI ---------------------


setwd("C:/Users/federico/Desktop/Progetto_sl")
library(caret)
path="mnist.mat"

data=readMat(path)
X=data$X
summary(X)
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
togliere<-which(nearZeroVar(dati, saveMetrics = T)$zeroVar==TRUE)
dati=dati[,-togliere]
X=dati[,-ncol(dati)]
p=dim(X)[2]; n=dim(X)[1]
summary(X)

### DBSCAN non Norm ---------------------------


#primo k, media tra i tre
k=mean(c(p+1,log(n),2*p))
#secondo k max tra i due 
#k2=max(p+1,log(n))
#3)
#k3=max(2*p,log(n))
a=kNNdist(X,k=k)
head(a)
kNNdistplot(X,k=k)
distanze=as.vector(a)
####inserire qui il metodo di taglio con "Alberelli"
head(distanze)


(valori=rpart_split_finder(distanze,length(distanze)))
(valori2=ctree_split_finder(distanze,length(distanze)))
#
plot(sort(distanze),type="l")
max(distanze)
#

abline(h=valori,col=valori+1)
abline(h=valori2,col=valori2+1)
abline(h=5)
###scelta di buonsenso, max
abline(h=40)
eps=max(valori)
res=dbscan(X,eps=eps,minPts = k)
indicatore=res$cluster==0
out_previsti=rep(0,n);out_previsti[indicatore]=1
sum(out_previsti)
##outlier reali
sum(is_out)
t3=table(is_out,out_previsti)
t3

hullplot(X,res)



#Confusion Matrix & AUC:
table(DBSCAN=out_previsti, Actual=is_out)
#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
(precision=t3[2,2]/sum(t3[2,]))
#[1] 0.1714286


# auc(roc_dbscan)
#Area under the curve: 0.5816



(recall=t3[2,2]/sum(t3[,2]))
#[1][1] 0.6779661




#####DBSCAN scalato ------------
head(X)
X=scale(X)
head(X)
a=kNNdist(X,k=k)
head(a)
kNNdistplot(X,k=k)
distanze=as.vector(a)
####inserire qui il metodo di taglio con "Alberelli"
head(distanze)


(valori=rpart_split_finder(distanze,length(distanze)))
(valori2=ctree_split_finder(distanze,length(distanze)) )
#
plot(sort(distanze),type="l")
#

abline(h=valori,col=valori+1)
abline(h=valori2)
###scelta di buonsenso, nessuno di questi 
abline(h=10)

eps=10
res=dbscan(X,eps=eps,minPts = k)
indicatore=res$cluster==0
out_previsti=rep(0,n);out_previsti[indicatore]=1
sum(out_previsti)
##outlier reali
sum(is_out)
t3=table(is_out,out_previsti)
t3

hullplot(X,res)



#Confusion Matrix & AUC:
table(DBSCAN=out_previsti, Actual=is_out)
#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
(precision=t3[2,2]/sum(t3[2,]))
#Area under the curve: 0.5759
#0.1628571


# auc(roc_dbscan)

(recall=t3[2,2]/sum(t3[,2]))
#0.5968586



###OF NON scalato
##impossibile
###OF SCALATO ------------------
X=scale(X)
res<-optics(X,eps=30,minPts = k)
a=kNN(X,k)
head(a$id,3)
numerid=a$id
# numeridi_esimo=[i,]
axx<-kNNdist(X,k)
head(res$order,n=15)
plot(res)

#altro modo per estra1lo
distanzerech=res$reachdist
head(distanzerech)
distanzerech[1]=distanzerech[2]
#dist=mean(distanzerech[numerid[1,]])
dist=vector()
lrd=vector()

###modifica per evitare problemi dati uguali (dati ripetuti)
for (i in 1:n){
  minptsvicini=numerid[i,]
  dist[i]=mean(distanzerech[numerid[i,]])
  
  
}
distmagg=dist[dist>0]
valore=min(distmagg)
dist[dist==0]=valore/2
lrd=1/dist
#lrd[i]=1/dist[i]
summary(dist)
min(dist)
hist(dist)
dist
head(lrd,20)
numeratore=vector()
OF=vector()
for (i in 1:n){
  minptsvicini=numerid[i,]
  lrd_numero=lrd[i]
  lrd_minpts=lrd[numerid[i,]]
  numeratore[i]=sum(lrd_minpts/lrd_numero)
  OF[i]=numeratore[i]/k
}
summary(lrd)
lrd[max(lrd)]
sum(is.na(OF))
str(OF)
(sum=summary(OF))
str(sum)




###QUI INSERIRE ALBERELLI
(val_opt=rpart_split_finder(OF,length(OF)))
(val_opt2=ctree_split_finder(OF,length(OF)))
####

#OF RISCALATO
plot(sort(OF),type="l",ylim=c(sum[1],sum[6]))

##alternativa
plot(sort(OF), col=is_out[order(OF)]+1)
abline(h=val_opt)
abline(h=val_opt2)
sort(val_opt)
##la soglia è il terzo
soglia=sort(val_opt2)[3]
##taglio va effettuato 
out_previsti=rep(0,n)
out_previsti=ifelse(OF>soglia,1,0)
table(out_previsti)
(toptics=table(out_previsti,is_out))



#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
#auc(roc_dbscan)
#Area under the curve: Area under the curve: 0.6768



(precision=toptics[2,2]/sum(toptics[2,]))
#[1] 0.5384615

(recall=toptics[2,2]/sum(toptics[,2]))
#[1] 0.3866279









##OPTIDIGITS

rm(list=ls())
###devo cambiare distnze per togliere gli zeri
#devo inserire albero nel dbscan
#install.packages("pROC")
library(pROC)
library(rpart)
library(rpart.plot)
library(R.matlab)
library(dbscan)
library(partykit)





###codice operativo


rm(list=ls())
###FUNZIONI
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

###DATI ---------------------


setwd("C:/Users/federico/Desktop/Progetto_sl")
library(caret)
path="optdigits.mat"

data=readMat(path)
X=data$X
summary(X)
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))
togliere<-which(nearZeroVar(dati, saveMetrics = T)$zeroVar==TRUE)
dati=dati[,-togliere]
X=dati[,-ncol(dati)]
p=dim(X)[2]; n=dim(X)[1]
summary(X)

### DBSCAN non Norm ---------------------------


#primo k, media tra i tre
k=mean(c(p+1,log(n),2*p))
#secondo k max tra i due 
#k2=max(p+1,log(n))
#3)
#k3=max(2*p,log(n))
a=kNNdist(X,k=k)
head(a)
kNNdistplot(X,k=k)
distanze=as.vector(a)
####inserire qui il metodo di taglio con "Alberelli"
head(distanze)


(valori=rpart_split_finder(distanze,length(distanze)))
(valori2=ctree_split_finder(distanze,length(distanze)))
#
plot(sort(distanze),type="l")
max(distanze)
#

abline(h=valori,col=valori+1)
abline(h=valori2,col=valori2+1)
abline(h=5)
###scelta di buonsenso, max
abline(h=40)
eps=max(valori)
res=dbscan(X,eps=eps,minPts = k)
indicatore=res$cluster==0
out_previsti=rep(0,n);out_previsti[indicatore]=1
sum(out_previsti)
##outlier reali
sum(is_out)
t3=table(is_out,out_previsti)
t3

hullplot(X,res)



#Confusion Matrix & AUC:
table(DBSCAN=out_previsti, Actual=is_out)
#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan) ##o.4951
(precision=t3[2,2]/sum(t3[2,]))
#[1] 0.006667


# auc(roc_dbscan)



(recall=t3[2,2]/sum(t3[,2]))
#[1][1] 0.01111




#####DBSCAN scalato ------------
head(X)
X=scale(X)
head(X)
a=kNNdist(X,k=k)
head(a)
kNNdistplot(X,k=k)
distanze=as.vector(a)
####inserire qui il metodo di taglio con "Alberelli"
head(distanze)


(valori=rpart_split_finder(distanze,length(distanze)))
(valori2=ctree_split_finder(distanze,length(distanze)) )
#
plot(sort(distanze),type="l")
#

abline(h=valori,col=valori+1)
abline(h=valori2)
###scelta di buonsenso, nessuno di questi 
abline(h=10)

eps=10
res=dbscan(X,eps=eps,minPts = k)
indicatore=res$cluster==0
out_previsti=rep(0,n);out_previsti[indicatore]=1
sum(out_previsti)
##outlier reali
sum(is_out)
t3=table(is_out,out_previsti)
t3

hullplot(X,res)



#Confusion Matrix & AUC:
table(DBSCAN=out_previsti, Actual=is_out)
#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
(precision=t3[2,2]/sum(t3[2,]))
#Area under the curve:0.4941
#0


# auc(roc_dbscan)

(recall=t3[2,2]/sum(t3[,2]))
#0



###OF NON scalato-------------------
X=data$X


res<-optics(X,eps=200,minPts = k)
a=kNN(X,k)
head(a$id,3)
numerid=a$id
# numeridi_esimo=[i,]
axx<-kNNdist(X,k)
head(res$order,n=15)
plot(res)

#altro modo per estra1lo
distanzerech=res$reachdist
head(distanzerech)
distanzerech[1]=distanzerech[2]
#dist=mean(distanzerech[numerid[1,]])
dist=vector()
lrd=vector()

###modifica per evitare problemi dati uguali (dati ripetuti)
for (i in 1:n){
  minptsvicini=numerid[i,]
  dist[i]=mean(distanzerech[numerid[i,]])
  
  
}
distmagg=dist[dist>0]
valore=min(distmagg)
dist[dist==0]=valore/2
lrd=1/dist
#lrd[i]=1/dist[i]
summary(dist)
min(dist)
hist(dist)
dist
head(lrd,20)
numeratore=vector()
OF=vector()
for (i in 1:n){
  minptsvicini=numerid[i,]
  lrd_numero=lrd[i]
  lrd_minpts=lrd[numerid[i,]]
  numeratore[i]=sum(lrd_minpts/lrd_numero)
  OF[i]=numeratore[i]/k
}
summary(lrd)
lrd[max(lrd)]
sum(is.na(OF))
str(OF)
(sum=summary(OF))
str(sum)




###QUI INSERIRE ALBERELLI
(val_opt=rpart_split_finder(OF,length(OF)))
(val_opt2=ctree_split_finder(OF,length(OF)))
####

#OF RISCALATO
plot(sort(OF),type="l",ylim=c(sum[1],sum[6]))

##alternativa
plot(sort(OF), col=is_out[order(OF)]+1)
abline(h=val_opt)
sort(val_opt)

##la soglia è nessunodi qusti valori
soglia=1.04
##taglio va effettuato 
out_previsti=rep(0,n)
out_previsti=ifelse(OF>soglia,1,0)
table(out_previsti)
(toptics=table(out_previsti,is_out))

#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
#auc(roc_dbscan
#Area under the curve: 0.4644




(precision=toptics[2,2]/sum(toptics[2,]))
#[1] 0.002531

(recall=toptics[2,2]/sum(toptics[,2]))
#[1] 0.006666667






###OF SCALATO ------------------
X=scale(X)
res<-optics(X,eps=50,minPts = k)
a=kNN(X,k)
head(a$id,3)
numerid=a$id
# numeridi_esimo=[i,]
axx<-kNNdist(X,k)
head(res$order,n=15)
plot(res)

#altro modo per estra1lo
distanzerech=res$reachdist
head(distanzerech)
distanzerech[1]=distanzerech[2]
#dist=mean(distanzerech[numerid[1,]])
dist=vector()
lrd=vector()

###modifica per evitare problemi dati uguali (dati ripetuti)
for (i in 1:n){
  minptsvicini=numerid[i,]
  dist[i]=mean(distanzerech[numerid[i,]])
  
  
}
distmagg=dist[dist>0]
valore=min(distmagg)
dist[dist==0]=valore/2
lrd=1/dist
#lrd[i]=1/dist[i]
summary(dist)
min(dist)
hist(dist)
dist
head(lrd,20)
numeratore=vector()
OF=vector()
for (i in 1:n){
  minptsvicini=numerid[i,]
  lrd_numero=lrd[i]
  lrd_minpts=lrd[numerid[i,]]
  numeratore[i]=sum(lrd_minpts/lrd_numero)
  OF[i]=numeratore[i]/k
}
summary(lrd)
lrd[max(lrd)]
sum(is.na(OF))
str(OF)
(sum=summary(OF))
str(sum)




###QUI INSERIRE ALBERELLI
(val_opt=rpart_split_finder(OF,length(OF)))
(val_opt2=ctree_split_finder(OF,length(OF)))
####

#OF RISCALATO
plot(sort(OF),type="l",ylim=c(sum[1],sum[6]))

##alternativa
plot(sort(OF), col=is_out[order(OF)]+1)
abline(h=val_opt)
abline(h=val_opt2)
sort(val_opt)
##la soglia è nessun valore di questi
soglia=1.1
##taglio va effettuato 
out_previsti=rep(0,n)
out_previsti=ifelse(OF>soglia,1,0)
table(out_previsti)
(toptics=table(out_previsti,is_out))



#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
#auc(roc_dbscan)
#Area under the curve: Area under the curve: 0.4754



(precision=toptics[2,2]/sum(toptics[2,]))
#[1] 0

(recall=toptics[2,2]/sum(toptics[,2]))
#[1] 0



##HTTP


rm(list=ls())
###devo cambiare distnze per togliere gli zeri
#devo inserire albero nel dbscan
#install.packages("pROC")
library(pROC)
library(rpart)
library(rpart.plot)
library(R.matlab)
library(dbscan)
library(partykit)





###codice operativo


rm(list=ls())
###FUNZIONI
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




setwd("C:/Users/federico/Desktop/Progetto_sl")

path="X_http.mat"

data=readMat(path2)
X=data$X
path2="y_http.mat"
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))

p=dim(X)[2]; n=dim(X)[1]

summary(X)
#one hot encoding

### DBSCAN non Norm ---------------------------


#primo k, media tra i tre
k=mean(c(p+1,log(n),2*p))
#secondo k max tra i due 
#k2=max(p+1,log(n))
#3)
#k3=max(2*p,log(n))
a=kNNdist(X,k=k)
head(a)
kNNdistplot(X,k=k)
distanze=as.vector(a)
####inserire qui il metodo di taglio con "Alberelli"
head(distanze)


(valori=rpart_split_finder(distanze,length(distanze)))
(valori2=ctree_split_finder(distanze,length(distanze)))
#
plot(sort(distanze),type="l",ylim=c(0,0.1))
max(distanze)
#

abline(h=valori,col=valori+1)

abline(h=valori2,col=valori2+1)
abline(h=0.01,col=3)
###scelta di buonsenso, secondo piu alto
abline(h=40)
eps=0.01
res=dbscan(X,eps=eps,minPts = k)
indicatore=res$cluster==0
out_previsti=rep(0,n);out_previsti[indicatore]=1
sum(out_previsti)
##outlier reali
sum(is_out)
t3=table(is_out,out_previsti)
t3

#hullplot(X,res)



#Confusion Matrix & AUC:
table(DBSCAN=out_previsti, Actual=is_out)
#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
(precision=t3[2,2]/sum(t3[2,]))
#[1] 0.04115785

# auc(roc_dbscan)
#Area under the curve: 0.5061

(recall=t3[2,2]/sum(t3[,2]))
#[1] 0.005545399




#####DBSCAN scalato ------------
head(X)
X=scale(X)
head(X)
a=kNNdist(X,k=k)
head(a)
#kNNdistplot(X,k=k)
distanze=as.vector(a)
####inserire qui il metodo di taglio con "Alberelli"
head(distanze)


(valori=rpart_split_finder(distanze,length(distanze)))
(valori2=ctree_split_finder(distanze,length(distanze)) )
#
plot(sort(distanze),type="l",ylim=c(0,0.1))
#
abline(h=0.0125)

abline(h=valori,col=valori+1)
abline(h=valori2)
###scelta di buonsenso, nessuno di questi 
abline(h=0.15)

eps=0.018
res=dbscan(X,eps=eps,minPts = k)
indicatore=res$cluster==0
out_previsti=rep(0,n);out_previsti[indicatore]=1
sum(out_previsti)
##outlier reali
sum(is_out)
t3=table(is_out,out_previsti)
t3

hullplot(X,res)



#Confusion Matrix & AUC:
table(DBSCAN=out_previsti, Actual=is_out)
#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
(precision=t3[2,2]/sum(t3[2,]))
#[1] Area under the curve: #0.5121

#???0.005545399


# auc(roc_dbscan)

(recall=t3[2,2]/sum(t3[,2]))
#0.04115785



###OF scalato --------------

path="X_http.mat"

data=readMat(path)
X=data$X
X=data$X
X

res<-optics(X,eps=0.1,minPts = k)
a=kNN(X,k)
head(a$id,3)
numerid=a$id
# numeridi_esimo=[i,]
axx<-kNNdist(X,k)
head(res$order,n=15)
plot(res)

#altro modo per estra1lo
distanzerech=res$reachdist
head(distanzerech)
distanzerech[1]=distanzerech[2]
#dist=mean(distanzerech[numerid[1,]])
dist=vector()
lrd=vector()

###modifica per evitare problemi dati uguali (dati ripetuti)
for (i in 1:n){
  minptsvicini=numerid[i,]
  dist[i]=mean(distanzerech[numerid[i,]])
  
  
}
distmagg=dist[dist>0]
valore=min(distmagg)
dist[dist==0]=valore/2
lrd=1/dist
#lrd[i]=1/dist[i]
summary(dist)
min(dist)
hist(dist)
dist
head(lrd,20)
numeratore=vector()
OF=vector()
for (i in 1:n){
  minptsvicini=numerid[i,]
  lrd_numero=lrd[i]
  lrd_minpts=lrd[numerid[i,]]
  numeratore[i]=sum(lrd_minpts/lrd_numero)
  OF[i]=numeratore[i]/k
}
summary(lrd)
lrd[max(lrd)]
sum(is.na(OF))
str(OF)
(sum=summary(OF))
str(sum)




###QUI INSERIRE ALBERELLI
(val_opt=rpart_split_finder(OF,length(OF)))
(val_opt2=ctree_split_finder(OF,length(OF)))
####

#OF RISCALATO
plot(sort(OF),type="l",ylim=c(sum[1],sum[6]))

##alternativa
plot(sort(OF), col=is_out[order(OF)]+1)
abline(h=val_opt)
sort(val_opt)

##la soglia è max
soglia=55
##taglio va effettuato 
out_previsti=rep(0,n)
out_previsti=ifelse(OF>soglia,1,0)
table(out_previsti)
(toptics=table(out_previsti,is_out))

#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
#auc(roc_dbscan
#Area under the curve 0.4823



(precision=toptics[2,2]/sum(toptics[2,]))
#[1] 0.00006

(recall=toptics[2,2]/sum(toptics[,2]))
#[1] 0.0007






####forest cover



###devo cambiare distnze per togliere gli zeri
#devo inserire albero nel dbscan
#install.packages("pROC")
library(pROC)
library(rpart)
library(rpart.plot)
library(R.matlab)
library(dbscan)
library(partykit)





###codice operativo


rm(list=ls())
###FUNZIONI
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




setwd("C:/Users/federico/Desktop/Progetto_sl")

path="cover.mat"

data=readMat(path)
X=data$X
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))

p=dim(X)[2]; n=dim(X)[1]


### DBSCAN non Norm ---------------------------


#primo k, media tra i tre
k=mean(c(p+1,log(n),2*p))
#secondo k max tra i due 
#k2=max(p+1,log(n))
#3)
#k3=max(2*p,log(n))
a=kNNdist(X,k=k)
head(a)
kNNdistplot(X,k=k)
distanze=as.vector(a)
####inserire qui il metodo di taglio con "Alberelli"
head(distanze)


(valori=rpart_split_finder(distanze,length(distanze)))

#
plot(sort(distanze),type="l")
#

abline(h=valori,col=valori+1)
abline(h=100)
###scelta di buonsenso, 100

eps=100
res=dbscan(X,eps=eps,minPts = k)
indicatore=res$cluster==0
out_previsti=rep(0,n);out_previsti[indicatore]=1
sum(out_previsti)
##outlier reali
sum(is_out)
t3=table(is_out,out_previsti)
t3

# hullplot(X,res)



#Confusion Matrix & AUC:
table(DBSCAN=out_previsti, Actual=is_out)
#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
(precision=t3[2,2]/sum(t3[2,]))
#[1] 0.357844

# auc(roc_dbscan)
#Area under the curve: 0.667

(recall=t3[2,2]/sum(t3[,2]))
#[1] 0.12748




#####DBSCAN scalato ------------
X=scale(X)
a=kNNdist(X,k=k)
head(a)
kNNdistplot(X,k=k)
distanze=as.vector(a)
####inserire qui il metodo di taglio con "Alberelli"
head(distanze)


(valori=rpart_split_finder(distanze,length(distanze)))

#
plot(sort(distanze),type="l")
#
abline(h=1)
abline(h=valori,col=valori+1)
###scelta di buonsenso, 1

eps=1
res=dbscan(X,eps=eps,minPts = k)
indicatore=res$cluster==0
out_previsti=rep(0,n);out_previsti[indicatore]=1
sum(out_previsti)
##outlier reali
sum(is_out)
t3=table(is_out,out_previsti)
t3

# hullplot(X,res)



#Confusion Matrix & AUC:
table(DBSCAN=out_previsti, Actual=is_out)
#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
(precision=t3[2,2]/sum(t3[2,]))
#[1] 0.054969

# auc(roc_dbscan)
#Area under the curve: 0.5249

(recall=t3[2,2]/sum(t3[,2]))
#0.0939



###smtp 

rm(list=ls())
###devo cambiare distnze per togliere gli zeri
#devo inserire albero nel dbscan
#install.packages("pROC")
library(pROC)
library(rpart)
library(rpart.plot)
library(R.matlab)
library(dbscan)
library(partykit)

covtype.data



###codice operativo


rm(list=ls())
###FUNZIONI
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

###DATI ---------------------


setwd("C:/Users/federico/Desktop/Progetto_sl")
library(caret)
path="smtp1.mat"

data=readMat(path)
X=data$X
summary(X)
is_out<-as.vector(data$y)

p=dim(X)[2]; n=dim(X)[1]
summary(X)

### DBSCAN non Norm ---------------------------


#primo k, media tra i tre
k=mean(c(p+1,log(n),2*p))
#secondo k max tra i due 
#k2=max(p+1,log(n))
#3)
#k3=max(2*p,log(n))
a=kNNdist(X,k=k)
head(a)
kNNdistplot(X,k=k)
distanze=as.vector(a)
####inserire qui il metodo di taglio con "Alberelli"
head(distanze)


(valori=rpart_split_finder(distanze,length(distanze)))
(valori2=ctree_split_finder(distanze,length(distanze)))
#
plot(sort(distanze),type="l",ylim=c(0,0.15))
max(distanze)
#

abline(h=valori,col=valori+1)
abline(h=valori2,col=valori2+1)
abline(h=0.08)
###scelta di buonsenso, max
abline(h=40)
eps=0.08
res=dbscan(X,eps=eps,minPts = k)
indicatore=res$cluster==0
out_previsti=rep(0,n);out_previsti[indicatore]=1
sum(out_previsti)
##outlier reali
sum(is_out)
t3=table(is_out,out_previsti)
t3

# hullplot(X,res)



#Confusion Matrix & AUC:
table(DBSCAN=out_previsti, Actual=is_out)
#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan) ##o.824
(precision=t3[2,2]/sum(t3[2,]))
#[1] 0.6667


# auc(roc_dbscan)



(recall=t3[2,2]/sum(t3[,2]))
#[1][1] 0.01111




#####DBSCAN scalato ------------
head(X)
X=scale(X)
head(X)
a=kNNdist(X,k=k)
head(a)
kNNdistplot(X,k=k)
distanze=as.vector(a)
####inserire qui il metodo di taglio con "Alberelli"
head(distanze)


(valori=rpart_split_finder(distanze,length(distanze)))
(valori2=ctree_split_finder(distanze,length(distanze)) )
#
plot(sort(distanze),type="l",ylim=c(0,0.6))
#

abline(h=valori,col=valori+1)
abline(h=valori2)
###scelta di buonsenso, nessuno di questi 
abline(h=0.1)

eps=0.1
res=dbscan(X,eps=eps,minPts = k)
indicatore=res$cluster==0
out_previsti=rep(0,n);out_previsti[indicatore]=1
sum(out_previsti)
##outlier reali
sum(is_out)
t3=table(is_out,out_previsti)
t3

# hullplot(X,res)



#Confusion Matrix & AUC:
table(DBSCAN=out_previsti, Actual=is_out)
#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
(precision=t3[2,2]/sum(t3[2,]))
#Area under the curve:0.8293
#precision 0.7


# auc(roc_dbscan)

(recall=t3[2,2]/sum(t3[,2]))
#00053



##àmammography




rm(list=ls())
###devo cambiare distnze per togliere gli zeri
#devo inserire albero nel dbscan
#install.packages("pROC")
library(pROC)
library(rpart)
library(rpart.plot)
library(R.matlab)
library(dbscan)
library(partykit)





###codice operativo


rm(list=ls())
###FUNZIONI
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

###DATI ---------------------


setwd("C:/Users/federico/Desktop/Progetto_sl")

path="mammography.mat"

data=readMat(path)
X=data$X
summary(X)

is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))

p=dim(X)[2]; n=dim(X)[1]

summary(X)
scale(X)
### DBSCAN non Norm ---------------------------


#primo k, media tra i tre
k=mean(c(p+1,log(n),2*p))
#secondo k max tra i due 
#k2=max(p+1,log(n))
#3)
#k3=max(2*p,log(n))
a=kNNdist(X,k=k)
head(a)
kNNdistplot(X,k=k)
distanze=as.vector(a)
####inserire qui il metodo di taglio con "Alberelli"
head(distanze)


(valori=rpart_split_finder(distanze,length(distanze)))
(valori2=ctree_split_finder(distanze,length(distanze)))
#
plot(sort(distanze),type="l",ylim=c(0,5))
max(distanze)
#

abline(h=valori,col=valori+1)
abline(h=valori2,col=valori2+1)
abline(h=5)
###scelta di buonsenso, nessuna di queste
abline(h=40)
eps=1
res=dbscan(X,eps=eps,minPts = k)
indicatore=res$cluster==0
out_previsti=rep(0,n);out_previsti[indicatore]=1
sum(out_previsti)
##outlier reali
sum(is_out)
t3=table(is_out,out_previsti)
t3

hullplot(X,res)



#Confusion Matrix & AUC:
table(DBSCAN=out_previsti, Actual=is_out)
#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
(precision=t3[2,2]/sum(t3[2,]))
#[1] 0.1653846


# auc(roc_dbscan)
#Area under the curve: 0.5784



(recall=t3[2,2]/sum(t3[,2]))
#[1] 0.3138686


###OF NON scalato --------------
X=data$X


res<-optics(X,eps=20,minPts = k)
a=kNN(X,k)
head(a$id,3)
numerid=a$id
# numeridi_esimo=[i,]
axx<-kNNdist(X,k)
head(res$order,n=15)
plot(res)

#altro modo per estra1lo
distanzerech=res$reachdist
head(distanzerech)
distanzerech[1]=distanzerech[2]
#dist=mean(distanzerech[numerid[1,]])
dist=vector()
lrd=vector()

###modifica per evitare problemi dati uguali (dati ripetuti)
for (i in 1:n){
  minptsvicini=numerid[i,]
  dist[i]=mean(distanzerech[numerid[i,]])
  
  
}
distmagg=dist[dist>0]
valore=min(distmagg)
dist[dist==0]=valore/2
lrd=1/dist
#lrd[i]=1/dist[i]
summary(dist)
min(dist)
hist(dist)
dist
head(lrd,20)
numeratore=vector()
OF=vector()
for (i in 1:n){
  minptsvicini=numerid[i,]
  lrd_numero=lrd[i]
  lrd_minpts=lrd[numerid[i,]]
  numeratore[i]=sum(lrd_minpts/lrd_numero)
  OF[i]=numeratore[i]/k
}
summary(lrd)
lrd[max(lrd)]
sum(is.na(OF))
str(OF)
(sum=summary(OF))
str(sum)




###QUI INSERIRE ALBERELLI
(val_opt=rpart_split_finder(OF,length(OF)))
(val_opt2=ctree_split_finder(OF,length(OF)))
####

#OF RISCALATO
plot(sort(OF),type="l",ylim=c(sum[1],sum[6]))

##alternativa
plot(sort(OF), col=is_out[order(OF)]+1)
abline(h=val_opt)
sort(val_opt)
abline(h=1.15)
##la soglia è nessunodi essi
soglia=1.16
##taglio va effettuato 
out_previsti=rep(0,n)
out_previsti=ifelse(OF>soglia,1,0)
table(out_previsti)
(toptics=table(out_previsti,is_out))

#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
#auc(roc_dbscan
#Area under the curve 0.544



(precision=toptics[2,2]/sum(toptics[2,]))
#[1] 0.1666667

(recall=toptics[2,2]/sum(toptics[,2]))
#[1] 0.1






####annthyroid


##GIA SCALATO


rm(list=ls())
###devo cambiare distnze per togliere gli zeri
#devo inserire albero nel dbscan
#install.packages("pROC")
library(pROC)
library(rpart)
library(rpart.plot)
library(R.matlab)
library(dbscan)
library(partykit)





###codice operativo


rm(list=ls())
###FUNZIONI
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




setwd("C:/Users/federico/Desktop/Progetto_sl")

path="annthyroid.mat"

data=readMat(path)
X=data$X
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))

p=dim(X)[2]; n=dim(X)[1]


### DBSCAN non Norm ---------------------------


#primo k, media tra i tre
k=mean(c(p+1,log(n),2*p))
#secondo k max tra i due 
#k2=max(p+1,log(n))
#3)
#k3=max(2*p,log(n))
a=kNNdist(X,k=k)
head(a)
kNNdistplot(X,k=k)
distanze=as.vector(a)
####inserire qui il metodo di taglio con "Alberelli"
head(distanze)


(valori=rpart_split_finder(distanze,length(distanze)))
#
plot(sort(distanze),type="l")
#

abline(h=valori,col=valori+1)
abline(h=0.04)
###scelta di buonsenso:fatta da me a mano


eps=0.04
res=dbscan(X,eps=eps,minPts = k)
indicatore=res$cluster==0
out_previsti=rep(0,n);out_previsti[indicatore]=1
sum(out_previsti)
##outlier reali
sum(is_out)
t3=table(is_out,out_previsti)
t3

hullplot(X,res)



#Confusion Matrix & AUC:
table(DBSCAN=out_previsti, Actual=is_out)
#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
(precision=t3[2,2]/sum(t3[2,]))
#[1] 0.164794

# auc(roc_dbscan)
#Area under the curve: 0.5708

(recall=t3[2,2]/sum(t3[,2]))
#[1] 0.3636364


###OF NON scalato --------------


res<-optics(X,eps=30,minPts = k)
a=kNN(X,k)
head(a$id,3)
numerid=a$id
# numeridi_esimo=[i,]
axx<-kNNdist(X,k)
head(res$order,n=15)
plot(res)

#altro modo per estra1lo
distanzerech=res$reachdist
head(distanzerech)
distanzerech[1]=distanzerech[2]
#dist=mean(distanzerech[numerid[1,]])
dist=vector()
lrd=vector()

###modifica per evitare problemi dati uguali (dati ripetuti)
for (i in 1:n){
  minptsvicini=numerid[i,]
  dist[i]=mean(distanzerech[numerid[i,]])
  
  
}
distmagg=dist[dist>0]
valore=min(distmagg)
dist[dist==0]=valore/2
lrd=1/dist
#lrd[i]=1/dist[i]
summary(dist)
min(dist)
hist(dist)
dist
head(lrd,20)
numeratore=vector()
OF=vector()
for (i in 1:n){
  minptsvicini=numerid[i,]
  lrd_numero=lrd[i]
  lrd_minpts=lrd[numerid[i,]]
  numeratore[i]=sum(lrd_minpts/lrd_numero)
  OF[i]=numeratore[i]/k
}
summary(lrd)
lrd[max(lrd)]
sum(is.na(OF))
str(OF)
(sum=summary(OF))
str(sum)




###QUI INSERIRE ALBERELLI
(val_opt=rpart_split_finder(OF,length(OF)))
(val_opt2=ctree_split_finder(OF,length(OF)))
####

#OF RISCALATO
plot(sort(OF),type="l",ylim=c(sum[1],sum[6]))

##alternativa
plot(sort(OF), col=is_out[order(OF)]+1)
abline(h=val_opt)
sort(val_opt)
##la soglia è nessuna di queste
soglia=1.1
abline(h=1.32)
##taglio va effettuato 
out_previsti=rep(0,n)
out_previsti=ifelse(OF>soglia,1,0)
table(out_previsti)
(toptics=table(out_previsti,is_out))



#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
#auc(roc_dbscan)
#Area under the curve: 0.6061



(precision=toptics[2,2]/sum(toptics[2,]))
#0.2057716

(recall=toptics[2,2]/sum(toptics[,2]))
#[1] 0.05430712









#####Completo --------


ann_train <- read.table("C:/Users/federico/Desktop/Progetto_sl/ann_train.csv", quote="\"", comment.char="")
ann_test<-read.table("C:/Users/federico/Desktop/Progetto_sl/ann_test.csv", quote="\"", comment.char="")

dati=ann_train
summary(dati)
y=dati[,22]
is_out=ifelse(y>1,0,1)
X=dati[,-22]
dati<-data.frame(cbind(X, is_out))

p=dim(X)[2]; n=dim(X)[1]
summary(X)#one hot encoding

fac<-c(2:16)
for (i in 1:length(fac)){
  dati[,i]<-as.factor(dati[,i])
}
#One hot encoding:
dati<-model.matrix(~.-1,dati[,-n])
X=dati[,1:(ncol(dati)-1)]
p=dim(X)[2]; n=dim(X)[1]



#####RICOMINCIAMO DA CAPO --------
#dbscan 

X=scale(X)


#primo k, media tra i tre
k=mean(c(p+1,log(n),2*p))
k=100
#secondo k max tra i due 
#k2=max(p+1,log(n))
#3)
#k3=max(2*p,log(n))
a=kNNdist(X,k=k)
head(a)
kNNdistplot(X,k=k)
distanze=as.vector(a)
####inserire qui il metodo di taglio con "Alberelli"
head(distanze)


(valori=rpart_split_finder(distanze,length(distanze)))
#
plot(sort(distanze),type="l")
#

abline(h=valori,col=valori+1)
abline(h=1)
###scelta di buonsenso: il prima del massimo

valori
eps=sort(valori)[7]
res=dbscan(X,eps=eps,minPts = k)
indicatore=res$cluster==0
out_previsti=rep(0,n);out_previsti[indicatore]=1
sum(out_previsti)
##outlier reali
sum()
t3=table(is_out,out_previsti)
t3

hullplot(X,res)



#Confusion Matrix & AUC:
table(DBSCAN=out_previsti, Actual=is_out)
#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
(precision=t3[2,2]/sum(t3[2,]))
#[1] 0.03846154

# auc(roc_dbscan)
#Area under the curve: 0.5112

(recall=t3[2,2]/sum(t3[,2]))
#[1] 0.03225806




###optics completo ---------
###OF NON scalato --------------


res<-optics(X,eps=100,minPts = k)
a=kNN(X,k)
head(a$id,3)
numerid=a$id
# numeridi_esimo=[i,]
axx<-kNNdist(X,k)
head(res$order,n=15)
plot(res)

#altro modo per estra1lo
distanzerech=res$reachdist
head(distanzerech)
distanzerech[1]=distanzerech[2]
#dist=mean(distanzerech[numerid[1,]])
dist=vector()
lrd=vector()

###modifica per evitare problemi dati uguali (dati ripetuti)
for (i in 1:n){
  minptsvicini=numerid[i,]
  dist[i]=mean(distanzerech[numerid[i,]])
  
  
}
distmagg=dist[dist>0]
valore=min(distmagg)
dist[dist==0]=valore/2
lrd=1/dist
#lrd[i]=1/dist[i]
summary(dist)
min(dist)
hist(dist)
head(lrd,20)
numeratore=vector()
OF=vector()
for (i in 1:n){
  minptsvicini=numerid[i,]
  lrd_numero=lrd[i]
  lrd_minpts=lrd[numerid[i,]]
  numeratore[i]=sum(lrd_minpts/lrd_numero)
  OF[i]=numeratore[i]/k
}
summary(lrd)
lrd[max(lrd)]
sum(is.na(OF))
str(OF)
(sum=summary(OF))
str(sum)




###QUI INSERIRE ALBERELLI
(val_opt=rpart_split_finder(OF,length(OF)))
(val_opt2=ctree_split_finder(OF,length(OF)))
####

#OF RISCALATO
plot(sort(OF),type="l",ylim=c(sum[1],sum[6]))

##alternativa
plot(sort(OF), col=is_out[order(OF)]+1)
abline(h=val_opt)
abline(h=val_opt2)
sort(val_opt)
##la soglia è il valore piu alto
soglia=max(val_opt)
##taglio va effettuato 
out_previsti=rep(0,n)
out_previsti=ifelse(OF>soglia,1,0)
table(out_previsti)
(toptics=table(out_previsti,is_out))



#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
#auc(roc_dbscan)
#Area under the curve: 0.5007



(precision=toptics[2,2]/sum(toptics[2,]))
#[1] 0.03333333

(recall=toptics[2,2]/sum(toptics[,2]))
#[1] 0.04301075




###pendigits



rm(list=ls())
###devo cambiare distnze per togliere gli zeri
#devo inserire albero nel dbscan
#install.packages("pROC")
library(pROC)
library(rpart)
library(rpart.plot)
library(R.matlab)
library(dbscan)
library(partykit)





###codice operativo


rm(list=ls())
###FUNZIONI
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

###DATI ---------------------


setwd("C:/Users/federico/Desktop/Progetto_sl")
library(caret)
path="pendigits.mat"

data=readMat(path)
X=data$X
summary(X)
is_out<-as.vector(data$y)
dati<-data.frame(cbind(X, is_out))


### DBSCAN non Norm ---------------------------


#primo k, media tra i tre
k=mean(c(p+1,log(n),2*p))
#secondo k max tra i due 
#k2=max(p+1,log(n))
#3)
#k3=max(2*p,log(n))
a=kNNdist(X,k=k)
head(a)
kNNdistplot(X,k=k)
distanze=as.vector(a)
####inserire qui il metodo di taglio con "Alberelli"
head(distanze)


(valori=rpart_split_finder(distanze,length(distanze)))
(valori2=ctree_split_finder(distanze,length(distanze)))
#
plot(sort(distanze),type="l")
max(distanze)
#

abline(h=valori,col=valori+1)
abline(h=valori2,col=valori2+1)
abline(h=0.43)
###scelta di buonsenso, nessuna di queste

eps=0.43
res=dbscan(X,eps=eps,minPts = k)
indicatore=res$cluster==0
out_previsti=rep(0,n);out_previsti[indicatore]=1
sum(out_previsti)
##outlier reali
sum(is_out)
t3=table(is_out,out_previsti)
t3

hullplot(X,res)



#Confusion Matrix & AUC:
table(DBSCAN=out_previsti, Actual=is_out)
#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan) ##0.521

(precision=t3[2,2]/sum(t3[2,]))
#[1] 0.064102


# auc(roc_dbscan)



(recall=t3[2,2]/sum(t3[,2]))
#[1][1] 0.06289308




#####DBSCAN scalato
#gia scalato
###OF NON scalato-------------------
X=data$X


res<-optics(X,eps=200,minPts = k)
a=kNN(X,k)
head(a$id,3)
numerid=a$id
# numeridi_esimo=[i,]
axx<-kNNdist(X,k)
head(res$order,n=15)
plot(res)

#altro modo per estra1lo
distanzerech=res$reachdist
head(distanzerech)
distanzerech[1]=distanzerech[2]
#dist=mean(distanzerech[numerid[1,]])
dist=vector()
lrd=vector()

###modifica per evitare problemi dati uguali (dati ripetuti)
for (i in 1:n){
  minptsvicini=numerid[i,]
  dist[i]=mean(distanzerech[numerid[i,]])
  
  
}
distmagg=dist[dist>0]
valore=min(distmagg)
dist[dist==0]=valore/2
lrd=1/dist
#lrd[i]=1/dist[i]
summary(dist)
min(dist)
hist(dist)
dist
head(lrd,20)
numeratore=vector()
OF=vector()
for (i in 1:n){
  minptsvicini=numerid[i,]
  lrd_numero=lrd[i]
  lrd_minpts=lrd[numerid[i,]]
  numeratore[i]=sum(lrd_minpts/lrd_numero)
  OF[i]=numeratore[i]/k
}
summary(lrd)
lrd[max(lrd)]
sum(is.na(OF))
str(OF)
(sum=summary(OF))
str(sum)




###QUI INSERIRE ALBERELLI
(val_opt=rpart_split_finder(OF,length(OF)))
(val_opt2=ctree_split_finder(OF,length(OF)))
####

#OF RISCALATO
plot(sort(OF),type="l",ylim=c(sum[1],sum[6]))

##alternativa
plot(sort(OF), col=is_out[order(OF)]+1)
abline(h=val_opt)
sort(val_opt)

##la soglia è nessunodi qusti valori
soglia=1.1
##taglio va effettuato 
out_previsti=rep(0,n)
out_previsti=ifelse(OF>soglia,1,0)
table(out_previsti)
(toptics=table(out_previsti,is_out))

#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
#auc(roc_dbscan
#Area under the curve: 0.5244




(precision=toptics[2,2]/sum(toptics[2,]))
#[1] [1] 0.04844291

(recall=toptics[2,2]/sum(toptics[,2]))
#[1] 0.08974359



####ecoli







rm(list=ls())
###devo cambiare distnze per togliere gli zeri
#devo inserire albero nel dbscan
#install.packages("pROC")
library(pROC)
library(rpart)
library(rpart.plot)
library(R.matlab)
library(dbscan)
library(partykit)





###codice operativo


rm(list=ls())
###FUNZIONI
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

###DATI ---------------------


setwd("C:/Users/federico/Desktop/Progetto_sl")
library(caret)
path="ecoli2.csv"

data=read.csv(path, sep=";", header=F)
is_out<-as.vector(data$V8)
X <- data[,-8]
dati<-data.frame(cbind(X, is_out))
head(dati)
summary(dati)
p=dim(X)[2]; n=dim(X)[1]

### DBSCAN non Norm ---------------------------


#primo k, media tra i tre
k=mean(c(p+1,log(n),2*p))
#secondo k max tra i due 
#k2=max(p+1,log(n))
#3)
#k3=max(2*p,log(n))
a=kNNdist(X,k=k)
head(a)
kNNdistplot(X,k=k)
distanze=as.vector(a)
####inserire qui il metodo di taglio con "Alberelli"
head(distanze)


(valori=rpart_split_finder(distanze,length(distanze)))
(valori2=ctree_split_finder(distanze,length(distanze)))
#
plot(sort(distanze),type="l")
max(distanze)
#

abline(h=valori,col=valori+1)
abline(h=valori2,col=valori2+1)
abline(h=5)

###scelta di buonsenso, nostr
abline(h=40)
eps=0.22
res=dbscan(X,eps=eps,minPts = k)
indicatore=res$cluster==0
out_previsti=rep(0,n);out_previsti[indicatore]=1
sum(out_previsti)
##outlier reali
sum(is_out)
t3=table(is_out,out_previsti)
t3

hullplot(X,res)



#Confusion Matrix & AUC:
table(DBSCAN=out_previsti, Actual=is_out)
#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan) ##o.8705
(precision=t3[2,2]/sum(t3[2,]))
#[1] 0.77


# auc(roc_dbscan)



(recall=t3[2,2]/sum(t3[,2]))
#[1][1] 0.3684111




#####DBSCAN scalato ------------
head(X)
X=scale(X)
head(X)
a=kNNdist(X,k=k)
head(a)
kNNdistplot(X,k=k)
distanze=as.vector(a)
####inserire qui il metodo di taglio con "Alberelli"
head(distanze)


(valori=rpart_split_finder(distanze,length(distanze)))
(valori2=ctree_split_finder(distanze,length(distanze)) )
#
plot(sort(distanze),type="l")
#

abline(h=valori,col=valori+1)
abline(h=valori2)
###scelta di buonsenso, nessuno di questi 
abline(h=2)

eps=2
res=dbscan(X,eps=eps,minPts = k)
indicatore=res$cluster==0
out_previsti=rep(0,n);out_previsti[indicatore]=1
sum(out_previsti)
##outlier reali
sum(is_out)
t3=table(is_out,out_previsti)
t3

hullplot(X,res)



#Confusion Matrix & AUC:
table(DBSCAN=out_previsti, Actual=is_out)
#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
(precision=t3[2,2]/sum(t3[2,]))
#Area under the curve:0.4941
#0


# auc(roc_dbscan)

(recall=t3[2,2]/sum(t3[,2]))
#0



###OF NON scalato-------------------
X <- data[,-8]



res<-optics(X,eps=200,minPts = k)
a=kNN(X,k)
head(a$id,3)
numerid=a$id
# numeridi_esimo=[i,]
axx<-kNNdist(X,k)
head(res$order,n=15)
plot(res)

#altro modo per estra1lo
distanzerech=res$reachdist
head(distanzerech)
distanzerech[1]=distanzerech[2]
#dist=mean(distanzerech[numerid[1,]])
dist=vector()
lrd=vector()

###modifica per evitare problemi dati uguali (dati ripetuti)
for (i in 1:n){
  minptsvicini=numerid[i,]
  dist[i]=mean(distanzerech[numerid[i,]])
  
  
}
distmagg=dist[dist>0]
valore=min(distmagg)
dist[dist==0]=valore/2
lrd=1/dist
#lrd[i]=1/dist[i]
summary(dist)
min(dist)
hist(dist)
dist
head(lrd,20)
numeratore=vector()
OF=vector()
for (i in 1:n){
  minptsvicini=numerid[i,]
  lrd_numero=lrd[i]
  lrd_minpts=lrd[numerid[i,]]
  numeratore[i]=sum(lrd_minpts/lrd_numero)
  OF[i]=numeratore[i]/k
}
summary(lrd)
lrd[max(lrd)]
sum(is.na(OF))
str(OF)
(sum=summary(OF))
str(sum)




###QUI INSERIRE ALBERELLI
(val_opt=rpart_split_finder(OF,length(OF)))
(val_opt2=ctree_split_finder(OF,length(OF)))
####

#OF RISCALATO
plot(sort(OF),type="l",ylim=c(sum[1],sum[6]))

##alternativa
plot(sort(OF), col=is_out[order(OF)]+1)
abline(h=val_opt)
sort(val_opt)

##la soglia è nessunodi qusti valori
soglia=1.2
##taglio va effettuato 
out_previsti=rep(0,n)
out_previsti=ifelse(OF>soglia,1,0)
table(out_previsti)
(toptics=table(out_previsti,is_out))

#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
#auc(roc_dbscan
#Area under the curve: 0.8736




(precision=toptics[2,2]/sum(toptics[2,]))
#[1] 0.41176

(recall=toptics[2,2]/sum(toptics[,2]))
#[1] 0.77777






####wine




rm(list=ls())
###devo cambiare distnze per togliere gli zeri
#devo inserire albero nel dbscan
#install.packages("pROC")
library(pROC)
library(rpart)
library(rpart.plot)
library(R.matlab)
library(dbscan)
library(partykit)





###codice operativo


rm(list=ls())
###FUNZIONI
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

###DATI ---------------------


setwd("C:/Users/federico/Desktop/Progetto_sl")
library(caret)
path="wine.mat"

data=readMat(path)
X=data$X
summary(X)
is_out<-as.vector(data$y)

p=dim(X)[2]; n=dim(X)[1]
summary(X)

### DBSCAN non Norm ---------------------------


#primo k, media tra i tre
k=mean(c(p+1,log(n),2*p))
#secondo k max tra i due 
#k2=max(p+1,log(n))
#3)
#k3=max(2*p,log(n))
a=kNNdist(X,k=k)
head(a)
kNNdistplot(X,k=k)
distanze=as.vector(a)
####inserire qui il metodo di taglio con "Alberelli"
head(distanze)


(valori=rpart_split_finder(distanze,length(distanze)))
(valori2=ctree_split_finder(distanze,length(distanze)))
#
plot(sort(distanze),type="l")
max(distanze)
#

abline(h=valori,col=valori+1)
abline(h=valori2,col=valori2+1)
abline(h=5)
###scelta di buonsenso, max
abline(h=40)
eps=100
res=dbscan(X,eps=eps,minPts = k)
indicatore=res$cluster==0
out_previsti=rep(0,n);out_previsti[indicatore]=1
sum(out_previsti)
##outlier reali
sum(is_out)
t3=table(is_out,out_previsti)
t3

hullplot(X,res)



#Confusion Matrix & AUC:
table(DBSCAN=out_previsti, Actual=is_out)
#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan) ##o.4951
(precision=t3[2,2]/sum(t3[2,]))
#[1] 1


# auc(roc_dbscan)=0.9832



(recall=t3[2,2]/sum(t3[,2]))
#[1][1] 0.7142




#####DBSCAN scalato ------------
head(X)
X=scale(X)
head(X)
a=kNNdist(X,k=k)
head(a)
kNNdistplot(X,k=k)
distanze=as.vector(a)
####inserire qui il metodo di taglio con "Alberelli"
head(distanze)


(valori=rpart_split_finder(distanze,length(distanze)))
(valori2=ctree_split_finder(distanze,length(distanze)) )
#
plot(sort(distanze),type="l")
#

abline(h=valori,col=valori+1)
abline(h=valori2)
###scelta di buonsenso, nessuno di questi 
abline(h=10)

eps=3
res=dbscan(X,eps=eps,minPts = k)
indicatore=res$cluster==0
out_previsti=rep(0,n);out_previsti[indicatore]=1
sum(out_previsti)
##outlier reali
sum(is_out)
t3=table(is_out,out_previsti)
t3

hullplot(X,res)



#Confusion Matrix & AUC:
table(DBSCAN=out_previsti, Actual=is_out)
#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
(precision=t3[2,2]/sum(t3[2,]))
#Area under the curve:0.8286
#0.8


# auc(roc_dbscan)

(recall=t3[2,2]/sum(t3[,2]))
#0.32



###OF NON scalato-------------------
X=data$X


res<-optics(X,eps=2000,minPts = k)
a=kNN(X,k)
head(a$id,3)
numerid=a$id
# numeridi_esimo=[i,]
axx<-kNNdist(X,k)
head(res$order,n=15)
plot(res)

#altro modo per estra1lo
distanzerech=res$reachdist
head(distanzerech)
distanzerech[1]=distanzerech[2]
#dist=mean(distanzerech[numerid[1,]])
dist=vector()
lrd=vector()

###modifica per evitare problemi dati uguali (dati ripetuti)
for (i in 1:n){
  minptsvicini=numerid[i,]
  dist[i]=mean(distanzerech[numerid[i,]])
  
  
}
distmagg=dist[dist>0]
valore=min(distmagg)
dist[dist==0]=valore/2
lrd=1/dist
#lrd[i]=1/dist[i]
summary(dist)
min(dist)
hist(dist)
dist
head(lrd,20)
numeratore=vector()
OF=vector()
for (i in 1:n){
  minptsvicini=numerid[i,]
  lrd_numero=lrd[i]
  lrd_minpts=lrd[numerid[i,]]
  numeratore[i]=sum(lrd_minpts/lrd_numero)
  OF[i]=numeratore[i]/k
}
summary(lrd)
lrd[max(lrd)]
sum(is.na(OF))
str(OF)
(sum=summary(OF))
str(sum)




###QUI INSERIRE ALBERELLI
(val_opt=rpart_split_finder(OF,length(OF)))
(val_opt2=ctree_split_finder(OF,length(OF)))
####

#OF RISCALATO
plot(sort(OF),type="l",ylim=c(sum[1],sum[6]))

##alternativa
plot(sort(OF), col=is_out[order(OF)]+1)
abline(h=val_opt)
sort(val_opt)

##la soglia è nessunodi qusti valori
soglia=1.08
##taglio va effettuato 
out_previsti=rep(0,n)
out_previsti=ifelse(OF>soglia,1,0)
table(out_previsti)
(toptics=table(out_previsti,is_out))

#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
#auc(roc_dbscan
#Area under the curve: 0.9244




(precision=toptics[2,2]/sum(toptics[2,]))
#[1] 0.9244

(recall=toptics[2,2]/sum(toptics[,2]))
#[1] 1






###OF SCALATO ------------------
X=scale(X)
res<-optics(X,eps=500,minPts = k)
a=kNN(X,k)
head(a$id,3)
numerid=a$id
# numeridi_esimo=[i,]
axx<-kNNdist(X,k)
head(res$order,n=15)
plot(res)

#altro modo per estra1lo
distanzerech=res$reachdist
head(distanzerech)
distanzerech[1]=distanzerech[2]
#dist=mean(distanzerech[numerid[1,]])
dist=vector()
lrd=vector()

###modifica per evitare problemi dati uguali (dati ripetuti)
for (i in 1:n){
  minptsvicini=numerid[i,]
  dist[i]=mean(distanzerech[numerid[i,]])
  
  
}
distmagg=dist[dist>0]
valore=min(distmagg)
dist[dist==0]=valore/2
lrd=1/dist
#lrd[i]=1/dist[i]
summary(dist)
min(dist)
hist(dist)
dist
head(lrd,20)
numeratore=vector()
OF=vector()
for (i in 1:n){
  minptsvicini=numerid[i,]
  lrd_numero=lrd[i]
  lrd_minpts=lrd[numerid[i,]]
  numeratore[i]=sum(lrd_minpts/lrd_numero)
  OF[i]=numeratore[i]/k
}
summary(lrd)
lrd[max(lrd)]
sum(is.na(OF))
str(OF)
(sum=summary(OF))
str(sum)




###QUI INSERIRE ALBERELLI
(val_opt=rpart_split_finder(OF,length(OF)))
(val_opt2=ctree_split_finder(OF,length(OF)))
####

#OF RISCALATO
plot(sort(OF),type="l",ylim=c(sum[1],sum[6]))

##alternativa
plot(sort(OF), col=is_out[order(OF)]+1)
abline(h=val_opt)
abline(h=val_opt2)
sort(val_opt)
##la soglia è nessun valore di questi
soglia=1
##taglio va effettuato 
out_previsti=rep(0,n)
out_previsti=ifelse(OF>soglia,1,0)
table(out_previsti)
(toptics=table(out_previsti,is_out))



#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
#auc(roc_dbscan)
#Area under the curve: Area under the curve: 0.8666



(precision=toptics[2,2]/sum(toptics[2,]))
#[1] 0.5

(recall=toptics[2,2]/sum(toptics[,2]))
#[1] 0.8





###vertebral



rm(list=ls())
###devo cambiare distnze per togliere gli zeri
#devo inserire albero nel dbscan
#install.packages("pROC")
library(pROC)
library(rpart)
library(rpart.plot)
library(R.matlab)
library(dbscan)
library(partykit)





###codice operativo


rm(list=ls())
###FUNZIONI
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

###DATI ---------------------


setwd("C:/Users/federico/Desktop/Progetto_sl")
library(caret)
path="vertebral.mat"

data=readMat(path)
X=data$X
summary(X)
is_out<-as.vector(data$y)

p=dim(X)[2]; n=dim(X)[1]
summary(X)

### DBSCAN non Norm ---------------------------


#primo k, media tra i tre
k=mean(c(p+1,log(n),2*p))
#secondo k max tra i due 
#k2=max(p+1,log(n))
#3)
#k3=max(2*p,log(n))
a=kNNdist(X,k=k)
head(a)
kNNdistplot(X,k=k)
distanze=as.vector(a)
####inserire qui il metodo di taglio con "Alberelli"
head(distanze)


(valori=rpart_split_finder(distanze,length(distanze)))
(valori2=ctree_split_finder(distanze,length(distanze)))
#
plot(sort(distanze),type="l")
max(distanze)
#

abline(h=valori,col=valori+1)
abline(h=valori2,col=valori2+1)
abline(h=5)
###scelta di buonsenso, max
abline(h=50)
eps=30
res=dbscan(X,eps=eps,minPts = k)
indicatore=res$cluster==0
out_previsti=rep(0,n);out_previsti[indicatore]=1
sum(out_previsti)
##outlier reali
sum(is_out)
t3=table(is_out,out_previsti)
t3

hullplot(X,res)



#Confusion Matrix & AUC:
table(DBSCAN=out_previsti, Actual=is_out)
#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan) ##o.4762
(precision=t3[2,2]/sum(t3[2,]))
#[1] 0


# auc(roc_dbscan)



(recall=t3[2,2]/sum(t3[,2]))
#[1][1] 0




#####DBSCAN scalato ------------
head(X)
X=scale(X)
head(X)
a=kNNdist(X,k=k)
head(a)
kNNdistplot(X,k=k)
distanze=as.vector(a)
####inserire qui il metodo di taglio con "Alberelli"
head(distanze)


(valori=rpart_split_finder(distanze,length(distanze)))
(valori2=ctree_split_finder(distanze,length(distanze)) )
#
plot(sort(distanze),type="l")
#

abline(h=valori,col=valori+1)
abline(h=valori2)
###scelta di buonsenso, nessuno di questi 
abline(h=3)

eps=2
res=dbscan(X,eps=eps,minPts = k)
indicatore=res$cluster==0
out_previsti=rep(0,n);out_previsti[indicatore]=1
sum(out_previsti)
##outlier reali
sum(is_out)
t3=table(is_out,out_previsti)
t3

hullplot(X,res)



#Confusion Matrix & AUC:
table(DBSCAN=out_previsti, Actual=is_out)
#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
(precision=t3[2,2]/sum(t3[2,]))
#Area under the curve:0.4857
#0


# auc(roc_dbscan)

(recall=t3[2,2]/sum(t3[,2]))
#0



###OF NON scalato-------------------
X=data$X


res<-optics(X,eps=2000,minPts = k)
a=kNN(X,k)
head(a$id,3)
numerid=a$id
# numeridi_esimo=[i,]
axx<-kNNdist(X,k)
head(res$order,n=15)
plot(res)

#altro modo per estra1lo
distanzerech=res$reachdist
head(distanzerech)
distanzerech[1]=distanzerech[2]
#dist=mean(distanzerech[numerid[1,]])
dist=vector()
lrd=vector()

###modifica per evitare problemi dati uguali (dati ripetuti)
for (i in 1:n){
  minptsvicini=numerid[i,]
  dist[i]=mean(distanzerech[numerid[i,]])
  
  
}
distmagg=dist[dist>0]
valore=min(distmagg)
dist[dist==0]=valore/2
lrd=1/dist
#lrd[i]=1/dist[i]
summary(dist)
min(dist)
hist(dist)
dist
head(lrd,20)
numeratore=vector()
OF=vector()
for (i in 1:n){
  minptsvicini=numerid[i,]
  lrd_numero=lrd[i]
  lrd_minpts=lrd[numerid[i,]]
  numeratore[i]=sum(lrd_minpts/lrd_numero)
  OF[i]=numeratore[i]/k
}
summary(lrd)
lrd[max(lrd)]
sum(is.na(OF))
str(OF)
(sum=summary(OF))
str(sum)




###QUI INSERIRE ALBERELLI
(val_opt=rpart_split_finder(OF,length(OF)))
(val_opt2=ctree_split_finder(OF,length(OF)))
####

#OF RISCALATO
plot(sort(OF),type="l",ylim=c(sum[1],sum[6]))

##alternativa
plot(sort(OF), col=is_out[order(OF)]+1)
abline(h=val_opt)
sort(val_opt)

##la soglia è nessunodi qusti valori
soglia=1.08
##taglio va effettuato 
out_previsti=rep(0,n)
out_previsti=ifelse(OF>soglia,1,0)
table(out_previsti)
(toptics=table(out_previsti,is_out))

#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
#auc(roc_dbscan
#Area under the curve: 0.4905




(precision=toptics[2,2]/sum(toptics[2,]))
#[1] 0.1111111111

(recall=toptics[2,2]/sum(toptics[,2]))
#[1] 0.133333333333






###OF SCALATO ------------------
X=scale(X)
res<-optics(X,eps=500,minPts = k)
a=kNN(X,k)
head(a$id,3)
numerid=a$id
# numeridi_esimo=[i,]
axx<-kNNdist(X,k)
head(res$order,n=15)
plot(res)

#altro modo per estra1lo
distanzerech=res$reachdist
head(distanzerech)
distanzerech[1]=distanzerech[2]
#dist=mean(distanzerech[numerid[1,]])
dist=vector()
lrd=vector()

###modifica per evitare problemi dati uguali (dati ripetuti)
for (i in 1:n){
  minptsvicini=numerid[i,]
  dist[i]=mean(distanzerech[numerid[i,]])
  
  
}
distmagg=dist[dist>0]
valore=min(distmagg)
dist[dist==0]=valore/2
lrd=1/dist
#lrd[i]=1/dist[i]
summary(dist)
min(dist)
hist(dist)
dist
head(lrd,20)
numeratore=vector()
OF=vector()
for (i in 1:n){
  minptsvicini=numerid[i,]
  lrd_numero=lrd[i]
  lrd_minpts=lrd[numerid[i,]]
  numeratore[i]=sum(lrd_minpts/lrd_numero)
  OF[i]=numeratore[i]/k
}
summary(lrd)
lrd[max(lrd)]
sum(is.na(OF))
str(OF)
(sum=summary(OF))
str(sum)




###QUI INSERIRE ALBERELLI
(val_opt=rpart_split_finder(OF,length(OF)))
(val_opt2=ctree_split_finder(OF,length(OF)))
####

#OF RISCALATO
plot(sort(OF),type="l",ylim=c(sum[1],sum[6]))

##alternativa
plot(sort(OF), col=is_out[order(OF)]+1)
abline(h=val_opt)
abline(h=val_opt2)
sort(val_opt)
##la soglia è nessun valore di questi
soglia=1.08
##taglio va effettuato 
out_previsti=rep(0,n)
out_previsti=ifelse(OF>soglia,1,0)
table(out_previsti)
(toptics=table(out_previsti,is_out))



#Valutazione performance:
roc_dbscan<- roc(response = is_out, predictor = out_previsti)
auc(roc_dbscan)
#auc(roc_dbscan)
#Area under the curve: Area under the curve: 0.4857



(precision=toptics[2,2]/sum(toptics[2,]))
#[1] 0.1052632

(recall=toptics[2,2]/sum(toptics[,2]))
#[1] 0.13333333







