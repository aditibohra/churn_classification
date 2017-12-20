setwd("G:/Practice/prasad")
tel<-read.csv("TelcoChurn.csv")
str(tel)
head(tel)
View(tel)
tel$MultipleLines<-ifelse(grepl('No',tel$MultipleLines),'No',
                          ifelse(grepl('Yes',tel$MultipleLines),'Yes','Nothing'))
tel$MultipleLines
str(tel)
tel$MultipleLines<-as.factor(tel$MultipleLines)
#online  security reducing the extra words 
tel$OnlineSecurity<-ifelse(grepl('No',tel$OnlineSecurity),'No',
                          ifelse(grepl('Yes',tel$OnlineSecurity),'Yes','Nothing'))

tel$OnlineBackup<-ifelse(grepl('No',tel$OnlineBackup),'No',
                         ifelse(grepl('Yes',tel$OnlineBackup),'Yes','Nothing'))

tel$DeviceProtection<-ifelse(grepl('No',tel$DeviceProtection),'No',
                         ifelse(grepl('Yes',tel$DeviceProtection),'Yes','Nothing'))
tel$TechSupport<-ifelse(grepl('No',tel$TechSupport),'No',
                        ifelse(grepl('Yes',tel$TechSupport),'Yes','Nothing'))
tel$StreamingTV<-ifelse(grepl('No',tel$StreamingTV),'No',
                        ifelse(grepl('Yes',tel$StreamingTV),'Yes','Nothing'))
tel$StreamingMovies<-ifelse(grepl('No',tel$StreamingMovies),'No',
                            ifelse(grepl('Yes',tel$StreamingMovies),'Yes','Nothing'))
str(tel)
tel$OnlineSecurity<-as.factor(tel$OnlineSecurity)
tel$OnlineBackup<-as.factor(tel$OnlineBackup)
tel$DeviceProtection<-as.factor(tel$DeviceProtection)
tel$TechSupport<-as.factor(tel$TechSupport)
tel$StreamingTV<-as.factor(tel$StreamingTV)
tel$StreamingMovies<-as.factor(tel$StreamingMovies)
str(tel)
#3precrossing the data
sum(is.na(tel))
impute<-centralImputation(tel)
sum(is.na(impute))
impute$customerID<-NULL
impute$Churn<-as.numeric(tel$Churn)
dummy<-dummyVars("~.",data=impute,fullRank = F)
head(dummy)
teldum<-as.data.frame(predict(dummy,impute))
head(teldum)
teldum$Churn<-as.factor(teldum$Churn)
length(teldum)
tsnd<-decostand(teldum[,-39],"range")
head(tsnd)
teldata<-cbind(tsnd,Churn=teldum$Churn)
head(teldata)
teldata$Churn<-as.factor(teldata$Churn)
str(teldata)
#spliting the data 
set.seed(102)
intrain<-sample(1:nrow(teldata),nrow(teldata)*0.7)
tel_train<-teldata[intrain,]
tel_test<-teldata[-intrain,]
str(tel_test)
#running xgboost model 
tel_train$Churn=as.factor(tel_train$Churn)
labels <- tel_train$Churn 
ts_label <-tel_test$Churn
new_tr <- model.matrix(~.+0,data = tel_train[,-39]) 
new_ts <- model.matrix(~.+0,data = tel_test[,-39])
#converting to numeric type
labels <- as.numeric(labels)-1
ts_label <- as.numeric(ts_label)-1
##preparing matrix 
dtrain1<- xgb.DMatrix(data = new_tr,label = labels) 
dtest1 <- xgb.DMatrix(data = new_ts,label=ts_label)
str(dtrain1)
#default para metrr 
params <- list(booster = "gbtree", objective = "binary:logistic", eta=0.3,
               gamma=0, max_depth=6, min_child_weight=1, subsample=1, colsample_bytree=1)

#building model using this parameters 
xgbcv <- xgb.cv( params = params, data = dtrain1, nrounds = 100, 
                 nfold = 5, showsd = T, stratified = T, print.every_n = 10, 
                 maximize = F)
##best iteration = 31
#min(xgbcv$test.error.mean)

###3first traing model 
xgb1 <- xgb.train (params = params, data = dtrain1, nrounds = 60, 
                   watchlist = list(val=dtest,train=dtrain), print.every_n = 10, 
                   early_stop_round = 10, maximize = F , eval_metric = "error")


####prediction metric
xgbpred <- predict (xgb1,dtrain1)
xgbpred <- ifelse (xgbpred > 0.5,1,0)
trainacc<-confusionMatrix (xgbpred, labels)
trainacc
# test acc
xgbpred <- predict (xgb1,dtest1)
xgbpred <- ifelse (xgbpred > 0.3,1,0)
testacc<-confusionMatrix (xgbpred, ts_label)
testacc
#rocr curve 
ROCRpred = prediction(xgbpred, ts_label)
as.numeric(performance(ROCRpred, "auc")@y.values)
ROCRperf <- performance(ROCRpred, "tpr", "fpr")
par(mfrow=c(1,1))
plot(ROCRperf, colorize = TRUE, print.cutoffs.at=seq(0,1,by=0.1), text.adj=c(-0.2,1.7))


#feature selection 
mat <- xgb.importance (feature_names = colnames(new_tr),model = xgb1)
xgb.plot.importance (importance_matrix = mat[1:20]) 


dtrain<-subset(tel_train,select=c("Contract.Month-to-month","TotalCharges","MonthlyCharges","tenure",
                                  "InternetService.Fiber optic","PaymentMethod.Electronic check","Churn"))

head(dtrain)
dtest<-subset(tel_test,select=c("Contract.Month-to-month","TotalCharges","MonthlyCharges","tenure",
                                  "InternetService.Fiber optic","PaymentMethod.Electronic check","Churn"))

str(new_tr)
new_tr1 <- model.matrix(~.+0,data = dtrain[,-39]) 
new_ts1 <- model.matrix(~.+0,data = dtest[,-39])
labels1 <- dtrain$Churn 
ts_label1 <-dtest$Churn
labels1 <- as.numeric(labels1)-1
ts_label1 <- as.numeric(ts_label1)-1
dtrain1 <- xgb.DMatrix(data = new_tr1,label = labels1) 
dtest1 <- xgb.DMatrix(data = new_ts1,label=ts_label1)
#bbuilding a model
xgbcv1 <- xgb.cv( params = params, data =dtrain1, nrounds = 100, 
                 nfold = 5, showsd = T, stratified = T, print.every_n = 10, 
                 maximize = F)

xgb2 <- xgb.train (params = params, data = dtrain1, nrounds = 60, 
                   watchlist = list(val=dtest1,train=dtrain1), print.every_n = 10, 
                   early_stop_round = 10, maximize = F , eval_metric = "error")
#train data 100 % acc 
xgbpred1 <- predict (xgb2,dtrain1)
xgbpred1 <- ifelse (xgbpred1 > 0.5,1,0)
trainacc1<-confusionMatrix (xgbpred1, labels1)
trainacc1
# test acc 100 accuracy
xgbpred1 <- predict (xgb2,dtest1)
xgbpred1 <- ifelse (xgbpred1 > 0.5,1,0)
testacc1<-confusionMatrix (xgbpred1, ts_label1)
testacc1
#################################################
ROCRpred = prediction(xgbpred1, ts_label1)
as.numeric(performance(ROCRpred, "auc")@y.values)
ROCRperf <- performance(ROCRpred, "tpr", "fpr")
par(mfrow=c(1,1))
plot(ROCRperf, colorize = TRUE, print.cutoffs.at=seq(0,1,by=0.1), text.adj=c(-0.2,1.7))


##################################################
