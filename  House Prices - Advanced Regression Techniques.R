rm(list=ls()) #remove all objects
dev.off() #clear all plots
library('plyr')
library(reshape2)
library("ggcorrplot")
library(caret)
library(Hmisc)
library(ParamHelpers)
library(gbm)
library(pls)
library(leaps)
library(glmnet)


df<-read.csv('/Users/Work/Desktop/Work/Projects/Kaggle/House Prices-Advanced Regression Techniques /train.csv')
df_test<-read.csv('/Users/Work/Desktop/Work/Projects/Kaggle/House Prices-Advanced Regression Techniques /test.csv')
t(colSums(is.na(df))) #missing values check

str(df) #81 vars, 1460 obs, factors and integers; vars were made into integers or factors
ordinal    <- c('BsmtQual','BsmtCond','BsmtExposure','HeatingQC','KitchenQual','GarageQual','GarageCond',
                'OverallQual','OverallCond','ExterQual','ExterCond')
nominal    <- c('MSSubClass','MSZoning','LotShape','LandContour','Utilities','LotConfig','LandSlope',
                'Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl',
                'Heating','Exterior1st','Exterior2nd','MasVnrType','Foundation','Electrical','Functional',
                'GarageType','GarageFinish','PavedDrive','SaleType','SaleCondition','BsmtFinType2','BsmtFinType1',
                'Street','CentralAir')
discrete   <- c('BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr',
                'KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageCars')
continuous <- c('LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','TotalBsmtSF','X1stFlrSF',
                'X2ndFlrSF','LowQualFinSF','GrLivArea','GarageArea','WoodDeckSF','OpenPorchSF',
                'EnclosedPorch','X3SsnPorch','ScreenPorch','PoolArea','MiscVal')
date       <- c('YearBuilt','YearRemodAdd','GarageYrBlt','MoSold','YrSold')

df_ordinal    <- df[ordinal]
df_nominal    <- df[nominal]
df_discrete   <- df[discrete]
df_continuous <- df[continuous]
df_date       <- df[date]

summary(df)

#Label Encode Functions for Ordinal Predictors
#ifelse(condition, if true, if false) -> ifelse(condition, if true, ifelse(condition, if true, ifelse))
#I tried using IF, ELSE IF, and ELSE, but it generated an error in LAPPLY.
label_encode <- function(x) {
  ifelse(x=='Ex',5,
         ifelse(x=='Gd',4,
                ifelse(x=='TA',3,
                       ifelse(x=='Fa',2,
                              ifelse(x=='Po',1,
                                     ifelse(x=='NA',0,
                                            ifelse(x=='nan','nan','ERROR'
                                            )))))))
}

bsmtexposure_encode <- function(x) {
  ifelse(x=='Gd',4,
         ifelse(x=='Av',3,
                ifelse(x=='Mn',2,
                       ifelse(x=='No',1,
                              ifelse(x=='NA',0,
                                     ifelse(x=='nan','nan','ERROR')
                                            )))))
}



#Create Label Encoded Dataframe for Ordinal Predictors
df_ordinal_encoded <- df_ordinal
for (j in 1:ncol(df_ordinal)) {
  if (colnames(df_ordinal)[j] == 'OverallQual' |
      colnames(df_ordinal)[j] == 'OverallCond') {
      df_ordinal_encoded[j] <- df_ordinal[j]}
    else if (colnames(df_ordinal)[j] != 'BsmtExposure') {
      df_ordinal_encoded[j] <- lapply(df_ordinal[j], label_encode)
    } else if (colnames(df_ordinal)[j] == 'BsmtExposure') {
      df_ordinal_encoded[j] <- lapply(df_ordinal[j], bsmtexposure_encode )
    }}

'''
#apply applies an aggregate function to each df row (2nd arg = 1) or column (2nd arg =2)
lapply applies a function to each list item and returns a list of items back 
#each item prints to a separate line
sapply applies a function to each list item and returns a vector of items back
#each item prints to the same line (in a returned array)
#vapply applies a function to each list item and returns a vector of items back
whose datatype and length is specified in vapply
'''

#Frequencies of label encoded ordinal predictors
for (j in 1:ncol(df_ordinal)) {
  print(count(df_ordinal_encoded,ordinal[j]))
}

#Correlations of Numeric and Ordinal Variables
#Convert dataframe values to numeric for cor() function
#R numeric vars can have NA's (so you can convert a chr var to a numeric even if it has NA's)
df_numeric_ordinal_encoded <- data.frame(df_discrete,df_continuous,df_ordinal_encoded)
df_numeric_ordinal_encoded2<-data.frame(lapply(df_numeric_ordinal_encoded, function(x) as.numeric(x)))
count(df_numeric_ordinal_encoded2,'GarageQual') 

#Generate correlation map
cormat<-cor(df_numeric_ordinal_encoded2,df_numeric_ordinal_encoded2, method = 'spearman', use='pairwise')
upper_tri <- get_upper_tri(cormat)
melted_cormat <- melt(upper_tri, na.rm = TRUE)
head(melted_cormat)
ggcorrplot(cormat, hc.order = TRUE, type = "lower", outline.col = "white",
           tl.cex = 6) # make variable font size smaller

#Inspect "correlation" of Nominal Predictors with Sale Price
for (j in 1:ncol(df_nominal)) {
  print(aggregate(df$SalePrice, by=df_nominal[j], FUN=mean, na.rm=TRUE))
}

#Histogram of numeric and ordinal predictor values
dim(df_numeric_ordinal_encoded2)
par(mfrow=c(5,8),mai = c(.13, .17, .13, 0.1), cex=.3)
#mai(bottom, left, top, right) <- controls margin spacing around each plot in inches
for (j in colnames(df_numeric_ordinal_encoded2)){
  hist(df_numeric_ordinal_encoded2[[j]], main=j)
}

#One-Hot Encode the Nominal Data
df_nominal$MSSubClass<-as.character(df_nominal$MSSubClass)
df_nominal_dummies <- dummyVars(" ~ .", data = df_nominal)
df_nominal_dummies <- data.frame(predict(df_nominal_dummies, newdata = df_nominal))

#Create final dataframe and rename response column to its proper name
df_final <- cbind(df_nominal_dummies,df_numeric_ordinal_encoded2,df_date, data.frame(df$SalePrice))
colnames(df_final)[colnames(df_final)=="df.SalePrice"] <- "SalePrice"

#There are still missing values that need to be addressed (AFTER TRAIN / TEST SPLITTING)
t(colSums(is.na(df_nominal_dummies)))
t(colSums(is.na(df_numeric_ordinal_encoded2)))
t(colSums(is.na(df_date)))

# Linear Regression (via 5-Fold CV)
set.seed(1)
control <- trainControl(method="cv", number=5) #specifies cross validation method!
model_lr <- train(SalePrice~., data=df_final, method="lm",  #train hyperparameter tunes and gives a model performance estimate via CV
               preProcess=c("medianImpute"), #"center","scale", -> (subtracts mean, divides by sd)
               trControl=control, tuneLength=5,na.action=na.pass) #na.action=na.omit -> omits rows w/ missings
names(getModelInfo()) #view which methods can be passed to "method"
coef(model_lr$finalModel) #some variables have NA because they are linearly related to other variables (exact collinearity)
model_lr

# Random Forest (this code takes time to run)
set.seed(1)
model_rf <- train(SalePrice~., data=df_final, method="rf",  #train hyperparameter tunes and gives a model performance estimate via CV
                  preProcess=c("medianImpute"), #"center","scale", -> (subtracts mean, divides by sd)
                  trControl=control, tuneLength=5,na.action=na.pass) #na.action=na.omit -> omits rows w/ missings
model_rf


#KNN
set.seed(1)
grid = expand.grid(k = c(3, 9, 12)) #specify hyperparameter values
model_knn <- train(SalePrice~., data=df_final, method="knn",  #train hyperparameter tunes and gives a model performance estimate via CV
                  preProcess=c("medianImpute"), #"center","scale", -> (subtracts mean, divides by sd)
                  trControl=control, tuneLength=5,na.action=na.pass,
                  tunegrid=grid) 
model_knn


#GBM
#these parameters produce an error so are not used 
grid1 <- expand.grid(
  shrinkage = c(.01, .05, .1),
  interaction.depth = c(3, 5, 7),
  n.minobsinnode = c(5, 7, 10),
  bag.fraction = c(.65, .8, 1), 
  optimal_trees = 0,               # a place to dump results
  min_RMSE = 0                     # a place to dump results
)

grid2=expand.grid(
  .n.trees = c(100),
  .interaction.depth = c(11,12),
  .shrinkage = c(0.001),
  .n.minobsinnode = c(10)
)

#GBM without hyperparameter tuning produced a better result, so better hyperparameter values could likely be supplied
set.seed(1)
model_gbm <- train(SalePrice~., data=df_final, method="gbm",  #train hyperparameter tunes and gives a model performance estimate via CV
                   preProcess=c("medianImpute"), #"center","scale", -> (subtracts mean, divides by sd)
                   trControl=control, tuneLength=5,na.action=na.pass) 

model_gbm2 <- train(SalePrice~., data=df_final, method="gbm",  #train hyperparameter tunes and gives a model performance estimate via CV
                    preProcess=c("medianImpute"), #"center","scale", -> (subtracts mean, divides by sd)
                    trControl=control, tuneLength=5,na.action=na.pass,tuneGrid=grid2) 

#Principal components regression
set.seed(1)
model_pcr <- train(SalePrice~., data=df_final, method="pcr",  #train hyperparameter tunes and gives a model performance estimate via CV
                  preProcess=c("medianImpute","center","scale"), #"center","scale", -> (subtracts mean, divides by sd)
                  trControl=control, tuneLength=5,na.action=na.pass) #na.action=na.omit -> omits rows w/ missings
model_pcr #auto-selects 4 principal components as the final model based on K-Fold CV RMSE result
model_pcr$results  #automatically shows results for just the first 5 components
summary(model_pcr)  #automatically shows explained variance for first 4 components

names(getModelInfo()) #get model names that can be used in train() function
getModelInfo("pcr")   #check which model is actually associated with the model name (scroll to top)

#Principal components regression NOT using train method
#Data needs to be median imputed
model_pcr2=pcr(SalePrice~., data=df_final ,scale=TRUE, validation ="CV") #10-fold CV



#Partial Least Squares Regression
set.seed(1)
control <- trainControl(method="cv", number=5) #specifies cross validation method!
model_pls <- train(SalePrice~., data=df_final, method="pls",  #train hyperparameter tunes and gives a model performance estimate via CV
                   preProcess=c("medianImpute","center","scale"), #"center","scale", -> (subtracts mean, divides by sd)
                   trControl=control, tuneLength=5,na.action=na.pass) #na.action=na.omit -> omits rows w/ missings
model_pls #selects 3 components based on 5-fold CV RMSE
summary(model_pls) #shows % of variance explained for # of comps as selected above
names(getModelInfo()) #get model names that can be used in train() function
getModelInfo("pls")   #check which model is actually associated with the model name (scroll to top)

#Partial Least Squares Regression NOT using the train method
model_pls2=plsr(SalePrice~.,data=df_final, scale=TRUE, validation = "CV") #subset=train (to use only some observations)
summary(model_pls2)
pls.pred=predict(model_pls2,x[test,],ncomp=2)
mean((pls.pred-y.test)^2)
warnings()

#Using methods outside the train function
#Splitting dataframe into train and test splits (DID NOT USE THIS!)
n <- nrow(df_final)
df_final_shuffled <- df_final[sample(n), ]
train_indices <- 1:round(0.6 * n)
df_final_train <- df_final_shuffled[train_indices, ]
test_indices <- (round(0.6 * n) + 1):n
df_final_test <- df_final_shuffled[test_indices, ]


#Median Imputation of train/test splits before using non-train() methods
temp            <- preProcess(df_final_train, method=c("medianImpute"))
df_final_train2 <- predict(temp, df_final_train) #apply preproc object to train split
df_final_test2  <- predict(temp, df_final_test)  #apply preproc object to test (using train medians)
t(colSums(is.na(df_final_test2)))                #imputed test has no missings
t(colSums(is.na(df_final_train2)))               #imputed train has no missings
#na.omit() removes rows that have any missing value (this could have been done before splitting)



#Forward Stepwise Selection
#nvmax specifies max # of variable models to try; n seems to result in (n+1) variable models
#nvmax=max_vars yields 31 models, where the nth model has the top n predictors
set.seed(1)
temp<-df_final_train2[,which((names(df_final_train2) %in% c(continuous,discrete,ordinal,date, 'SalePrice')))]
model_forward  <- regsubsets(SalePrice~.,temp,nvmax=max_vars,method='forward') 
model_backward <- regsubsets(SalePrice~.,temp,nvmax=max_vars,method='backward')
model_forward.summary<-summary(model_forward)
model_forward.summary<-summary(model_backward)
model_forward.summary$rsq #train rsq increases as # of variables increase

#Choosing the best model: selecting the number of variables in Forward Selection
dev.off() #shuts down the current visualization in the plot window
plot(model_forward.summary$adjr2,xlab="Number of Variables ", ylab="Adjusted RSq",type="l")
which.max(model_forward.summary$adjr2) #27
points(27,model_forward.summary$adjr2[27], col="red",cex=2,pch=20) #max point indicaed with a dot
plot(model_forward.summary$cp ,xlab="Number of Variables ",ylab="Cp", type='l')
which.min(model_forward.summary$cp ) #26 var model optimizes CP
points(26,model_forward.summary$cp[26],col="red",cex=2,pch=20) 
which.min(model_forward.summary$bic )
plot(model_forward.summary$bic ,xlab="Number of Variables ",ylab="BIC",type='l')
points(18,model_forward.summary$bic[18],col="red",cex=2,pch=20) #18 vars optimizes BIC

#regsubsets() has a built in plot() function which visualizes included vars in each model
#Each row is for an n-variable model.  Included variables have their squares shaded
par(mfrow=c(1,1),mai = c(.6, .6, .13, 0.1), cex=.6) 
plot(model_forward,scale="r2")
dev.off()
par(mfrow=c(1,1),mai = c(.6, .6, .13, 0.1), cex=.6) 
plot(model_forward,scale="adjr2")
dev.off()
par(mfrow=c(1,1),mai = c(.6, .6, .13, 0.1), cex=.6) 
plot(model_forward,scale="Cp")
dev.off()
par(mfrow=c(1,1),mai = c(.6, .6, .13, 0.1), cex=.6) 
plot(model_forward,scale="bic")

#Select the best model using 10-fold cross validation (THE BETTER METHOD!)
k=5         #number of folds in k-fold cross validation
max_vars=40 #max number of vars to consider including
set.seed(1)
folds=sample(1:k,nrow(df_final),replace=TRUE)
cv.errors=matrix(NA,k,max_vars, dimnames=list(NULL, paste(1:max_vars)))
df_final_cont<-df_final[,which((names(df_final) %in% c(continuous,discrete,ordinal,date, 'SalePrice')))]

predict.regsubsets = function (object ,newdata ,id ,...) {
  form=as.formula(object$call [[2]])
  mat=model.matrix(form,newdata)
  coefi=coef(object ,id=id)
  xvars=names(coefi)
  mat[,xvars]%*%coefi}

#Build Forward Selection Models to produce test rmses for 1-40 variables
#The method failed (when I used all variables --> double check this)
#for each of k=10 test folds, median impute on the 9 other train folds and train model off them
#this trained model will produce max_vars models (each with n vars, the optimal vars)
#each of these max_vars models is tested on the test split: 
#the resultant cv rmse matrix contains: for each of the max_vars var models, its results for all 10
#folds. To get the avg rmse across 10 folds per n-var model, avg its column of rmses'
for(j in 1:k){ 
  preproc <- preProcess(df_final_cont[folds!=j,], method=c("medianImpute"))
  train_folds_imputed <- predict(preproc, df_final_cont[folds!=j,]) #apply preproc object to train split
  model<-regsubsets(SalePrice~.,data=train_folds_imputed,nvmax=max_vars,method='forward')
  for (i in 1:max_vars){
    test_fold_imputed <- predict(preproc, df_final_cont[folds==j,])
    pred=predict.regsubsets(model,test_fold_imputed,id=i)
    cv.errors[j,i]=mean((df_final_cont$SalePrice[folds==j]-pred)^2)
  }
}

#Plot Avg CV Score for each N-Var Model generated from Forward Selection
mean.cv.errors=apply(cv.errors ,2,mean)
par(mfrow=c(1,1),mai = c(.6, .6, .13, 0.1), cex=.7) 
plot(mean.cv.errors ,type='b')
sqrt(mean.cv.errors)
which(mean.cv.errors==min(mean.cv.errors)) #the more variables, the better! using all 40 produces best result
forward_selection_rmse<-sqrt(min(mean.cv.errors))

#Ridge Regression
#Ridge regression is good when there are many predictors since it limits the effects of 
#predictors on the model by constraining their coefficients
grid=10^seq(3, -2, by = -.1) #lambda values
model_ridge=glmnet(x,y,alpha=0,lambda=grid,standardize=TRUE) 
#alpha = 0 for ridge; alpha = 1 for lasso; standardize is true by default (glmnet standardizes)

#Using 5-fold cross validation (with median imputation)
df_final_x<-df_final[,-which((names(df_final) %in% 'SalePrice'))] #drop saleprice
df_final_y<-df_final[,which((names(df_final) %in% 'SalePrice'))]  #keep saleprice

k=5 #number of folds in k-fold cross validation
set.seed(1)
folds=sample(1:k,nrow(df_final),replace=TRUE)
model_count<-length(grid)
cv.errors=matrix(NA,k,model_count, dimnames=list(NULL, paste(1:model_count)))
cv.errors.train=matrix(NA,k,model_count, dimnames=list(NULL, paste(1:model_count)))

for(j in 1:k){  
  preproc <- preProcess(df_final_x[folds!=j,], method=c("medianImpute"))
  x_train_folds <- data.matrix(predict(preproc, df_final_x[folds!=j,])) #need matrix of x values
  y_train_folds <- as.vector(df_final_y[folds!=j])
  model_ridge=glmnet(x_train_folds,y_train_folds,alpha=0,lambda=grid,standardize=TRUE) 
  for (i in 1:model_count){
    x_test_folds <- data.matrix(predict(preproc, df_final_x[folds==j,]))
    pred=predict(model_ridge,s=grid[i],newx=x_test_folds)
    pred_train=predict(model_ridge,s=grid[i],newx=x_train_folds)
    cv.errors[j,i]=mean((df_final_y[folds==j]-pred)^2) #row j is the fold, col i is the model for lambda i
    cv.errors.train[j,i]=mean((df_final_y[folds!=j]-pred_train)^2)
  }
}

#Ridge Regression - Test Avg CV RMSE Scores for each lambda 
mean.cv.errors=apply(cv.errors ,2,mean)
par(mfrow=c(1,1),mai = c(.6, .6, .13, 0.1), cex=.7) 
plot(mean.cv.errors ,type='b')
sqrt(mean.cv.errors)
which(mean.cv.errors==min(mean.cv.errors))
ridge_regression_rmse<-sqrt(min(mean.cv.errors))
sqrt(mean.cv.errors) #as lambda gets smaller, the test error gets worse

#Ridge Regression - Train Avg CV RMSE Scores for each lambda
mean.cv.errors.train=apply(cv.errors.train ,2,mean)
par(mfrow=c(1,1),mai = c(.6, .6, .13, 0.1), cex=.7) 
plot(mean.cv.errors.train ,type='b')
sqrt(mean.cv.errors.train)
which(mean.cv.errors.train==min(mean.cv.errors.train))
ridge_regression_train_rmse<-sqrt(min(mean.cv.errors.train))
sqrt(mean.cv.errors.train) #as lambda gets smaller, the test error gets worse

'Ridge Train scores get worse as lambda increases (as the R L2 Norm penalty increases)
 Ridge Test scores get better as lambda increases
 On the train set, the less we penalize over-fitting, the better the fit
 On the test set,  the less we penalized over-fitting in building the model, the better the result
 There comes a point at which test error does not get any worse upon further lowering the penalty parameter
 There also comes a point at which train error does not get any better upon further lowering the penalty parameter (bringing the model close to the OLS linreg solution)
 If I had included larger values of lambda, then we would see the errors levelling 
 off as lambda increases to a certain point.
'




#The Lasso
k=5 
set.seed(1)
model_count<-length(grid)
grid=10^seq(3, -2, by = -.1) #lambda values
folds=sample(1:k,nrow(df_final),replace=TRUE)
cv.errors=matrix(NA,k,model_count, dimnames=list(NULL, paste(1:model_count)))
cv.errors.train=matrix(NA,k,model_count, dimnames=list(NULL, paste(1:model_count)))

#cv.glmnet performs k-fold but how do i do median imputation in the train folds??
for(j in 1:k){  
  preproc <- preProcess(df_final_x[folds!=j,], method=c("medianImpute"))
  x_train_folds <- data.matrix(predict(preproc, df_final_x[folds!=j,])) #need matrix of x values
  y_train_folds <- as.vector(df_final_y[folds!=j])
  model_lasso=glmnet(x_train_folds,y_train_folds,alpha=1,lambda=grid,standardize=TRUE) 
  for (i in 1:model_count){
    x_test_folds <- data.matrix(predict(preproc, df_final_x[folds==j,]))
    pred=predict(model_lasso,s=grid[i],newx=x_test_folds)
    pred_train=predict(model_lasso,s=grid[i],newx=x_train_folds)
    cv.errors[j,i]=mean((df_final_y[folds==j]-pred)^2) #row j is the fold, col i is the model for lambda i
    cv.errors.train[j,i]=mean((df_final_y[folds!=j]-pred_train)^2)
  }
}

#Lasso - Test Avg CV RMSE Scores for each lambda 
mean.cv.errors=apply(cv.errors ,2,mean)
par(mfrow=c(1,1),mai = c(.6, .6, .13, 0.1), cex=.7) 
plot(mean.cv.errors ,type='b')
sqrt(mean.cv.errors) #as lambda gets smaller, the test error gets worse
best_lambda<-grid[which(mean.cv.errors==min(mean.cv.errors))] #best lambda value
lasso_rmse<-sqrt(min(mean.cv.errors))
best_lambda

#Lasso - Train Avg CV RMSE Scores for each lambda
mean.cv.errors.train=apply(cv.errors.train ,2,mean)
par(mfrow=c(1,1),mai = c(.6, .6, .13, 0.1), cex=.7) 
plot(mean.cv.errors.train ,type='b')
sqrt(mean.cv.errors.train)
best_lambda_train<-grid[which(mean.cv.errors.train==min(mean.cv.errors.train))]
lasso_train_rmse<-sqrt(min(mean.cv.errors.train))


#For the last group of folds the model was built on:
plot(model_lasso) #shows coefficients as L1 Norm increases (as penalty decreases)
model_lasso.coef=predict(model_lasso,type="coefficients",s=best_lambda)
model_lasso.coef #Many predictors have there coefficients set to zero




#Polynomial Regression
#as.formula() creates a formula from a string
#paste() converts arguments to strings and concatenates them
#I didn't use poly() b/c you can't use it with missing values, even w/ median imp / na.action=na.exclude
set.seed(1)
poly_formula <-
          as.formula(
          paste('SalePrice ~ ',
          paste(paste(colnames(df_final_x),collapse='+'),  #add non-poly predictors var1+var2+...+varj
          paste('LotArea^2'),     #add in poly variable
          paste('LotFrontage^2'), 
          paste('LotFrontage^3'), 
          sep='+'))) #insert + between these 3 terms (var1+...+varj poly(lotarea,2) poly(lotfrontage,3)) 

model_poly<-train(poly_formula, data=df_final, method="lm", 
      preProcess=c("medianImpute"),
      trControl=control, tuneLength=5,na.action=na.pass) 

model_poly
summary(model_poly)












#Comparing model RMSE results
min(model_lr$results$RMSE)   # RMSE 34937
min(model_knn$results$RMSE)  # RMSE 48012
min(model_rf$results$RMSE)   # RMSE 29588
min(model_gbm$results$RMSE)  # RMSE 27837 <--- winning model
min(model_gbm2$results$RMSE) # RMSE 73560
min(model_pcr$results$RMSE)  # RMSE 39840
min(model_pls$results$RMSE)  # RMSE 33387
min(model_poly$results$RMSE) # RMSE 34926
forward_selection_rmse       # RMSE 38375
ridge_regression_rmse        # RMSE 34658
lasso_rmse                   # RMSE 33131



 
final_model = gbm(formula = SalePrice ~ . ,
               distribution = 'gaussian',
               data = df_final,
               #n.trees = 2500,
               shrinkage = .01,
               n.minobsinnode = 20)

#Preparing the Test Data
df_test_ordinal    <- df_test[ordinal]
df_test_nominal    <- df_test[nominal]
df_test_discrete   <- df_test[discrete]
df_test_continuous <- df_test[continuous]
df_test_date       <- df_test[date]

#Label Encode Test Ordinal Data
df_test_ordinal_encoded <- df_test_ordinal
for (j in 1:ncol(df_test_ordinal)) {
  if (colnames(df_test_ordinal)[j] == 'OverallQual' |
      colnames(df_test_ordinal)[j] == 'OverallCond') {
    df_test_ordinal_encoded[j] <- df_test_ordinal[j]}
  else if (colnames(df_test_ordinal)[j] != 'BsmtExposure') {
    df_test_ordinal_encoded[j] <- lapply(df_test_ordinal[j], label_encode)
  } else if (colnames(df_test_ordinal)[j] == 'BsmtExposure') {
    df_test_ordinal_encoded[j] <- lapply(df_test_ordinal[j], bsmtexposure_encode )
  }}

#Combine Numeric Data and Convert it to Numeric Data Type
df_test_numeric_ordinal_encoded <- data.frame(df_test_discrete,df_test_continuous,df_test_ordinal_encoded)
df_test_numeric_ordinal_encoded2<-data.frame(lapply(df_test_numeric_ordinal_encoded, function(x) as.numeric(x)))

#One-Hot Encode Test Nominal Data
df_test_nominal$MSSubClass<-as.character(df_test_nominal$MSSubClass)
df_test_nominal_dummies <- dummyVars(" ~ .", data = df_test_nominal)
df_test_nominal_dummies <- data.frame(predict(df_test_nominal_dummies, newdata = df_test_nominal))
#error: contrasts can be applied only to factors with 2 or more levels

#Per above error, exploring count of unique levels for df_test_nominal predictors
for (j in 1:length(colnames(df_test_nominal))){
  print(c(colnames(df_test_nominal)[j], length(unique(df_test_nominal[,j]))))
}

#df_test predictors with <= 2 unique 
df_test_nominal[,c('Street','CentralAir','Utilities')]
table(df_test_nominal['Street'])
table(df_test_nominal['CentralAir'])
table(df_test_nominal['Utilities'])  # <-- only has AllPub level and NA levels!!
unique(df_test_nominal['Utilities']) # <-- only has AllPub level and 2 NA levels!!
nrow(df_test_nominal) # <- 1459 total levels, and above shows 1457 AllPub levels

#Manually make Utilities column a dummy, since dummyVars can't handle vars with 1 level
df_test_nominal$Utilities<-as.integer(as.factor(df_test_nominal$Utilities))
df_test_nominal_dummies <- dummyVars(" ~ .", data = df_test_nominal)
df_test_nominal_dummies <- data.frame(predict(df_test_nominal_dummies, newdata = df_test_nominal))
df_test_nominal_dummies$Utilities

#Create Final Test Dataframe
df_test_final <- cbind(df_test_nominal_dummies,df_test_numeric_ordinal_encoded2,df_test_date) #Excludes SalePrice b/c there is none!!

#The below fails because some df_test nominal level columns don't exist and they did when building the model in the train set
final_predictions = predict(object = final_model,
                              newdata = df_test_final,
                              n.trees = 100,
                              type = "response")

#Check which nominal level dummy columns are not in the test set (but are in train)
missing_test_nominals<-setdiff(colnames(df_nominal_dummies),colnames(df_test_nominal_dummies))
missing_test_nominals

#Use a for loop to add columns to the dataframe
for (j in missing_test_nominals){
  df_test_nominal_dummies[j]<-rep(0,nrow(df_test_nominal_dummies)) #using $j does not work!
}

#Check that the test nominal df has all train nominal df columns
setdiff(colnames(df_nominal_dummies),colnames(df_test_nominal_dummies)) 

#Build final test dataframe, make predictions, and spit out in csv
df_test_final <- cbind(df_test_nominal_dummies,df_test_numeric_ordinal_encoded2,df_test_date)
final_predictions = data.frame(predict(object = final_model,newdata = df_test_final, n.trees = 100,type = "response"))
final_predictions$Id<-df_test$Id #add Id to prediction dataframe
final_predictions <- final_predictions[c(2,1)] #reorder columns so Id comes first (proper format for kaggle)
colnames(final_predictions)[2] <- "SalePrice" #rename 2nd column to SalePrice (proper format)
write.csv(final_predictions, "/Users/Work/Desktop/Work/Projects/Kaggle/House Prices-Advanced Regression Techniques /Submission.csv", row.names = FALSE)




#NOT USED IN FINAL CODE SHOWN BELOW
"Decision Tree model isn't working due to error: 
  There were missing values in resampled performance measures 
  Perhaps there are too many variabes to split on
  So try removing nominals with too many levels
  Return number of unique values in each dataframe column (and the column name)"

for (j in 1:length(colnames(df[,nominal]))){
  print(c(colnames(df[,nominal])[j], length(unique(df[,nominal][,j]))))
}

length(row.names(table(df[,nominal][,1]))) #return # of unique values in column
length(unique(df[,nominal][,1]))           #return # of unique values in column

df[,-c("MSSubClass", "Neighborhood", "Exterior1st", "Exterior2nd")]
subset(df, select = c("MSSubClass", "Neighborhood", "Exterior1st", "Exterior2nd"))

temp<-which( colnames(df)=="MSSubClass" | colnames(df)=="Neighborhood"
       | colnames(df) == "Exterior1st" | colnames(df)=="Exterior2nd")
df_large_nominals_removed<-subset(df,select=-temp)


#Remove rows withs missing observations and build models 
#(UNNECESSARY BECAUSE train()'s na.action=na.omit does this)
df_final_rows_w_missings <- df_final[rowSums(is.na(df_final)) > 0,]
df_final_missings_indices<-rownames(df_final_rows_w_missings)
df_final_nulls_removed<-df_final[-as.integer(df_final_missings_indices),]



#Unsupervised Learning 

#Median Impute Missings
library(DMwR)
#impute(df_final$ptratio, median) <--each column must be specified manually
#KNN imputation computes the weighted avg variable value from the k closest observations
df_final_imputed<- knnImputation(df_final[, !names(df_final) %in% "SalePrice"])  #<-- imputes all vars, unless you exclude some


#Plot clusters of the first two principal components (we need to standardize continuous vars for this)
#First standardize "continuous" variables
#Code to drop / remove columns
temp<-df_final_imputed[,-which((names(df_final_imputed) %in% c(continuous,discrete,ordinal,date)))]
#Code to keep certain columns
vars_to_standardize<-df_final_imputed[,which((names(df_final_imputed) %in% c(continuous,discrete,ordinal,date)))]
#Use above code to standardize continuous,discrete,date,ordinal (but not nominal) predictors
preprocess_fit<-preProcess(vars_to_standardize, method=c("center", "scale"))
vars_standardized <- predict(preprocess_fit, vars_to_standardize)
vars_unstandardized <- df_final_imputed[,-which((names(df_final_imputed) %in% c(continuous,discrete,ordinal,date)))]
df_final_imputed_standardized<-data.frame(c(vars_standardized,vars_unstandardized))

#Principal Components Analysis
#Generate first 2 principal components (calculate their scores)
princomp<-prcomp(df_final_imputed_standardized)
biplot(princomp) #plot first two principal components
princomp_2<-princomp$x[,1:2] #extract first two principal components

#Calculate eigenvalues (principal component variances) used to select the # of principal components
princomp_std <- princomp$sdev #compute standard deviation of each principal component
princomp_var <- princomp_std^2 #compute variance (eigenvalue) of each principal component
princomp_var[1:10] #variance of first 10 components (eigenvalues)
princomp_var_prop <- princomp_var/sum(princomp_var) #Scree Plot (% of variance explained by # of princomp)
princomp_var_prop[1:10] #% of variance explained by each component

#Use a Scree Plot to select the optimal number of principal components
par(mfrow=c(1,1),mai = c(.6, .6, .13, 0.1), cex=.6) 
plot(princomp_var_prop, xlab = "Principal Component", ylab = "Proportion of Variance Explained", type = "b")
plot(cumsum(princomp_var_prop), xlab = "Principal Component",ylab = "Cumulative Proportion of Variance Explained",type = "b")
plot(princomp_var, xlab = "Principal Component", ylab="Eigenvalue", type="b") 
title("Scree Plot")
cumsum(princomp_var_prop)[1:10] #cum % of variance explained by first 10 components

#Select the # of principal components such that 80% of the total variation in the original variables is explained
data.frame(cumsum(princomp_var_prop)[1:max_vars]) #we need 29 principal components to explain at least 80% of the variance 


#KMeans Clustering
set.seed(3) #this is important so that cluster results can be replicated
kmeans=kmeans(df_final_imputed,3,nstart=20) 
#nstart is important and should be between 20-50; 
#it is the number of initial randomized cluster configurations that attempted; the one w/ the best results is chosen
kmeans #79.3% of the variation in points is explained by the between cluster variation (SSB/SST)
kmeans$betweenss/(kmeans$tot.withinss+kmeans$betweenss) #79.3%
#The higher this value, the better
#where SSB = sum of squared distances of each cluster centroid to the global mean
#where SST = sum of squared distances of each point to the global sample mean
kmeans$cluster #show each observation and its associated cluster
count(kmeans$cluster) #count the number of each assigned cluster (most observations are assigned to cluster 1)

#Elbow Method for finding the optimal number of clusters
set.seed(123)
data<-df_final_imputed_standardized
#Compute and plot wss for k = 2 to k = 15. 
k.max <- 15
wss <- sapply(1:k.max, function(k){kmeans(data, k, nstart=50,iter.max = 15 )$tot.withinss})

#Plot WSS by # of clusters (the elbow is around 6 clusters)
par(mfrow=c(1,1),mai = c(.6, .6, .13, 0.1), cex=.6) 
#mai(bottom, left, top, right) <- controls margin spacing around each plot in inches
#cex <- controls plotting text and symbols size (the amt by which they should be scaled from their defaults)
plot(1:k.max, wss,type="b", pch = 19, frame = FALSE, xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")

#Plot clusters of first two principal components (princomp_2)
graphics.off()
par("mar")
par(mar=c(1,1,1,1))
kmeans=kmeans(df_final_imputed,6,nstart=20) 
plot(princomp_2, col=(kmeans$cluster),  main="K-Means Clustering with K=2", pch=10, cex=1)
#K-mean clusters seem undistinguishable when projecting data onto 2 principal components



#Hierarchical Clustering
temp<-df_final_imputed_standardized[1:max_vars,]
hc.complete <- hclust(dist(temp),method='complete')
hc.average  <- hclust(dist(temp), method="average")
hc.single   <- hclust(dist(temp), method="single")

#Creating a dendogram of the first max_vars observations
par(mfrow=c(1,3), mar=c(1,1,1,1)) 
#mar indicates margin size in lines:default = c(5, 4, 4, 2) + 0.1
#mai indicates margin size in inches (bottom, left, top, right)
plot(hc.complete,main="Complete Linkage", xlab="", sub="", cex =.9)
plot(hc.average , main="Average Linkage", xlab="", sub="", cex =.9)
plot(hc.single , main="Single Linkage", xlab="", sub="",cex =.9)

#Creating hierarchical clusters by cutting the tree (cutree produces cluster labels)
hc.complete <- hclust(dist(df_final_imputed_standardized),method='complete')
hc.average  <- hclust(dist(df_final_imputed_standardized), method="average")
hc.single   <- hclust(dist(df_final_imputed_standardized), method="single")
cutree(hc.complete, 6) #cutree(model,k,h) k=# of groups, h=cut height, k overrides h
table(cutree(hc.complete, 6)) #most observations are in cluster 1, unlike in kmeans
table(cutree(hc.average, 6))  #most observations are in cluster 1
table(cutree(hc.single, 6))   #most observations are in cluster 1

#Plot the k=6 hierarchical clusters of the first two principal components
plot(princomp_2, col=cutree(hc.complete, 6),  main="Hierarchical Clustering with K=6", pch=10, cex=1)


