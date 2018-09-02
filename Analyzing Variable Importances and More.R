library(cluster)
library(useful)
library(Hmisc)
library(HSAUR)
library(MVA)
library(HSAUR2)
library(fpc)
library(mclust)
library(lattice)
library(car)
library(plyr) 
library(plot3D)
library(corrplot)
library(randomForest)
library(caret)
library(gbm)

rm(list = ls()) #remove all objects from the global environment
IM<-read.csv('/Users/Work/Desktop/Work/Projects/Datasets/Analytics Exercise/IM.csv')


#Check for duplicated and missing values
sum(duplicated(IM$Contact.Key))
t(colSums(is.na(IM)))

#Inspect dataframe structure
str(IM)
#Change club_aff and Resp to factors, since they are categorical predictors
IM$club_aff<-factor(club_aff)
IM$Resp<-factor(Resp)
#Create dataframe of numerical predictors by removing non-numerical predictors
numsub<-subset(IM,select=c(-club_aff,-Contact.Key,-Resp))

#Check variable distributions for garbage values (weird outliers)
summary(IM)


#Make correlation matrix of numerical predictors
mcor<-cor(numsub)
corrplot(mcor,method='shade',shade.col=NA,tl.col='black',tl.cex=.5)

'''
Finish.Time, Swim.Time, Bike.Time, and Run.Time are very strongly positively
correlated with one another.  In other words, multicollinearity is present
in our dataset, so we might have multicollinearity issues if these predictors are
all kept in the dataset.  It will be difficult to know which of these predictors
is most predictive of the outcome, since all are so incredibly correlated.  

This is also the issue for Finish.Rank, Swim.Rank, Bike.Rank, and Run.Rank.
The 4 rank predictors are also strongly correlated with the 4 Time predictors,
so it is difficult to see.
'''

#Check categorical predictor relatedness to outcome
boxplot(IM$club_aff~IM$Resp, data=IM, notch=TRUE, 
        #col=(c("gold","darkgreen")),
        main="club_aff vs resp", xlab="Resp",
        outline = FALSE)

#Check numerical predictors' relatedness to outcome
lapply(aggregate(x ~ IM.df$Resp, FUN=mean)
       aggregate(IM.df[, 2:16], list(IM$Resp), mean)
       
       #Inspecting the relatedness of the categorical predictor to the outcome
       table(IM$club_aff,IM$Resp)
       3708/(3708+25729) #13% of non-tri-clubbers registered for the full race
       1592/(1592+5220) #23.4% of tri-clubbers registered for the full race
       
       #Check predictor distributions
       for (j in 1:dim(numsub)[2]) {
         hist(numsub[,j], main = paste("Histogram of" , attributes(numsub)$names[j]))
       }
       
       '''
       Right-skewed distributions: R2016, R2015, R2014, prior_races
       Normal distributions: Age, 
       Left-skewed distributions: Min_Year
       '''
       
       #Standardize numeric predictor distributions
       numsub.mean <- apply(numsub, 2, mean) 
       numsub.sd <- apply(numsub, 2, sd)
       numsub.std <- t((t(numsub)-numsub.mean)/numsub.sd) 
       apply(numsub.std, 2, mean) # check zero mean
       apply(numsub.std, 2, sd) # check unit sd
       IM.std <- data.frame(numsub.std,IM$club_aff,IM$Resp)
       
       #Check standardized predictor distributions
       for (j in 1:dim(numsub.std)[2]) {
         hist(numsub.std[,j], main = paste("Histogram of" , attributes(numsub.std)$names[j]))
       }
       
       #The outcome distribution is very unbalanced (5300 y=1 observations)
       table(IM$Resp)
       
       #Dataframe of standardized numerics, subsetted data
       numsub2<-c('R2016','Finish.Time','Finish.Rank','prior_races','Age','Min_Year')
       IM.std.sub <- IM.std[,c(numsub2,'IM.Resp','IM.club_aff')]
       
       #Dataframe of standardized numerics, un-subsetted data
       IM.std
       
       #Dataframe of unstandardized numerics, subsetted data
       IM.sub <- IM[,c(numsub2,'Resp','club_aff')]
       
       #Dataframe of unstandardized numerics, un-subsetted data
       IM.df <- IM[,-1] #remove ID variable
       
       
       
       #MODEL BUILDING FOR FEATURE IMPORTANCE EVALUATION
       #RANDOM FOREST
       #standardized, non-subsetted
       set.seed(1)
       model.rf1=randomForest(IM.std$IM.Resp~.,data=IM.std,importance =TRUE,ntree=30)
       options(repr.plot.width=10, repr.plot.height=5) #resize plots
       varImpPlot(model.rf1)
       varImp(model.rf1)
       
       #standardized, subsetted
       set.seed(1)
       model.rf2=randomForest(IM.std.sub$IM.Resp~.,data=IM.std.sub,importance =TRUE,ntree=30)
       varImp(model.rf2)
       
       #non-standardized, subsetted
       set.seed(1)
       model.rf3=randomForest(IM.sub$Resp~.,data=IM.sub,importance =TRUE,ntree=30)
       varImp(model.rf3) #from caret package
       
       #non-standardized, non-subsetted
       set.seed(1)
       model.rf4=randomForest(IM.df$Resp~.,data=IM.df,importance =TRUE,ntree=30)
       options(repr.plot.width=10, repr.plot.height=5) #resize plots
       varImpPlot(model.rf4)
       varImp(model.rf4) #from caret package
       
       #non-standardized, non-subsetted, all predictors considered at each split
       set.seed(1)
       model.rf5=randomForest(IM.df$Resp~.,data=IM.df,importance =TRUE,ntree=30,mtry=ncol(IM.df)-1)
       varImp(model.rf5)
       
       #standardized, subsetted, all predictors considered at each split
       set.seed(1)
       model.rf6=randomForest(IM.std.sub$IM.Resp~.,data=IM.std.sub,importance =TRUE,ntree=30,mtry=ncol(IM.df)-1)
       varImp(model.rf6)
       
       #standardized, non-subsetted, all predictors considered at each split
       set.seed(1)
       model.rf7=randomForest(IM.std$IM.Resp~.,data=IM.std,importance =TRUE,ntree=30,mtry=ncol(IM.df)-1)
       varImp(model.rf7)
       
       
       #LOGISTIC MODEL
       #standardized, non-subsetted
       model.log1 <- glm(IM.std$IM.Resp~.,data=IM.std, family=binomial("logit"))
       varImp(model.log1)
       #standardized, subsetted
       model.log2 <- glm(IM.std.sub$IM.Resp~.,data=IM.std.sub, family=binomial("logit"))
       varImp(model.log2)
       #non-standardized, subsetted
       model.log3 <- glm(IM.sub$Resp~.,data=IM.sub, family=binomial("logit"))
       varImp(model.log3)
       #non-standardized, non-subsetted
       model.log4 <- glm(IM.df$Resp~.,data=IM.df, family=binomial("logit"))
       varImp(model.log4)
       
       
       
       #GBM MODEL
       #reformat Resp variable to numeric for GBM since summary function generates error when it's a factor
       IM.std2<-IM.std
       IM.std2$IM.Resp<-as.numeric(IM$Resp)
       IM.std.sub2<-IM.std.sub
       IM.std.sub2$IM.Resp<-as.numeric(IM$Resp)
       IM.sub2<-IM.sub
       IM.sub2$IM.Resp<-as.numeric(IM$Resp)
       IM.df2<-IM.df
       IM.df2$Resp<-as.numeric(IM$Resp)
       
       #standardized, non-subsetted
       set.seed(1)
       model.gbm1=gbm(IM.std2$IM.Resp~.,data=IM.std2,distribution ="bernoulli",n.trees=500, interaction.depth = 4)
       summary(model.gbm1)
       #standardized, subsetted
       set.seed(1)
       model.gbm2=gbm(IM.std.sub2$IM.Resp~.,data=IM.std.sub2,distribution = "bernoulli",n.trees=500, interaction.depth = 4)
       summary(model.gbm2)
       #non-standardized, subsetted
       set.seed(1)
       model.gbm3=gbm(IM.sub2$Resp~.,data=IM.sub2,distribution = "bernoulli",n.trees=500, interaction.depth = 4)
       summary(model.gbm3)
       #non-standardized, non-subsetted
       set.seed(1)
       model.gbm4=gbm(IM.df2$Resp~.,data=IM.df2,distribution ="bernoulli",n.trees=500, interaction.depth = 4)
       summary(model.gbm4)
       
       
       
       #Testing significance of R2016, R2015, R2014, prior_races, Min_Year variables within the logistic regression
       #Removing variables collinear with R2016 (all the ones listed above) in the subsetted model
       IM.std.sub3<-IM.std.sub[,c(-4,-6)]
       set.seed(1)
       model.rf8=randomForest(IM.std.sub3$IM.Resp~.,data=IM.std.sub3,importance =TRUE,ntree=30)
       varImp(model.rf8)
       
       set.seed(1)
       model.gbm5=gbm(IM.std.sub3$IM.Resp~.,data=IM.std.sub3,distribution = "bernoulli",n.trees=500, interaction.depth = 4)
       summary(model.gbm5)
       
       
       
       #Only Including (R2014, R2015, R2016)
       IM.std.sub4<-IM.std.sub3
       IM.std.sub4$R2015<-IM.df$R2015
       IM.std.sub4$R2014<-IM.df$R2014
       set.seed(1)
       model.rf9=randomForest(IM.std.sub4$IM.Resp~.,data=IM.std.sub4,importance =TRUE,ntree=30)
       varImp(model.rf9)
       
       #Only Including (min_year, prior_races) and (excluding Rxxx variables)
       IM.std.sub5<-IM.std.sub4[,c(-1,-7,-8)]
       IM.std.sub5$Min_Year<-IM.df$Min_Year
       IM.std.sub5$prior_races<-IM.df$prior_races
       set.seed(1)
       model.rf10=randomForest(IM.std.sub5$IM.Resp~.,data=IM.std.sub5,importance =TRUE,ntree=30)
       varImp(model.rf10)
       
       #Only including (R2014, Min_Year, prior_races)
       IM.std.sub6<-IM.std.sub
       IM.std.sub6$R2014<-IM.df$R2014
       IM.std.sub6<-IM.std.sub6[,-1]
       set.seed(1)
       model.rf11=randomForest(IM.std.sub6$IM.Resp~.,data=IM.std.sub6,importance =TRUE,ntree=30)
       varImp(model.rf11)
       
       set.seed(1)
       model.log5 <- glm(IM.std.sub6$IM.Resp~.,data=IM.std.sub6, family=binomial("logit"))
       varImp(model.log5)
       
       #Only including (R2015, Min_Year, prior_races)
       IM.std.sub7<-IM.std.sub
       IM.std.sub7$R2015<-IM.df$R2015
       IM.std.sub7<-IM.std.sub7[,-1]
       set.seed(1)
       model.rf12=randomForest(IM.std.sub7$IM.Resp~.,data=IM.std.sub7,importance =TRUE,ntree=30)
       varImp(model.rf12)
       
       set.seed(1)
       model.log6 <- glm(IM.std.sub7$IM.Resp~.,data=IM.std.sub7, family=binomial("logit"))
       varImp(model.log6)
       
       #Inspecting R2014, R2015, R2016 values in non-subsetted, standardized model when min_year, prior races are removed
       IM.std2<-IM.std[,c(-9,-10)]
       set.seed(1)
       model.rf13=randomForest(IM.std2$IM.Resp~.,data=IM.std2,importance =TRUE,ntree=30)
       varImp(model.rf13)
       
       #Same as above but using non-standardized, non-subsetted data (all predictors considered at each split)
       set.seed(1)
       IM.df3<-IM.df[,c(-10,-11)]
       model.rf14=randomForest(IM.df3$Resp~.,data=IM.df3,importance =TRUE,ntree=30,mtry=ncol(IM.df)-1)
       varImp(model.rf14)
       
       
       