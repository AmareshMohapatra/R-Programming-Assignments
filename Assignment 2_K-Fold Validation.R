install.packages("titanic")
install.packages("rpart.plot")
install.packages("randomForest")
install.packages("DAAG")
library(titanic)
library(rpart.plot)
library(gmodels)
library(Hmisc)
library(pROC)
library(ResourceSelection)
library(car)
library(caret)
library(dplyr)
library(InformationValue)
library(rpart)
library(randomForest)
library("DAAG")

cat("\014") # Clearing the screen

getwd()
setwd("C:/Users/Amaresh-PC/Documents/Amma 2017/Data/titanic") 

titanic_train_2<-read.csv('train.csv')
titanic_train<-titanic_train_2
titanic_train_3 <- read.csv('train.csv')

#titanic test
titanic_test_const <-read.csv('test-3.csv')

#splitting titanic train into 70,30
set.seed(100) # for reproducibility
titanic_train$rand <- runif(nrow(titanic_train))
titanic_train_start <- titanic_train[titanic_train$rand <= 0.7,]
titanic_test_start <- titanic_train[titanic_train$rand > 0.7,]


# number of survived vs number of dead
CrossTable(titanic_train$Survived)

# replacing NA by mean
mean_age = mean(titanic_train_2$Age)
titanic_train_mean_amaresh <- titanic_train_start
titanic_train_mean_amaresh2 <- titanic_train_start
titanic_train_mean_amaresh$Age[is.na(titanic_train_mean_amaresh$Age)] = mean(titanic_train_mean_amaresh$Age, na.rm = TRUE)
titanic_train_mean_amaresh2$Age[is.na(titanic_train_mean_amaresh2$Age)] = mean(titanic_train_mean_amaresh2$Age, na.rm = TRUE)

########## Build model from mean imputed into the data set ##########

full.model.titanic.mean <- glm(formula = Survived ~ Pclass + Sex + SibSp + Parch + Fare + Age,
                               data=titanic_train_mean_amaresh, family = binomial) #family = binomial implies that the type of regression is logistic

#lm
fit.train.mean <- lm(formula = Survived ~ Pclass + Sex + SibSp + Parch + Fare + Age,
                     data=titanic_train_mean_amaresh2) #family = binomial implies that the type of regression is logistic
summary(fit.train.mean)

#vif - remove those variables which have high vif >5
vif(fit.train.mean) 

#removing insignificant variables
titanic_train_mean_amaresh$Parch<-NULL
full.model.titanic.mean <- glm(formula = Survived ~ Pclass + Sex + SibSp + Fare + Age,
                               data=titanic_train_mean_amaresh, family = binomial) #family = binomial implies that the type of regression is logistic
summary(full.model.titanic.mean)

titanic_train_mean_amaresh$Fare<-NULL
full.model.titanic.mean <- glm(formula = Survived ~ Pclass + Sex + SibSp + Age,
                               data=titanic_train_mean_amaresh, family = binomial) #family = binomial implies that the type of regression is logistic
summary(full.model.titanic.mean)


#Testing performance on Train set

titanic_train_mean_amaresh$prob = predict(full.model.titanic.mean, type=c("response"))
titanic_train_mean_amaresh$Survived.pred = ifelse(titanic_train_mean_amaresh$prob>=.5,'pred_yes','pred_no')
table(titanic_train_mean_amaresh$Survived.pred,titanic_train_mean_amaresh$Survived)

#Testing performance on test set
nrow(titanic_test)
nrow(titanic_test2_mean_amaresh)
titanic_test2_mean_amaresh <- titanic_test_start

#imputation by replacing NAs by means in the test set
titanic_test2_mean_amaresh$Age[is.na(titanic_test2_mean_amaresh$Age)] = mean(titanic_test2_mean_amaresh$Age, na.rm = TRUE)

titanic_test2_mean_amaresh$prob = predict(full.model.titanic.mean, newdata=titanic_test2_mean_amaresh, type=c("response"))
titanic_test2_mean_amaresh$Survived.pred = ifelse(titanic_test2_mean_amaresh$prob>=.5,'pred_yes','pred_no')
table(titanic_test2_mean_amaresh$Survived.pred,titanic_test2_mean_amaresh$Survived)

########## END - Model with mean included instead of NA #########

### Testing for Jack n Rose's survival ###
df.jackrose <- read.csv('Book1.csv')
df.jackrose$prob = predict(full.model.titanic.mean, newdata=df.jackrose, type=c("response"))
df.jackrose$Survived.pred = ifelse(df.jackrose$prob>=.5,'pred_yes','pred_no')
head(df.jackrose)

# Jack dies, Rose survives

### END - Testing on Jack n Rose ###

### START decile chart ###

# decile chart and decision making
titanic_train_mean_amaresh$decile <- with(titanic_train_mean_amaresh, cut(prob, 
                                                                          breaks=quantile(prob, probs=seq(0,1, by=0.1), na.rm=TRUE), 
                                                                          include.lowest=TRUE))
titanic_train_mean_amaresh$decile <- as.numeric(titanic_train_mean_amaresh$decile)
titanic_train_mean_amaresh <- titanic_train_mean_amaresh[order(-titanic_train_mean_amaresh$decile),]
titanic.train.mean.group <- titanic_train_mean_amaresh %>% group_by(decile)
#train.group %>% summarise(total_yes = sum(yact), total = count(yact))
#titanic.train.mean.group %>% summarise(total_yes = sum(as.numeric(Survived)), total = count(as.numeric(Survived)))

decile.titanic.mean <- aggregate(titanic_train_mean_amaresh$Survived, by=list(titanic_train_mean_amaresh$decile), FUN = sum)
decile.titanic.mean  <- decile.titanic.mean[order(-decile.titanic.mean$Group.1),]
colnames(decile.titanic.mean) <- c('decile', 'total_yes')
decile.titanic.mean$random_yes <- round(sum(decile.titanic.mean$total_yes)/10)
decile.titanic.mean$pcnt_yes <- decile.titanic.mean$total_yes/sum(decile.titanic.mean$total_yes)
decile.titanic.mean$cum_pcnt_yes <- cumsum(decile.titanic.mean$pcnt_yes)
print(decile.titanic.mean)

### END decile chart ###

### START Decision Tree ###

fit.rpart = rpart(Survived ~ Pclass + Sex + Age + Parch + Fare + SibSp, titanic_train_2, method = "class")
summary(fit.rpart)
# to view the decision tree
rpart.plot(fit.rpart)

# run together start
plot_fit.rpart = plot(fit.rpart, unifrom= TRUE, main = "classification")
text(fit.rpart, use.n=TRUE, all=TRUE, cex = .8)
#run together end

### END Decision Tree ###

## START Random Forest ##
titanic_train_NA_allcols <- titanic_train_2[!apply(titanic_train_2[,c("Pclass", "Sex", "SibSp", "Parch", "Fare", "Age")], 1, anyNA),]
rf <- randomForest(Survived ~ Pclass + Sex + Age + Parch + Fare + SibSp, titanic_train_NA_allcols, mtry = 2, importance = TRUE)
print(rf)
summary(rf)
## END Random Forest ##

## START  K-fold cross validation ##

# Defining the K Fold CV function here
Kfold_func <- function(dataset,formula,family,k)
{
  object <- glm(formula=formula, data=dataset, family = family)
  CVbinary(object, nfolds= k, print.details=TRUE)
}

#Defining the function to calculate Mean Squared Error here
MeanSquareError_func <- function(dataset,formula)
{
  LM_Object <- lm(formula=formula, data=dataset)
  LM_Object_sum <-summary(LM_Object)
  MSE <- mean(LM_Object_sum$residuals^2)
  print("Mean squared error")
  print(MSE)
}

#Performing KFold CV on Training set by calling the KFOLD CV function here
Kfoldobj <- Kfold_func(titanic_train_mean_amaresh,Survived ~ Pclass + Sex + SibSp + Age,binomial,10)

#Calling the Mean Squared Error function on the training set here
MSE_Train <-MeanSquareError_func(titanic_train_mean_amaresh,Survived ~ Pclass + Sex + SibSp + Age)

#confusion matrix on training set
table(titanic_train_mean_amaresh$Survived,round(Kfoldobj$cvhat))
print("Estimate of Accuracy")
print(Kfoldobj$acc.cv)

#Performing KFold CV on test set by calling the KFOLD CV function here
Kfoldobj.test <- Kfold_func(titanic_test2_mean_amaresh,Survived ~ Pclass + Sex + SibSp + Age,binomial,10)

#Calling the Mean Squared Error function on the test set here
MSE_Test <-MeanSquareError_func(titanic_test2_mean_amaresh,Survived ~ Pclass + Sex + SibSp + Age)

#Confusion matrix on test set
table(titanic_test2_mean_amaresh$Survived,round(Kfoldobj.test$cvhat))
print("Estimate of Accuracy")
print(Kfoldobj.test$acc.cv)

## END K-FOLD CROSS VALIDATION ##



###### Accuracy on Training Set ######
#                                    #
# "Estimate of Accuracy" = 0.8032787
#                                    #
###### Accuracy on Training Set ######



###### Accuracy on Testing Set ######
#                                   #
# "Estimate of Accuracy" = 0.7793594
#                                   #
###### Accuracy on Testing Set ######


###### Mean Squared Error on Training set #######
#                                               # 
# "Mean squared error" = 0.1371759              #
#                                               #      
###### Mean Squared Error #######################

###### Mean Squared Error on Testing set #######
#                                               # 
# "Mean squared error" = 0.1471306              #
#                                               #      
###### Mean Squared Error #######################
