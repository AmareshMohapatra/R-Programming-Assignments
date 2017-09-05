install.packages("gmodels")
install.packages("Hmisc")
install.packages("pROC")
install.packages("ResourceSelection")
install.packages("car")
install.packages("caret")
install.packages("dplyr")
library(gmodels)
library(Hmisc)
library(pROC)
library(ResourceSelection)
library(car)
library(caret)
library(dplyr)
install.packages("InformationValue")
library(InformationValue)


# Set working directory - 
getwd()
setwd("C:/Users/Amaresh-PC/Documents/Amma 2017/Data/assignment")

# read datasets
df.bclient <- read.csv('bank_client.csv')
str(df.bclient)

# read attributes
df.battr <- read.csv('bank_other_attributes.csv')
str(df.battr)

# read campaign data
df.campaign <- read.csv('latest_campaign.csv')
str(df.campaign)

# read campaign outcome
df.campOutcome <- read.csv('campaign_outcome.csv')
str(df.campOutcome)

df.temp1 <- merge(df.bclient, df.campaign, by = 'Cust_id', all.x = TRUE)
df.temp2 <- merge(df.temp1, df.battr, by = 'Cust_id', all.x = TRUE)
df.data <- merge(df.temp2, df.campOutcome, by = 'Cust_id', all.x = TRUE)

# merged dataset
head(df.data)
summary(df.data)
str(df.data)
CrossTable(df.data$y)

# split data into training and test
set.seed(1234) 
df.data$rand <- runif(nrow(df.data))
df.train <- df.data[df.data$rand <= 0.7,]
df.test <- df.data[df.data$rand > 0.7,]
nrow(df.train)
nrow(df.test)

CrossTable(df.train$job, df.train$y)
CrossTable(df.train$marital, df.train$y)
CrossTable(df.train$education, df.train$y)
CrossTable(df.train$default, df.train$y)
CrossTable(df.train$housing, df.train$y)
CrossTable(df.train$loan, df.train$y)
CrossTable(df.train$poutcome, df.train$y)

hist(df.train$age)
hist(df.train$balance)
hist(df.train$duration)
hist(df.train$campaign)
hist(df.train$pdays)
hist(df.train$previous)
describe(df.train[c("age", "balance", "duration", "campaign", "pdays", "previous")])


df.train$yact = ifelse(df.train$y == 'yes',1,0)
full.model <- glm(formula = yact ~ age + balance + duration + campaign + pdays + previous +
                    job + marital + education + default + housing + loan + poutcome, 
                  data=df.train, family = binomial)
summary(full.model)

# check for vif
fit <- lm(formula <- yact ~ age + balance + duration + campaign + pdays + previous +
            job + marital + education + default + housing + loan + poutcome, 
          data=df.train)
vif(fit)

backward <- step(full.model, direction = 'backward')
summary(backward)

# training probabilities and roc
df.train$prob = predict(full.model, type=c("response"))
class(df.train)
nrow(df.train)
q <- roc(y ~ prob, data = df.train)
plot(q)
auc(q)

# variable importance
varImp(full.model, scale = FALSE)

# confusion matrix on training set
df.train$ypred = ifelse(df.train$prob>=.5,'pred_yes','pred_no')
table(df.train$ypred,df.train$y)

#probabilities on test set
df.test$prob = predict(full.model, newdata = df.test, type=c("response"))

#confusion matrix on test set
df.test$ypred = ifelse(df.test$prob>=.5,'pred_yes','pred_no')
table(df.test$ypred,df.test$y)

#ks plot
ks_plot(actuals=df.train$y, predictedScores=df.train$ypred)

################### Proposed Model code begins ################

View(df.data)

# Loading df.data into df.data_final
df.data_final <- df.data
df.data_final$yact = ifelse(df.data$y == 'yes',1,0) #Loading 1s for 'yes' and 0s for 'no'
nrow(df.data_final)

#Removing every row with Not-Available entries
df.data_final <- df.data_final[!apply(df.data_final[,c("age", "balance", "duration", "campaign", "pdays", "previous", "job","marital", "education", "default", "housing", "loan", "poutcome")], 1, anyNA),]
nrow(df.data_final)
View(df.data_final)

set.seed(1234) # for reproducibility
df.data_final$rand <- runif(nrow(df.data_final))

#Training set = 90% of the entire data set #Test set = 10% of the entire data set
df.train_proposedmodel <- df.data_final[df.data_final$rand <= 0.9,]
df.test_proposedmodel <- df.data_final[df.data_final$rand > 0.9,]
nrow(df.train_proposedmodel)

#Building a tentative model-insignificant variables included
result_tentative_trainproposedmodel <- glm(formula = yact ~ age + balance + duration + campaign + pdays + previous +
                                      job + marital + education + default + housing + loan + poutcome, 
                                    data=df.train_proposedmodel, family = binomial)
summary(result_tentative_trainproposedmodel)

# The process of removing insignificant variables one at a time based on their p-values
# removing insignificant variables - 1) job unknown removed
df.train_proposedmodel_onlysig <- df.train_proposedmodel[df.train_proposedmodel$job!="unknown",]
result_tentative_trainproposedmodel_sig1 <- glm(formula = yact ~ age + balance + duration + campaign + pdays + previous +
                                           job + marital + education + default + housing + loan + poutcome, 
                                         data=df.train_proposedmodel_onlysig, family = binomial)

df.test_proposedmodel_onlysig <- df.test_proposedmodel[df.test_proposedmodel$job!="unknown",]

summary(result_tentative_trainproposedmodel_sig1)

# removing insignificant variables - 2) pdays removed
df.train_proposedmodel_onlysig$pdays <-NULL
result_tentative_trainproposedmodel_sig1 <- glm(formula = yact ~ age + balance + duration + campaign + previous +
                                           job + marital + education + default + housing + loan + poutcome, 
                                         data=df.train_proposedmodel_onlysig, family = binomial)


df.test_proposedmodel_onlysig$pdays <-NULL

summary(result_tentative_trainproposedmodel_sig1)

# removing insignificant variables - 3) marital status 'single' removed
df.train_proposedmodel_onlysig <- df.train_proposedmodel_onlysig[df.train_proposedmodel_onlysig$marital!="single",]
result_tentative_trainproposedmodel_sig1 <- glm(formula = yact ~ age + balance + duration + campaign + previous +
                                           job + marital + education + default + housing + loan + poutcome, 
                                         data=df.train_proposedmodel_onlysig, family = binomial)

df.test_proposedmodel_onlysig <- df.test_proposedmodel_onlysig[df.test_proposedmodel_onlysig$marital!="single",]

summary(result_tentative_trainproposedmodel_sig1)

# removing insignificant variables - 4) removing default altogether (because it holds only one value throughout)
df.train_proposedmodel_onlysig <- df.train_proposedmodel_onlysig[df.train_proposedmodel_onlysig$marital!="yes",]
result_tentative_trainproposedmodel_sig1 <- glm(formula = yact ~ age + balance + duration + campaign + previous +
                                           job + marital + education + housing + loan + poutcome, 
                                         data=df.train_proposedmodel_onlysig, family = binomial)

df.test_proposedmodel_onlysig <- df.test_proposedmodel_onlysig[df.test_proposedmodel_onlysig$marital!="yes",]

summary(result_tentative_trainproposedmodel_sig1)

# removing insignificant variables - 5) removing job 'management'
df.train_proposedmodel_onlysig <- df.train_proposedmodel_onlysig[df.train_proposedmodel_onlysig$job!="management",]
result_tentative_trainproposedmodel_sig1 <- glm(formula = yact ~ age + balance + duration + campaign + previous +
                                           job + marital + education + housing + loan + poutcome, 
                                         data=df.train_proposedmodel_onlysig, family = binomial)

df.test_proposedmodel_onlysig <- df.test_proposedmodel_onlysig[df.test_proposedmodel_onlysig$job!="management",]

summary(result_tentative_trainproposedmodel_sig1)

# removing insignificant variables - 6) removing poutcome 'other'
df.train_proposedmodel_onlysig <- df.train_proposedmodel_onlysig[df.train_proposedmodel_onlysig$poutcome!="other",]
result_tentative_trainproposedmodel_sig1 <- glm(formula = yact ~ age + balance + duration + campaign + previous +
                                           job + marital + education + housing + loan + poutcome, 
                                         data=df.train_proposedmodel_onlysig, family = binomial)

df.test_proposedmodel_onlysig <- df.test_proposedmodel_onlysig[df.test_proposedmodel_onlysig$poutcome!="other",]

summary(result_tentative_trainproposedmodel_sig1)

# removing insignificant variables - 7) removing job 'entrepreneur'
df.train_proposedmodel_onlysig <- df.train_proposedmodel_onlysig[df.train_proposedmodel_onlysig$job!="entrepreneur",]
result_tentative_trainproposedmodel_sig1 <- glm(formula = yact ~ age + balance + duration + campaign + previous +
                                           job + marital + education + housing + loan + poutcome, 
                                         data=df.train_proposedmodel_onlysig, family = binomial)

df.test_proposedmodel_onlysig <- df.test_proposedmodel_onlysig[df.test_proposedmodel_onlysig$job!="entrepreneur",]

summary(result_tentative_trainproposedmodel_sig1)

# removing insignificant variables - 8) removing education 'unknown'
df.train_proposedmodel_onlysig <- df.train_proposedmodel_onlysig[df.train_proposedmodel_onlysig$education!="unknown",]
result_tentative_trainproposedmodel_sig1 <- glm(formula = yact ~ age + balance + duration + campaign + previous +
                                           job + marital + education + housing + loan + poutcome, 
                                         data=df.train_proposedmodel_onlysig, family = binomial)

df.test_proposedmodel_onlysig <- df.test_proposedmodel_onlysig[df.test_proposedmodel_onlysig$education!="unknown",]

summary(result_tentative_trainproposedmodel_sig1)

# removing insignificant variables - 9) removing job 'student'
df.train_proposedmodel_onlysig <- df.train_proposedmodel_onlysig[df.train_proposedmodel_onlysig$job!="student",]
result_tentative_trainproposedmodel_sig1 <- glm(formula = yact ~ age + balance + duration + campaign + previous +
                                           job + marital + education + housing + loan + poutcome, 
                                         data=df.train_proposedmodel_onlysig, family = binomial)

df.test_proposedmodel_onlysig <- df.test_proposedmodel_onlysig[df.test_proposedmodel_onlysig$job!="student",]

summary(result_tentative_trainproposedmodel_sig1)

# removing insignificant variables - 10) removing job 'unemployed'
df.train_proposedmodel_onlysig <- df.train_proposedmodel_onlysig[df.train_proposedmodel_onlysig$job!="unemployed",]
result_tentative_trainproposedmodel_sig1 <- glm(formula = yact ~ age + balance + duration + campaign + previous +
                                           job + marital + education + housing + loan + poutcome, 
                                         data=df.train_proposedmodel_onlysig, family = binomial)

df.test_proposedmodel_onlysig <- df.test_proposedmodel_onlysig[df.test_proposedmodel_onlysig$job!="unemployed",]

summary(result_tentative_trainproposedmodel_sig1)


#no more insignificant variables left.

#Loading the final model into result_proposedmodel_sig1
result_proposedmodel_sig1 <- result_tentative_trainproposedmodel_sig1
class(result_proposedmodel_sig1)
print(result_proposedmodel_sig1)
plot(result_proposedmodel_sig1)

# Variable importance #
plot(result_proposedmodel_sig1)
varImp(result_proposedmodel_sig1, scale = FALSE)
# Variable importance #

# Limitations of this model: Interactions are excluded; Linearity of independent variables is assumed #

fit_proposedmodel <- lm(formula <- yact ~ age + balance + duration + campaign + previous +
                   job + marital + education + housing + loan + poutcome, 
                 data=df.train_proposedmodel_onlysig)
vif(fit_proposedmodel)

# automated variable selection - Backward
backward_proposedmodel <- step(result_proposedmodel_sig1, direction = 'backward')
summary(backward_proposedmodel)

# training probabilities and roc
result_proposedmodel_probs <- df.train_proposedmodel_onlysig
nrow(result_proposedmodel_probs)
class(result_proposedmodel_probs)
#Using the model made to make predictions in the column named 'prob'
result_proposedmodel_probs$prob = predict(result_proposedmodel_sig1, type=c("response"))
q_proposedmodel <- roc(y ~ prob, data = result_proposedmodel_probs)
plot(q_proposedmodel)
auc(q_proposedmodel)

# how the categorical variables are distributed and are related with target outcome
CrossTable(df.train_proposedmodel_onlysig$job, df.train_proposedmodel_onlysig$y)
CrossTable(df.train_proposedmodel_onlysig$marital, df.train_proposedmodel_onlysig$y)
CrossTable(df.train_proposedmodel_onlysig$education, df.train_proposedmodel_onlysig$y)
CrossTable(df.train_proposedmodel_onlysig$default, df.train_proposedmodel_onlysig$y)
CrossTable(df.train_proposedmodel_onlysig$housing, df.train_proposedmodel_onlysig$y)
CrossTable(df.train_proposedmodel_onlysig$loan, df.train_proposedmodel_onlysig$y)
CrossTable(df.train_proposedmodel_onlysig$poutcome, df.train_proposedmodel_onlysig$y)

# numerical variable distribution
hist(df.train_proposedmodel_onlysig$age)
hist(df.train_proposedmodel_onlysig$balance)
hist(df.train_proposedmodel_onlysig$duration)
hist(df.train_proposedmodel_onlysig$campaign)
hist(df.train_proposedmodel_onlysig$previous)

# confusion matrix on proposed model training set
# to check the accuracy of the model made by removing all the insignificant variables
result_proposedmodel_probs$ypred = ifelse(result_proposedmodel_probs$prob>=.5,'pred_yes','pred_no')
table(result_proposedmodel_probs$ypred,result_proposedmodel_probs$y)

#probabilities on test set
df.test_proposedmodel_onlysig$prob = predict(result_proposedmodel_sig1, newdata = df.test_proposedmodel_onlysig, type=c("response"))

#confusion matrix on test set
df.test_proposedmodel_onlysig$ypred = ifelse(df.test_proposedmodel_onlysig$prob>=.5,'pred_yes','pred_no')
table(df.test_proposedmodel_onlysig$ypred,df.test_proposedmodel_onlysig$y)

# ks plot #
ks_plot(actuals=result_proposedmodel_probs$y, predictedScores=result_proposedmodel_probs$ypred)


############### proposed code End #############
