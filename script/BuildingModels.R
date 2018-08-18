rm(list = ls())

require(MASS)
library(heuristica)
library(caret)
library(plyr) 
library(glmnet)
require(randomForest)
library(kknn)
library(ISLR)
library(base)
library(plotROC)
library(ROCR)
library(gplots)
library(pROC)
library(ggplot2)
library(corrplot)
library(vcd)

sm.d = read.csv("../data/student-mat.csv",header=TRUE)
lapply(sm.d,class)
dim(sm.d)

#Check for missing values in the dataset
sapply(sm.d,function(x) sum(is.na(x)))

#Creating the pass fail binary variable - Gnew from final grade G3(range 0-20)
sm.d <- within(sm.d, Gnew <- 1)
sm.d[(sm.d$G3>=0 & sm.d$G3<=10), "Gnew"] <- 0 

#Converting Fathers and Mothers eduction to factor variables
sm.d[c("Fedu","Medu")] <- lapply(sm.d[c("Fedu","Medu")], as.factor)

#Removing grades column from dataset
sm.d <- sm.d[c(-31,-32,-33)]
#sm.dd <- sm.d[c(-31)]
#Splitting dataset into training and test set
set.seed(100)
tr = sample(1:395,320)


##################   EDA    ##################################

student_mat  = read.csv("Desktop/student-mat.csv",header=TRUE)
attach(student_mat)
Final = (student_mat$G3)
boxplot(Final ~studytime, data = student_mat,xlab= 'Study Time',ylab = "Grade",col='orange',lncol='blue')
boxplot(Final ~failures, data = student_mat,xlab= 'Failures',ylab = "Grade",col='lightblue',lncol='blue')
boxplot(Final ~Mjob, data = student_mat,xlab= "Mother's Occupation",ylab = "Grade",col='limegreen',lncol='blue')
boxplot(Final ~internet, data = student_mat,xlab= 'Internet at Home',ylab = "Grade",col='pink',lncol='blue')
boxplot(Final ~address, data = student_mat,xlab= 'Home Location',ylab = "Grade",col='brown1',lncol='blue')
boxplot(Final ~goout, data = student_mat,xlab= 'Go-Out',ylab = "Grade",col='brown1',lncol='blue')
boxplot(Final ~absences, data = student_mat,xlab= 'Absences',ylab = "Grade",col='yellow',lncol='blue')

hist(Final, col='lightsalmon',xlab='Grades',ylab='Students',main= 'Total Grade Distribution')



# test data
final <- factor(c(Medu))
levels(final)

test <- factor(c(Fedu))
levels(test)    
## [1] "Cancer"

# gives expected error
model <- confusionMatrix(final, test)
#plot(model$table)
model
model$table

myTable <- model$table
plot(myTable)

#plot(myTable,col= '2',xlab='Pstatus',ylab='famsup')
mosaicplot(myTable,shade = T,xlab="Mother Education",ylab='Father Education',direction = 'v',main ='')


# test data
final <- factor(c(schoolsup))
levels(final)
## [1] "Cancer" "Normal"

test <- factor(c(romantic))
levels(test)    
## [1] "Cancer"

# gives expected error
model <- confusionMatrix(final, test)
#plot(model$table)
model
model$table

myTable <- model$table
plot(myTable)

fourfoldplot(model$table)



###########                Interaction of Variables                 ###########

dat <- as.data.frame(sapply(sm.d[,c(-31)], as.numeric))
Gnew = as.numeric(sm.d$Gnew)

XXca = NULL

set.seed(100)
XXca <- model.matrix(Gnew~.*schoolsup, data=data.frame(scale(dat)))[,c(-1,-31)]
#xtab<-table(sm.d$age,sm.d$Gnew)
#confusionMatrix(xtab)

#what is this:
STUdata = data.frame(Gnew,XXca)
STU_pred = as.data.frame(STUdata)

null = glm(Gnew~1, data=STUdata[tr,])
full = glm(Gnew~., data=STUdata[tr,])

regBoth = step(null, scope=formula(full), direction="both", k=log(length(tr)))
summary(regBoth)
STUdata = scale(STUdata)

predicted_IntVar <- predict(regBoth,STU_pred[-tr,],type='response')
predicted_IntVar <- ifelse(predicted_IntVar > 0.5,1,0)
misClasificError <- mean(predicted_IntVar != Gnew[-tr])

#MCR <- 1-sum(diag(kknn_pred))/sum(kknn_pred)

print(paste('Accuracy',1-misClasificError)) #38%
cm <- confusionMatrix(data=as.factor(predicted_IntVar), reference=as.factor(Gnew[-tr]))
print(paste('fpr',cm$table[2]/(nrow(sm.d[-tr,])))) #0.30
print(paste('fnr',cm$table[3]/(nrow(sm.d[-tr,])))) #0.05




########                   LOGISTIC REGRESSION                      #########
#reading Gnew as a factor
sm.d$Gnew = as.factor(sm.d$Gnew)
train = sm.d[tr,]
test = sm.d[-tr,]


set.seed(100)
#Full model - removing all grades(G1, G2 and G3) from predictors
model <- glm(Gnew~.,family=binomial(link='logit'),data=train)
#Null model
model_NULL <- glm(Gnew~1,family=binomial(link='logit'),data=train)
#Stepwise feature selection starting from Null model
step_glm_null <- stepAIC(model_NULL, scope=formula(model),direction="both")

summary(model) #Inportant variables - SexM, failures, schoolsupyes, famsupYes, nurseryYes)
summary(step_glm_null) #Important variables - failures, schoolsupyes, famsupyes,   sexM, MjobServices, nurseryYes, MjobHealth, age, Walc )


#Predict and check accuracy using a logistic regression on all variables
test$predicted_glm <- predict(model,test,type='response')
test$predicted_glm <- ifelse(test$predicted_glm > 0.5,1,0)
misClasificError <- mean(test$predicted_glm != test$Gnew)
print(paste('Accuracy',1-misClasificError)) #65%
cm1 <- confusionMatrix(data=as.factor(test$predicted_glm), reference=test$Gnew)
print(paste('fpr',cm1$table[2]/(nrow(test)))) #0.32
print(paste('fnr',cm1$table[3]/(nrow(test)))) #0.03

#Predict and check accuracy using null stepwise regression
test$predicted_glm_null <- predict(step_glm_null,test,type='response')
test$predicted_glm_null <- ifelse(test$predicted_glm_null > 0.5,1,0)
misClasificError <- mean(test$predicted_glm_null != test$Gnew)
print(paste('Accuracy',1-misClasificError))    #68%
cm2 <- confusionMatrix(data=as.factor(test$predicted_glm_null), reference=test$Gnew)
print(paste('fpr',cm2$table[2]/(nrow(test)))) #0.29
print(paste('fnr',cm2$table[3]/(nrow(test))))  #0.03



#########################         LASSO              ######################

XXstu <- model.matrix(Gnew~., data=sm.d)[,-1]
y <- ifelse(sm.d$Gnew=="1",1,0)
set.seed(999)
lasso <- cv.glmnet(XXstu,y,alpha=1,family="binomial",type.measure = "class" ) #class gives misclassification error.
plot(lasso)


#min value of lambda
lambda_min <- lasso$lambda.min
#best value of lambda
lambda_1se <- lasso$lambda.1se
#regression coefficients
(coef(lasso,s=lambda_1se))  #Important variables - Sex, Age, AddressU, Fedu4, MjobHealth, MjobSerives, traveltime, failures, schoolsupYes, higherYes, goout, WalC)

x_test <- model.matrix(Gnew~., data=test)[,c(-1,-47,-48)]
test$predicted_lasso <- predict(lasso,newx = x_test,s=lambda_1se,type="response")
test$predicted_lasso <- ifelse(test$predicted_lasso > 0.5,1,0)

misClasificError <- mean(test$predicted_lasso != test$Gnew)
print(paste('Accuracy',1-misClasificError))  #accuracy 64% with 1se
cm4 <- confusionMatrix(data=as.factor(test$predicted_lasso), reference=test$Gnew)
print(paste('fpr',cm4$table[2]/(nrow(test)))) #0.35
print(paste('fnr',cm4$table[3]/(nrow(test)))) #0.01

######################        K fold validation            #####################

kcv = 10
n=nrow(sm.d)
n0 = round(n/kcv,0)
fpr = NULL
fnr = NULL
out_acc = NULL

used = NULL
set = 1:n
set.seed(150)
for(j in 1:kcv){
  
  if(n0<length(set)){val = sample(set,n0)}
  if(n0>=length(set)){val=set}
  
  train_i = sm.d[-val,]
  test_i = sm.d[val,]
  
  # Predict results
  results_kfold2 <- predict(step_glm_null,test_i,type='response')
  results_kfold2 <- ifelse(results_kfold2 > 0.5,1,0)
  # Actual answers
  answers <- test_i$Gnew
  # Accuracy calculation
  misClasificError <- mean(answers != results_kfold2)
  # Collecting results
  out_acc[j] <- 1-misClasificError
  
  used = union(used,val)
  set = (1:n)[-used]
  
  # Confusion matrix
  cm <- confusionMatrix(data=as.factor(results_kfold2), reference=answers)
  fpr[j] <- cm$table[2]/(nrow(test_i))
  fnr[j] <- cm$table[3]/(nrow(test_i))
  cat(j,'\n')
}

print(mean(out_acc)) #71
print(mean(fpr)) #0.19
print(mean(fnr)) #0.09

#plot(fpr), fnr)

#roc.plot(mean(fpr), mean(fnr))

hist(out_acc,xlab='Accuracy',ylab='Freq',
     col='cyan',border='blue',density=30)
hist(fpr,xlab='% of fnr',ylab='Freq',main='FPR',
     col='cyan',border='blue',density=30)
hist(fnr,xlab='% of fnr',ylab='Freq',main='FNR',
     col='cyan',border='blue',density=30)






##############Random Forest################

#number of trees plot, boostrap data with replacement 


set.seed(123)
rf <- randomForest(Gnew~., data=train,
                   ntree = 500,
                   mtry = 8,#select 8 variables at a time
                   importance = TRUE,
                   proximity = TRUE)
print(rf)
attributes(rf)

# Prediction & Confusion Matrix - train data
library(caret)
p1 <- predict(rf, train)
confusionMatrix(p1, train$Gnew)

# Prediction & Confusion Matrix - test data
p2 <- predict(rf, test)
confusionMatrix(p2, test$Gnew)

# Error rate of Random Forest
plot(rf)


# Variable Importance
varImpPlot(rf,
           sort = T,
           n.var = 8,
           main = "Top 5 - Variable Importance")
importance(rf)
varUsed(rf)




##########                         knn                   ###############

kknn_mod <- kknn(Gnew ~ failures + schoolsup + Mjob + age + nursery + Walc + sex + Fedu + famsup + higher, train, test)
kknn_pred <- table(kknn_mod$fitted, test$Gnew)
ConfusionM <- confusionMatrix(data = as.factor(kknn_mod$fitted), reference = test$Gnew)
ConfusionM

print(paste('fpr',ConfusionM$table[2]/(nrow(test)))) #0.32
print(paste('fnr',ConfusionM$table[3]/(nrow(test)))) #0.03
kknn_pred

tpr_kknn <- kknn_pred[2,2]/(kknn_pred[2,2] + kknn_pred[1,2])
tpr_kknn
MCR <- 1-sum(diag(kknn_pred))/sum(kknn_pred) 
Accuracy <- 1-MCR
Accuracy


#K-Fold CV for Knn Model
set.seed(100)
kcv <- 10

#num points to take out into each fold
n0 <- round(nrow(sm.d)/kcv,0)

#initialize matrix for MCR of every fold, checking 100 neighbors
out_MCR <- matrix(0,kcv,100)

used = NULL
fpr1 = NULL
fnr1 = NULL
set = 1:nrow(sm.d)

for(j in 1:kcv){
  
  if(n0<length(set)){val = sample(set,n0)}
  if(n0>=length(set)){val=set}
  
  train_i = sm.d[-val,]
  test_i = sm.d[val,]
  
  for(i in 1:100){
    kknn_mod <- kknn(Gnew ~ failures + schoolsup + Mjob + age + nursery + Walc + sex + Fedu + famsup + higher, train_i, test_i, k = i, kernel = "rectangular")
    
    
    kknn_pred <- table(kknn_mod$fitted, test_i$Gnew)
    #kknn_mod$fitted <- ifelse(kknn_mod$fitted >0.7, 1, 0)   ##########TAKE NOTE OF THIS
    pred11 <- predict(kknn_mod, test_i, type = 'prob')[,-1]
    pred11<- ifelse(pred11 >0.7, 1, 0)
    
    MCR <- 1-sum(diag(kknn_pred))/sum(kknn_pred)
    out_MCR[j,i] = MCR
    

    # Confusion matrix
    ConfusionM1 <- confusionMatrix(data=as.factor(pred11), reference=test_i$Gnew)
    fpr1[j] <- ConfusionM1$table[2]/(nrow(test_i))
    fnr1[j] <- ConfusionM1$table[3]/(nrow(test_i))
    cat(j,'\n')
    
    
  }
  
  used = union(used,val)
  set = (1:nrow(sm.d))[-used]
  
  cat(j,'\n')
  
}


mMCR = apply(out_MCR,2,mean)   #computes the average of every col in the matrix
mMCR
best = which.min(mMCR)
best
mMCR[best]
1-mMCR[best]
print(mean(fpr1)) #0.19
print(mean(fnr1)) #0.09


#k=21, accuracy = 68.18%

#Now refit model and predict on original train and test

kknn_mod_new <- kknn(Gnew ~ failures + schoolsup + Mjob + age + nursery + Walc + sex + Fedu + famsup + higher, train, test, k = 21, kernel = "rectangular")
kknn_mod_new_7 <- predict(kknn_mod_new, test, type = 'prob')[,-1]
kknn_mod_new_7<- ifelse(kknn_mod_new_7 >0.7, 1, 0)

nrow(test)
pred1 <- predict(kknn_mod_new, test, type = 'prob')[,-1]
plot.roc(test$Gnew, pred1, col = "green", lwd = 3, print.auc = TRUE, print.auc.x = 0.2, print.auc.y = 0.6, smooth = TRUE)


pred2 <- predict(step_glm_null, test, type = 'response')
plot.roc(test$Gnew, add = T, pred2, col = "red", lwd = 3, print.auc = TRUE, print.auc.x = 0.2, print.auc.y = 0.7, smooth = TRUE)


pred3 <- predict(rf, test, type = 'prob')[,-1]
plot.roc(test$Gnew, add = T, pred3, col = "blue", lwd = 3, print.auc = TRUE, print.auc.x = 0.2, print.auc.y = 0.55, smooth = TRUE)

pred4<-predict(lasso,newx = x_test,s=lambda_1se,type="response")
plot.roc(test$Gnew, add = T, pred4, col = "orange", lwd = 3, print.auc = TRUE, print.auc.x = 0.2, print.auc.y = 0.65, smooth = TRUE)

abline(1, -1, col= "black", lwd = 4)
legend("topleft", legend = c("Logistic Regression","Lasso", "KNN", "Random Forest") , pch = 15, bty = 'n', col = c("red","orange", "green", "blue"))






#Reading the survey responses and changing the variables class appropriately
sm.d1 = read.csv("/Users/alysonbrown/Desktop/Survey Responses.csv",header=TRUE)
sm.d1[c("Fedu")] <- lapply(sm.d1[c("Fedu")], as.factor)
sm.d1$Gnew <- NA
sm.d1[c("Gnew")] <- lapply(sm.d1[c("Gnew")], as.factor)

#subsetting training data only for the important variables collected in the survey
train1 <- train[,c("age","sex","failures","schoolsup","famsup", "Mjob","Fedu","nursery","Walc","higher","absences","address","Gnew")]

#combining training and test and then detaching again - because random forest prediction won't work otherwise
new_data = rbind(train1,sm.d1)
train_survey = new_data[(1:320),]
test_survey = new_data[(321:nrow(new_data)),]

#training the model on subsetted data
set.seed(123)
rf3 <- randomForest(Gnew~failures + absences + Fedu + Mjob +schoolsup + sex+ age+famsup+nursery+Walc+higher+address,
                    data=train_survey,
                    ntree = 500,
                    mtry = 10,#select 10 variables at a time
                    importance = TRUE,
                    proximity = TRUE)

#predicting on survey results
p23 <- predict(rf3, test_survey)
p23

