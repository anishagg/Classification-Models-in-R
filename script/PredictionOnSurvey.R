
#Reading the survey responses and changing the variables class appropriately
sm.d1 = read.csv("../data/Survey Responses.csv",header=TRUE)
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

