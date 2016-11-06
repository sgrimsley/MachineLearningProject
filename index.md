# Machine Learning Project
S Grimsley  
November 1, 2016  

The purpose of this project is to develop a model to accurately predict how participants performed a Dumbbell Biceps Curl, using the Weightlifting Exercises Dataset.  Exercises were separated into 5 classes; class A is the correct form, and B - D are different incorrect forms as defined by the experimenters.

Exploratory analysis led to the reduction of the data to 54 variables (from 160).  The training data was split into a training (75%) and test set, and a random forest model developed using k-fold cross validation (k=10).  The model performed well when applied to the test set, with an accuracy of 99.35% and Kappa of 99.17%.  The model accurately predicted all 20 test cases from the test dataset.

Data Citation:
Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013. 

URL: http://groupware.les.inf.puc-rio.br/har


###Data Processing / Exploratory Analysis

Training and test data were downloaded from:

* Training: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
* Test: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv


```r
library(caret)
train <- read.csv("pml-training.csv", stringsAsFactors=FALSE)
names <- colnames(train)
```

Dimensions in training set:

```r
dim(train)
```

```
## [1] 19622   160
```

Exploratory analysis of the data showed 100 variables had values that were primarily N/A or empty strings (97.9% within these variable).  Review of the experimenter's methodology show these were computed values, based on time slices of 2.5 seconds, with results stored in the record at the beginning of each time window.  These variables were removed, instead of imputing the data, because a similar imputation would not be possible on the final test data.  

I considered whether a time-series model was appropriate, but ruled that out because the test data was not structured in that manner (isolated records were provided, instead of entire time windows).

The data was ordered sequentially, both by number (variable X) and time window measures (timestamps, windows).  I removed these variables after column X proved to be the most important variable (varImp of 100%) in an earlier version of the model.


```r
# Testing for and removing high proportion of NAs
isna <- lapply(train[,],function(x)sum(is.na(x))/length(train$classe))
remove<- isna[names(which(isna >0))]
train2 <- train[-which(colnames(train) %in% names(remove))]

# Test for and removing high proportion of empty character strings
isna2<-lapply(train2[,],function(x)sum(x=="")/length(train2$classe))
remove2<- isna2[names(which(isna2 >0))]
train2 <- train2[-which(colnames(train2) %in% names(remove2))]

# Remove ordered variables (row numbers and time window measures)
remove3 <- c(1,3:7)
train2 <- train2[,-remove3]
```

After processing, the training dataset contained 54 variables, instead of 160.  

```r
dim(train2)
```

```
## [1] 19622    54
```

As an alternative, I considered removing variables with near zero variance.  Near zero variance would have only removed 60 variables, so I decided to use the method described above.


```r
nsv <- nearZeroVar(train, saveMetrics=TRUE)
length(nsv$nzv[which(nsv$nzv==TRUE)])
```

```
## [1] 60
```

The training data was split into training (75%) and test sets.


```r
# Separate into training and test sets
set.seed(6483147)
inTrain <- createDataPartition(y=train2$classe, p=0.75, list=FALSE)
training <- train2[inTrain,]
testing <- train2[-inTrain,]
```

###Model Development

The caret package was used to fit a random forest model on the training set, using k-fold cross validation (k = 10) and parallel processing.


```r
library(parallel)
library(doParallel)
cluster <- makeCluster(detectCores() -1)
registerDoParallel(cluster)

# set trainControl to use k-fold cross validation (k=10) and allow parallel processing
fitControl <- trainControl(method = "cv", number = 10, allowParallel=TRUE)

# fit random forest model
modFit <- train(classe ~., data=training, method="rf", trControl=fitControl)

stopCluster(cluster)
```

The model performed with a 99.36% accuracy, and Kappa of 99.19%.


```r
modFit
```

```
## Random Forest 
## 
## 14718 samples
##    53 predictors
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 13247, 13245, 13247, 13245, 13248, 13245, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9920505  0.9899432
##   29    0.9936131  0.9919204
##   57    0.9851200  0.9811754
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 29.
```

```r
plot(modFit)
```

![](index_files/figure-html/unnamed-chunk-8-1.png)<!-- -->

The model was applied to the testing data, resulting in an accuracy of 99.35% and Kappa of 99.17%.  These measures are more accurate for how the model will perform on an independant data set.  The out of sample error rate is 0.65%.


```r
pred <- predict(modFit, newdata=testing)
confusionMatrix(pred, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1394    5    0    0    0
##          B    1  944    7    1    1
##          C    0    0  843   10    2
##          D    0    0    5  793    0
##          E    0    0    0    0  898
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9935          
##                  95% CI : (0.9908, 0.9955)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9917          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9993   0.9947   0.9860   0.9863   0.9967
## Specificity            0.9986   0.9975   0.9970   0.9988   1.0000
## Pos Pred Value         0.9964   0.9895   0.9860   0.9937   1.0000
## Neg Pred Value         0.9997   0.9987   0.9970   0.9973   0.9993
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2843   0.1925   0.1719   0.1617   0.1831
## Detection Prevalence   0.2853   0.1945   0.1743   0.1627   0.1831
## Balanced Accuracy      0.9989   0.9961   0.9915   0.9925   0.9983
```

Out of sample error rate of 0.65%:

```r
sum(pred != testing$classe)/length(pred)*100
```

```
## [1] 0.6525285
```


###Results
The test data was input and processed in the same manner as the training data (removing variables), and predictions generated from the model.  


```r
test <- read.csv("pml-testing.csv", stringsAsFactors=FALSE)

# Remove same columns as the testing data
test2 <- test[-which(colnames(test) %in% names(remove))]
test2 <- test2[-which(colnames(test2) %in% names(remove2))]  
test2 <- test2[,-remove3]

# Predictions for quiz
predquiz <- predict(modFit, newdata=test2)
```

The predictions were 100% accurate for the test set, based on the results of the project quiz.  This shows the model is accurate at predicting how participants performed a Dumbbell Biceps curl.  Since user_name was not a variable of importance (see below), the model should be able to predict the classe for people not included in the initial study.  (It would be interesting to test this.) 


```r
varImp(modFit)
```

```
## rf variable importance
## 
##   only 20 most important variables shown (out of 57)
## 
##                      Overall
## roll_belt             100.00
## pitch_forearm          58.38
## yaw_belt               53.29
## pitch_belt             44.92
## magnet_dumbbell_z      44.02
## magnet_dumbbell_y      42.75
## roll_forearm           42.06
## accel_dumbbell_y       21.56
## roll_dumbbell          19.57
## magnet_dumbbell_x      19.02
## accel_forearm_x        18.30
## magnet_belt_z          17.14
## accel_dumbbell_z       15.71
## accel_belt_z           15.08
## total_accel_dumbbell   14.62
## magnet_forearm_z       14.40
## magnet_belt_y          13.54
## yaw_arm                12.20
## gyros_belt_z           11.80
## magnet_belt_x          10.85
```

