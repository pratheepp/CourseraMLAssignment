### Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now
possible to collect a large amount of data about personal activity
relatively inexpensively. These type of devices are part of the
quantified self movement – a group of enthusiasts who take measurements
about themselves regularly to improve their health, to find patterns in
their behavior, or because they are tech geeks. One thing that people
regularly do is quantify how much of a particular activity they do, but
they rarely quantify how well they do it. In this project, your goal
will be to use data from accelerometers on the belt, forearm, arm, and
dumbell of 6 participants. They were asked to perform barbell lifts
correctly and incorrectly in 5 different ways. More information is
available from the website here:
<http://groupware.les.inf.puc-rio.br/har> (see the section on the Weight
Lifting Exercise Dataset).

### Description of the experiment and data

From: <http://groupware.les.inf.puc-rio.br/har>

Six young health participants were asked to perform one set of 10
repetitions of the Unilateral Dumbbell Biceps Curl in five different
fashions: exactly according to the specification (Class A), throwing the
elbows to the front (Class B), lifting the dumbbell only halfway (Class
C), lowering the dumbbell only halfway (Class D) and throwing the hips
to the front (Class E).

Class A corresponds to the specified execution of the exercise, while
the other 4 classes correspond to common mistakes. Participants were
supervised by an experienced weight lifter to make sure the execution
complied to the manner they were supposed to simulate. The exercises
were performed by six male participants aged between 20-28 years, with
little weight lifting experience. We made sure that all participants
could easily simulate the mistakes in a safe and controlled manner by
using a relatively light dumbbell (1.25kg).

### Overview of analysis

Since the testing sample does not have the dependant variable, the
training sample itself will be partitioned into training and validation
sub samples. The algorithm will learn on the training subsample and then
tested against the validation subsample, before being applied on the
testing sample, after assessment of the out of sample error.

To validate and cross check the prediction two different predictions
algorithms are used. C.50 will be applied first, followed by
randomForrest for comparison of results.

### Data

The training data for this project are available here:

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>

The test data are available here:

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>

The data for this project come from this source:
<http://groupware.les.inf.puc-rio.br/har>.

### Data procssing.

    #Load required libraries
    library(caret)
    library(kernlab)
    library(C50)
    library(randomForest)

    set.seed(12345)

#### Data loading and cleaning

    if (! file.exists('./pml-training.csv')) {
        download.file('http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv', destfile = './pml-training.csv')
    }
    if (! file.exists('./pml-testing.csv')) {
        download.file('http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv', destfile = './pml-testing.csv')
    }

Both test and training datasets were were downoaded to working directory
and scanned for NA values. The following were identified as na.strings:
"NA","\#DIV/0!",""

The first seven columns are not needed for prediction. They are removed.
Any columns with near zero variance are also removed.

    mltraining <- read.csv("./pml-training.csv", na.strings=c("NA","#DIV/0!",""))
    mltesting <- read.csv("./pml-testing.csv", na.strings=c("NA","#DIV/0!",""))


    mltraining <- mltraining[, -c(1:7)]
    mltesting <- mltesting[, -c(1:7)]

    # Remove rezo variance columns
    nearzerocols <- nearZeroVar(mltraining)
    mltraining <- mltraining[, -nearzerocols]
    mltesting <- mltesting[, -nearzerocols]

Keep variables with less than 40% of missing data.

    good_data_columns <- apply(mltraining,2, function(x) {  sum(is.na(x)) / length(x) < 0.4 }  )
    mltraining <- mltraining[, good_data_columns]
    mltesting <- mltesting[, good_data_columns]

    #Data after cleaned up
    dim(mltraining)

    ## [1] 19622    53

    dim(mltesting)

    ## [1] 20 53

    table(mltraining$classe)

    ## 
    ##    A    B    C    D    E 
    ## 5580 3797 3422 3216 3607

#### Data partitioning

The training data is partitioned into two subsamples: One is used for
training (60%) and the other used for validation and fine tuning of the
modal

    sub_training <- createDataPartition( mltraining$classe, p =0.6, list = F )
    sub_training_sample <- mltraining[sub_training, ]
    sub_validation_sample <- mltraining[-sub_training, ]
    dim(sub_training_sample)

    ## [1] 11776    53

    dim(sub_validation_sample)

    ## [1] 7846   53

#### Prediction

C50 decision tree algorithm will be used to train, with 10 trials. The
main advantage is interpretability and the algorithm is fairly
efficient. Later, the predicted results will be cross checked against
random Forest.

summary(c50fit) will print the entire C50 tree with additional
information. Since the output is too long for this report, it's not
shown here.

    sub_training_sample_c50 <- sub_training_sample[, -53]
    c50fit <- C5.0(sub_training_sample_c50, sub_training_sample$classe, trials = 10)
    #Summary output will be too long
    c50fit

    ## 
    ## Call:
    ## C5.0.default(x = sub_training_sample_c50, y =
    ##  sub_training_sample$classe, trials = 10)
    ## 
    ## Classification Tree
    ## Number of samples: 11776 
    ## Number of predictors: 52 
    ## 
    ## Number of boosting iterations: 10 
    ## Average tree size: 217.8 
    ## 
    ## Non-standard options: attempt to group attributes

Make a prediction on the training data, followed by the validation
sample. Comparing the two confusion matrices gives an indication of
model performance

    c50trainPred <- predict(c50fit, sub_training_sample)
    confusionMatrix(c50trainPred, sub_training_sample$classe)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 3348    0    0    0    0
    ##          B    0 2279    0    0    0
    ##          C    0    0 2054    0    0
    ##          D    0    0    0 1930    0
    ##          E    0    0    0    0 2165
    ## 
    ## Overall Statistics
    ##                                      
    ##                Accuracy : 1          
    ##                  95% CI : (0.9997, 1)
    ##     No Information Rate : 0.2843     
    ##     P-Value [Acc > NIR] : < 2.2e-16  
    ##                                      
    ##                   Kappa : 1          
    ##  Mcnemar's Test P-Value : NA         
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
    ## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
    ## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
    ## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
    ## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
    ## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1838
    ## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
    ## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000

Insample accurcy is 1. Sample error is 0

    c50ValidationPred <- predict(c50fit, sub_validation_sample)
    confusionMatrix(c50ValidationPred, sub_validation_sample$classe)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 2224   11    0    0    1
    ##          B    7 1493   12    2    2
    ##          C    1   13 1350   19    4
    ##          D    0    0    5 1259    0
    ##          E    0    1    1    6 1435
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9892          
    ##                  95% CI : (0.9866, 0.9913)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9863          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9964   0.9835   0.9868   0.9790   0.9951
    ## Specificity            0.9979   0.9964   0.9943   0.9992   0.9988
    ## Pos Pred Value         0.9946   0.9848   0.9733   0.9960   0.9945
    ## Neg Pred Value         0.9986   0.9961   0.9972   0.9959   0.9989
    ## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
    ## Detection Rate         0.2835   0.1903   0.1721   0.1605   0.1829
    ## Detection Prevalence   0.2850   0.1932   0.1768   0.1611   0.1839
    ## Balanced Accuracy      0.9971   0.9899   0.9906   0.9891   0.9969

#### Sample error

Out of sample error is very small, 0.01. ( 1 - Accuracy,0.9892)

The model will now be used on the testing dataset to predict the
outcome.

    c50TestPred <- predict(c50fit, mltesting)
    c50TestPred 

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E

To double check the results, a random forest will be built against the
training subset and applied on the test sample.

    rffit <- randomForest(classe ~ ., data = sub_training_sample,  ntrees = 10)
    #rfValidationPred <- predict(rffit, sub_validation_sample)
    #confusionMatrix(rfValidationPred, sub_validation_sample$classe)
    rffit.pred <- predict( rffit ,  mltesting)
    rffit.pred

    ##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
    ##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
    ## Levels: A B C D E

### Conclusion

Final predcited result is

    c50TestPred

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E

Both, random Forest and C50 predicted the same result, an indication the
prediction was highly accurate.
