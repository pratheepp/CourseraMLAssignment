suppressWarnings(library(caret))
suppressWarnings(library(kernlab))
suppressWarnings(library(C50))
suppressWarnings(library(randomForest))

set.seed(12345)

if (! file.exists('./pml-training.csv')) {
  download.file('http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv', destfile = './pml-training.csv')
}
if (! file.exists('./pml-testing.csv')) {
  download.file('http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv', destfile = './pml-testing.csv')
}

mltraining <- read.csv("./pml-training.csv", na.strings=c("NA","#DIV/0!",""))
mltesting <- read.csv("./pml-testing.csv", na.strings=c("NA","#DIV/0!",""))
mltraining <- mltraining[, -c(1:7)]
mltesting <- mltesting[, -c(1:7)]

nearzerocols <- nearZeroVar(mltraining)
mltraining <- mltraining[, -nearzerocols]
mltesting <- mltesting[, -nearzerocols]

good_data_columns <- apply(mltraining,2, function(x) {  sum(is.na(x)) / length(x) < 0.4 }  )
mltraining <- mltraining[, good_data_columns]
mltesting <- mltesting[, good_data_columns]

#Data after cleaned up
dim(mltraining)
dim(mltesting)
table(mltraining$classe)

sub_training <- createDataPartition( mltraining$classe, p =0.6, list = F )
sub_training_sample <- mltraining[sub_training, ]
sub_validation_sample <- mltraining[-sub_training, ]
dim(sub_training_sample)
dim(sub_validation_sample)


sub_training_sample_c50 <- sub_training_sample[, -53]
c50fit <- C5.0(sub_training_sample_c50, sub_training_sample$classe, trials = 10)
#Summary output will be too long
c50fit

c50trainPred <- predict(c50fit, sub_training_sample)
confusionMatrix(c50trainPred, sub_training_sample$classe)
c50ValidationPred <- predict(c50fit, sub_validation_sample)
confusionMatrix(c50ValidationPred, sub_validation_sample$classe)

c50TestPred <- predict(c50fit, mltesting)
c50TestPred


rffit <- randomForest(classe ~ ., data = sub_training_sample,  ntrees = 10)
#rfValidationPred <- predict(rffit, sub_validation_sample)
#confusionMatrix(rfValidationPred, sub_validation_sample$classe)
rffit.pred <- predict( rffit ,  mltesting)
rffit.pred








