### Final project 
## Jiaming Huang - jh3777 - Data mining   digit classification


rm(list = ls())
#cran <- getOption("repos")
#cran["dmlc"] <- "https://s3-us-west-2.amazonaws.com/apache-mxnet/R/CRAN/"
#options(repos = cran)
#install.packages("mxnet",dependencies = T)
## First install "bindrcpp" package
library(mxnet)


full <- read.csv("train.csv")
#Sample Indexes
set.seed(0)
indexes <- sample(1:nrow(full), size=0.2*nrow(full))
# Split data
test <- full[indexes,]
dim(test)  # 8400 785
train <- full[-indexes,]
dim(train) # 33600 785
test <- data.matrix(test)
train <- data.matrix(train)

head(full[1:10])

barplot(table(test[,1]), col=rainbow(10, 0.5), main="n Digits in Test")

# convert digit labels to factor for classification
#train[,1] <- as.factor(train[,1])
#Split the train and test dataset to x and y
train.x <- train[,-1]
train.y <- train[,1]
test.x <- test[,-1]
test.y <- data.matrix(test[,1])

# Create a 28*28 matrix with pixel color values
m <- matrix(unlist(train.x[6,]),nrow = 28,byrow = T)
# Plot that matrixpp
image(m,col=grey.colors(255))

rotate <- function(x){
  t(apply(x, 2, rev))
} # reverses (rotates the matrix)

par(mfrow=c(3,3))
lapply(1:9, 
       function(x){image(
         rotate(matrix(unlist(train[x,-1]),nrow = 28,byrow = T)),
         col=grey.colors(255),
         xlab=full[x,1]
       )
       })
par(mfrow=c(1,1))

train.x <- t(train.x/255)
test.x <- t(test.x/255)



##############################################
## Build a shallow neural network ###########
#############################################
m0.data <- mx.symbol.Variable("data") # Notice how each layer is passed to the next 
m0.fc1 <- mx.symbol.FullyConnected(m0.data, name="fc1", num_hidden=128)
m0.act1 <- mx.symbol.Activation(m0.fc1, name="activation1", act_type="relu")
m0.fc2 <- mx.symbol.FullyConnected(m0.act1, name="fc2", num_hidden=10)
m0.softmax <- mx.symbol.SoftmaxOutput(m0.fc2, name="softMax")

## decide which device to use
devices <- mx.cpu()

## train the neural network
log0 <- mx.metric.logger$new() #to keep track of the results of each iterration
tick <- proc.time() #mark the start time

mx.set.seed(0)
m0 <- mx.model.FeedForward.create(m0.softmax,  #the network configuration made above
                                  X = train.x, #the predictors
                                  y = train.y, #the labels
                                  #ctx = mx.cpu(),
                                  num.round = 60, # The kernel can only handle 1 (I suggest ~50ish to start)
                                  array.batch.size = 100,
                                  array.layout="colmajor",
                                  learning.rate = 0.001,
                                  momentum = 0.95,
                                  eval.metric = mx.metric.accuracy,
                                  initializer = mx.init.uniform(0.07),
                                  epoch.end.callback = mx.callback.log.train.metric(1,log0)
)
print(paste("Training took:", round((proc.time() - tick)[3],2),"seconds"))
## plot the train accuracy 
plot(log0$train, type="l", col="black", xlab="Iteration", ylab="Accuracy", lwd = 2,
     main = "SNN with only 1 hidden layers")

## make prediction
m0.preds <- predict(m0, test.x, array.layout = "colmajor")
t(round(m1.preds[,1:5], 2))

m0.preds.value <- max.col(t(m0.preds)) - 1
m0.preds.value[1:5]

## computing misclassification error
test.y.preds <- data.matrix(m0.preds.value)
d <- 0
for (i in 1:8400){
  if (test.y[i,] != test.y.preds[i,]){
    d = d+1
  }
}
(misclassification1 <- d/8400)


##############################################
### Building a deep neural neural network ####
##############################################
m1.data <- mx.symbol.Variable("data") # Notice how each layer is passed to the next 
m1.fc1 <- mx.symbol.FullyConnected(m1.data, name="fc1", num_hidden=128)
m1.act1 <- mx.symbol.Activation(m1.fc1, name="activation1", act_type="relu")
m1.fc2 <- mx.symbol.FullyConnected(m1.act1, name="fc2", num_hidden=64)
m1.act2 <- mx.symbol.Activation(m1.fc2, name="activation2", act_type="relu")
m1.fc3 <- mx.symbol.FullyConnected(m1.act2, name="fc3", num_hidden=32)
m1.act3 <- mx.symbol.Activation(m1.fc3, name="activation3", act_type="relu")
m1.fc4 <- mx.symbol.FullyConnected(m1.act3, name="fc4", num_hidden=10)
m1.softmax <- mx.symbol.SoftmaxOutput(m1.fc4, name="softMax")

## decide which device to use
devices <- mx.cpu()

## train the neural network
log1 <- mx.metric.logger$new() #to keep track of the results of each iterration
tick <- proc.time() #mark the start time

mx.set.seed(0)
m1 <- mx.model.FeedForward.create(m1.softmax,  #the network configuration made above
                                  X = train.x, #the predictors
                                  y = train.y, #the labels
                                  #ctx = mx.cpu(),
                                  num.round = 60, # The kernel can only handle 1 (I suggest ~50ish to start)
                                  array.batch.size = 100,
                                  array.layout="colmajor",
                                  learning.rate = 0.001,
                                  momentum = 0.95,
                                  eval.metric = mx.metric.accuracy,
                                  initializer = mx.init.uniform(0.07),
                                  epoch.end.callback = mx.callback.log.train.metric(1,log1)
)
print(paste("Training took:", round((proc.time() - tick)[3],2),"seconds"))
## plot the train accuracy 
plot(log1$train, type="l", col="red", xlab="Iteration", ylab="Accuracy", lwd = 2,
     main = "DNN with 3 hidden layers")

## make prediction
m1.preds <- predict(m1, test.x, array.layout = "colmajor")
t(round(m1.preds[,1:5], 2))

m1.preds.value <- max.col(t(m1.preds)) - 1
m1.preds.value[1:5]

## computing misclassification error
test.y.preds <- data.matrix(m1.preds.value)
a <- 0
for (i in 1:8400){
  if (test.y[i,] != test.y.preds[i,]){
    a = a+1
  }
}
(misclassification1 <- a/8400)


############################################################################
### fit convolutionla neural network with four fully-connected layers ######
############################################################################
m2.data <- mx.symbol.Variable("data")

# 1st convolutional layer
m2.conv1 <- mx.symbol.Convolution(m2.data, kernel=c(5,5), num_filter=16)
m2.bn1 <- mx.symbol.BatchNorm(m2.conv1)
m2.act1 <- mx.symbol.Activation(m2.bn1, act_type="relu")
m2.pool1 <- mx.symbol.Pooling(m2.act1, pool_type="max", kernel=c(2,2), stride=c(2,2))
m2.drop1 <- mx.symbol.Dropout(m2.pool1, p=0.5)

# 2nd convolutional layer
m2.conv2 <- mx.symbol.Convolution(m2.drop1, kernel=c(3,3), num_filter=32)
m2.bn2 <- mx.symbol.BatchNorm(m2.conv2)
m2.act2 <- mx.symbol.Activation(m2.bn2, act_type="relu")
m2.pool2 <- mx.symbol.Pooling(m2.act2, pool_type="max", kernel=c(2,2), stride=c(2,2))
m2.drop2 <- mx.symbol.Dropout(m2.pool2, p=0.5)
m2.flatten <- mx.symbol.Flatten(m2.drop2)

## 4 fully-connected layers
m2.fc1 <- mx.symbol.FullyConnected(m2.flatten, num_hidden=1024)
m2.act3 <- mx.symbol.Activation(m2.fc1, act_type="relu")

m2.fc2 <- mx.symbol.FullyConnected(m2.act3, num_hidden=512)
m2.act4 <- mx.symbol.Activation(m2.fc2, act_type="relu")

m2.fc3 <- mx.symbol.FullyConnected(m2.act4, num_hidden=256)
m2.act5 <- mx.symbol.Activation(m2.fc3, act_type="relu")

m2.fc4 <- mx.symbol.FullyConnected(m2.act5, num_hidden=10)
m2.softmax <- mx.symbol.SoftmaxOutput(m2.fc4)

## convert into arrays
train.array <- train.x
dim(train.array) <- c(28, 28, 1, ncol(train.x))
test.array <- test.x
dim(test.array) <- c(28, 28, 1, ncol(test.x))


log <- mx.metric.logger$new() 
tick <- proc.time() 
mx.set.seed(0)

m2 <- mx.model.FeedForward.create(m2.softmax, 
                                  X = train.array, 
                                  y = train.y,
                                  num.round = 200, # This many will take a couple of hours on a CPU
                                  array.batch.size = 500,
                                  array.layout="colmajor",
                                  learning.rate = 0.01,
                                  momentum = 0.95,
                                  wd = 0.00001,
                                  eval.metric = mx.metric.accuracy,
                                  initializer = mx.init.uniform(0.07),
                                  epoch.end.callback = mx.callback.log.train.metric(1, log)
)
print(paste("Training took:", round((proc.time() - tick)[3],2),"seconds"))
## plot train accuracy
plot(log$train, type="l", col="blue", xlab="Iteration", ylab="Accuracy",lwd = 2,
     main = "CNN with 3 hidden layers")

## make predictions for test dataset
m2.preds <- predict(m2, test.array)
m2.preds.value <- max.col(t(m2.preds)) - 1


## computing misclassification error
test.y.preds <- data.matrix(m2.preds.value)
b <- 0
for (i in 1:8400){
  if (test.y[i,] != test.y.preds[i,]){
    b = b+1
  }
}

(misclassification2 <- b/8400)


###################################################################################################
## fit a convolutional neural network with only two convolutional layers, no hidden layer #########
###################################################################################################
### fit convolutionla neural network
m3.data <- mx.symbol.Variable("data")

# 1st convolutional layer
m3.conv1 <- mx.symbol.Convolution(m3.data, kernel=c(5,5), num_filter=16)
m3.bn1 <- mx.symbol.BatchNorm(m3.conv1)
m3.act1 <- mx.symbol.Activation(m3.bn1, act_type="relu")
m3.pool1 <- mx.symbol.Pooling(m3.act1, pool_type="max", kernel=c(2,2), stride=c(2,2))
m3.drop1 <- mx.symbol.Dropout(m3.pool1, p=0.5)

# 2nd convolutional layer
m3.conv2 <- mx.symbol.Convolution(m3.drop1, kernel=c(3,3), num_filter=32)
m3.bn2 <- mx.symbol.BatchNorm(m3.conv2)
m3.act2 <- mx.symbol.Activation(m3.bn2, act_type="relu")
m3.pool2 <- mx.symbol.Pooling(m3.act2, pool_type="max", kernel=c(2,2), stride=c(2,2))
m3.drop2 <- mx.symbol.Dropout(m3.pool2, p=0.5)
m3.flatten <- mx.symbol.Flatten(m3.drop2)

m3.fc1 <- mx.symbol.FullyConnected(m3.flatten, num_hidden=10)
m3.softmax <- mx.symbol.SoftmaxOutput(m3.fc1)


log3 <- mx.metric.logger$new() 
tick3 <- proc.time() 
mx.set.seed(0)

m3 <- mx.model.FeedForward.create(m3.softmax, 
                                  X = train.array, 
                                  y = train.y,
                                  num.round = 60, # This many will take a couple of hours on a CPU
                                  array.batch.size = 500,
                                  array.layout="colmajor",
                                  learning.rate = 0.01,
                                  momentum = 0.95,
                                  wd = 0.00001,
                                  eval.metric = mx.metric.accuracy,
                                  initializer = mx.init.uniform(0.07),
                                  epoch.end.callback = mx.callback.log.train.metric(1, log3)
)
print(paste("Training took:", round((proc.time() - tick3)[3],2),"seconds"))
## plot train accuracy
plot(log3$train, type="l", col="green", xlab="Iteration", ylab="Accuracy",
     lwd = 2, main = "CNN without hidden layers")


## make predictions for test dataset
m3.preds <- predict(m3, test.array)
m3.preds.value <- max.col(t(m3.preds)) - 1


## computing misclassification error
test.y.preds3 <- data.matrix(m3.preds.value)
c <- 0
for (i in 1:8400){
  if (test.y[i,] != test.y.preds3[i,]){
    c = c+1
  }
}

(misclassification3 <- c/8400)
