require(xgboost)
require(methods)
require(reshape2)

featureclass = rep('numeric', length(classnames))

train = read.csv('train_cleaned.csv', header=T)
test = read.csv('test_cleaned.csv', header=T)

target = length(colnames(train))-1
classnames = unique(target)
target = as.integer(classnames)

trainMatrix <- data.matrix(train)
testMatrix <- data.matrix(test)

numberOfClasses <-max(target) +1

param <- list('objective' = 'multi:softprob', 'eval_metric' = 'mlogloss', 'num_class' = numberOfClasses)

cv.nround <- 500
cv.nfold <- 5
bst.cv = xgb.cv(param=param, data = trainMatrix, label = train$TripType, nfold = cv.nfold, nrounds = cv.nround)

plot(bst.cv$test.mlogloss.mean, lty = '1')
nround <- which(bst.cv$test.mlogloss.mean==min(bst.cv$test.mlogloss.mean))

bst = xgboost(data = trainMatrix, label = target, params = param, nrounds = nround)


ypred = predict(bst, testMatrix)
predMatrix <- data.frame(matrix(ypred, ncol=numberOfClasses, byrow=TRUE))
colnames(predMatrix) = classnames

res <- data.frame(id, predMatrix)
write.csv(res, 'submissionR.csv', quote=F, row.names=F)



