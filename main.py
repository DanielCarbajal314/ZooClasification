from data.GetGeneralData import GetGeneralData
from data.DescribeData import describeClases
from NaiveBayes.naivebayes import runNaiveBayes
from NeuralNet.network import runNeuralNet
from svm.svm import runSVM
from knn.knn import runKNN
from ada.ada import runAdaRF

from itertools import groupby
from operator import itemgetter
from collections import defaultdict


#DescribeData()
describeClases()
#X_train,X_test,Y_train,Y_test = GetGeneralData()
X_train,X_test,X_validation,Y_train,Y_test,Y_validation = GetGeneralData()

print('=======NeuralNet=====')
runNeuralNet(X_train,X_test,Y_train,Y_test)
print('=======NaiveBayes=====')
runNaiveBayes(X_train,X_test,X_validation,Y_train,Y_test,Y_validation)

print('=======KNN=====')
runKNN(X_train,X_test,X_validation,Y_train,Y_test,Y_validation)

print('=======SVM=====')
runSVM(X_train,X_test,X_validation,Y_train,Y_test,Y_validation)

print('=======Adaboost-RandomForest=====')
runAdaRF(X_train,X_test,X_validation,Y_train,Y_test,Y_validation)
