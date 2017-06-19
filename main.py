from data.GetGeneralData import GetGeneralData
from data.DescribeData import DescribeData
from NaiveBayes.naivebayes import runNaiveBayes
from NeuralNet.network import runNeuralNet
from svm.svm import runSVM
from knn.knn import runKNN
from ada.ada import runAdaRF

DescribeData()

X_train,X_test,Y_train,Y_test = GetGeneralData()
print('=======NeuralNet=====')
runNeuralNet(X_train,X_test,Y_train,Y_test)
print('=======NaiveBayes=====')
runNaiveBayes(X_train,X_test,Y_train,Y_test)
print('=======KNN=====')
runKNN(X_train,X_test,Y_train,Y_test)
print('=======SVM=====')
runSVM(X_train,X_test,Y_train,Y_test)
print('=======Adaboost-RandomForest=====')
runAdaRF(X_train,X_test,Y_train,Y_test)
