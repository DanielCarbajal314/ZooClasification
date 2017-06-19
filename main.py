from data.GetGeneralData import GetGeneralData
from NaiveBayes.naivebayes import runNaiveBayes
from NeuralNet.network import runNeuralNet
from svm.svm_test import SvmMain

X_train,X_test,Y_train,Y_test = GetGeneralData()
runNeuralNet(X_train,X_test,Y_train,Y_test)
runNaiveBayes(X_train,X_test,Y_train,Y_test)

SvmMain()

runKNN(X_train,X_test,Y_train,Y_test)