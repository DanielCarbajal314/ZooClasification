from data.GetGeneralData import GetGeneralData
from NaiveBayesModel.NaiveBayes import NaiveBayes
from NeuralNet.network import runNeuralNet
from svm.svm_test import SvmMain


X_train,X_test,Y_train,Y_test = GetGeneralData()
runNeuralNet(X_train,X_test,Y_train,Y_test)


#print(X_test)

SvmMain()