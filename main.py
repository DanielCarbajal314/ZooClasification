from data.GetGeneralData import GetGeneralData
from NaiveBayesModel.NaiveBayes import NaiveBayes
from NeuralNet.network import runNeuralNet
X_train,X_test,Y_train,Y_test = GetGeneralData()
runNeuralNet(X_train,X_test,Y_train,Y_test)
