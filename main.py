from data.GetGeneralData import GetGeneralData
<<<<<<< HEAD
from NaiveBayesModel.NaiveBayes import NaiveBayes

=======
from NeuralNet.network import runNeuralNet
>>>>>>> 20eaacc2d69647bbbcdf7069a96fe673b21a2fa3

X_train,X_test,Y_train,Y_test = GetGeneralData()
runNeuralNet(X_train,X_test,Y_train,Y_test)


<<<<<<< HEAD

print(X_test)
=======
>>>>>>> 20eaacc2d69647bbbcdf7069a96fe673b21a2fa3

