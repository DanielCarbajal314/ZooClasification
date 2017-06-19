from data.GetGeneralData import GetGeneralData
from NaiveBayes.naivebayes import runNaiveBayes
from NeuralNet.network import runNeuralNet
<<<<<<< HEAD

X_train,X_test,Y_train,Y_test = GetGeneralData()
runNeuralNet(X_train,X_test,Y_train,Y_test)
runNaiveBayes(X_train,X_test,Y_train,Y_test)
=======
from svm.svm_test import SvmMain


X_train,X_test,Y_train,Y_test = GetGeneralData()
runNeuralNet(X_train,X_test,Y_train,Y_test)


#print(X_test)

SvmMain()
>>>>>>> fd341d73d06e57949a79f453b9050c0f64ec297d
