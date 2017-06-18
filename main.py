from data.GetGeneralData import GetGeneralData
from svm.svm_test import SvmMain


X_train,X_test,Y_train,Y_test = GetGeneralData()

#print(X_test)

SvmMain()