from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
from data.GetAnimalType import GetAnimalType
import numpy as np
import pandas as pd

NUMBER_OF_CLASES = 7

def __getDataFromFile(fileName):
    datafile = pd.read_csv(fileName)
    Yvalues=datafile['class_type']
    XColumnNames = datafile.columns.difference(['class_type','animal_name'])
    Xvalues=datafile[XColumnNames]
    return Xvalues,Yvalues

def runSVM(X_train,X_test,X_validation,Y_train,Y_test,Y_validation):
    #Se seleccionó svm con kernel gaussiano
    #Pruebas con otros kernels
    #print(Y_validation)
    #print(Y_test)
    #SvmKernelTest(X_train,X_validation,Y_train,Y_validation)
    model = _testSvmGaussian(X_train,X_validation,Y_train,Y_validation)
    X_trainval =  np.concatenate((X_train,X_validation),axis=0)
    Y_trainval = np.concatenate((Y_train,Y_validation),axis=0)
    model.fit(X_trainval,Y_trainval)
    _testSVM(model,X_test,Y_test)

def testModel(model,X_test,Y_test):
    predictions = model.predict(X_test)
    _testSVM(predictions=predictions, Y_test=Y_test)

def _SvmGaussian(X_train,X_validation,Y_train,Y_validation):
    #Se elige los mejores parámetros C y Gamma
    Cs = np.logspace(-2, 2, 9)  # ~ [0.01, 0.03, 0.1, ..., 100]
    gammas = np.logspace(-4, 4, 9)  # [0.0001, 0.001, ..., 10000]

    mejor_modelo = None
    mejor_score = 0
    for C in Cs:
        for gamma in gammas:

            model = SVC(kernel='rbf', C=C, gamma=gamma)
            model.fit(X_train, Y_train)

            score_val = model.score(X_validation, Y_validation)

            if score_val > mejor_score:
                mejor_score = score_val
                mejor_modelo = model

    #Se imprime un resumen de la predicción.
    print('---------------- Kernel Gaussiano----------------')
    _testSVM(mejor_modelo, X_validation, Y_validation)
    print('Mejor valor de C: %0.4f' % mejor_modelo.get_params()['C'])
    print('Mejor valor de gamma: %0.4f' % mejor_modelo.get_params()['gamma'])
    return mejor_modelo

def _testSvmGaussian(X_train,X_validation,Y_train,Y_validation):
    #Se elige los mejores parámetros C y Gamma
    Cs = np.logspace(-2, 2, 9)  # ~ [0.01, 0.03, 0.1, ..., 100]
    gammas = np.logspace(-4, 4, 9)  # [0.0001, 0.001, ..., 10000]

    mejor_modelo = None
    mejor_score = 0
    for C in Cs:
        for gamma in gammas:

            model = SVC(kernel='rbf', C=C, gamma=gamma)
            model.fit(X_train, Y_train)

            score_val = model.score(X_validation, Y_validation)

            if score_val > mejor_score:
                mejor_score = score_val
                mejor_modelo = model
    return mejor_modelo

def _testSVM(model,X_test,Y_test):
    predictions = model.predict(X_test)
    predictedNames = getLabeledResult(predictions)
    testNames = getLabeledResult(Y_test)
    print(confusion_matrix(testNames, predictedNames))
    print(classification_report(testNames, predictedNames))

def getLabeledResult(predictions):
    animalTypeNames = []
    for result in predictions:
        animalTypeName = GetAnimalType(result)
        animalTypeNames.append(animalTypeName)
    return animalTypeNames

#Kernel Test section
def SvmKernelTest(X_train,X_validation,Y_train,Y_validation):
    _SvmLinear(X_train,X_validation,Y_train,Y_validation)
    _SvmGaussian(X_train,X_validation,Y_train,Y_validation)
    _SvmPolinomial(X_train,X_validation,Y_train,Y_validation)
    _SvmSigmoidal(X_train,X_validation,Y_train,Y_validation)

def _SvmLinear(X_train,X_validation,Y_train,Y_validation):
    #hiperplanos
    Cs = np.logspace(-2, 2, 9) #Se elige el parámetro C
    mejor_modelo = None
    mejor_score = 0
    for C in Cs:
        model = model = SVC(kernel='linear', C=C)
        model.fit(X_train, Y_train)

        score_val = model.score(X_validation, Y_validation)

        if score_val > mejor_score:
            mejor_score = score_val
            mejor_modelo = model

    print('---------------- Kernel Lineal -----------------')
    # Se imprime un resumen de la predicción.
    _testSVM(mejor_modelo, X_validation, Y_validation)
    print('Mejor valor de C: %0.4f' % mejor_modelo.get_params()['C'])
    return C

def _SvmPolinomial(X_train,X_validation,Y_train,Y_validation):
    #Curvas polinomiales de grado d
    Cs = np.logspace(-2, 2, 9)  # Se elige el parámetro C
    Ds = [1,2,3,4,5] #Se prueban varios grados polinomiales
    mejor_modelo = None
    mejor_score = 0
    for C in Cs:
        for D in Ds:
            model = model = SVC(kernel='poly', C=1, degree = D)
            model.fit(X_train, Y_train)

            score_val = model.score(X_validation, Y_validation)

            if score_val > mejor_score:
                mejor_score = score_val
                mejor_modelo = model
    # Se imprime un resumen de la predicción.
    print('---------------- Kernel Polinomial -----------------')
    _testSVM(mejor_modelo,X_validation,Y_validation)
    print('Mejor valor de C: %0.4f' % mejor_modelo.get_params()['C'])
    print('Mejor valor de grado: %d' % mejor_modelo.get_params()['degree'])
    return C, D


def _SvmSigmoidal(X_train,X_validation,Y_train,Y_validation):
    #Se elige los mejores parámetros C y Gamma
    Cs = np.logspace(-2, 2, 9)  # ~ [0.01, 0.03, 0.1, ..., 100]
    gammas = np.logspace(-4, 4, 9)  # [0.0001, 0.001, ..., 10000]

    mejor_modelo = None
    mejor_score = 0
    for C in Cs:
        for gamma in gammas:
            model = model = SVC(kernel='sigmoid', C=C, gamma=gamma)
            model.fit(X_train, Y_train)

            score_val = model.score(X_validation, Y_validation)

            if score_val > mejor_score:
                mejor_score = score_val
                mejor_modelo = model

    #Se imprime un resumen de la predicción.
    print('---------------- Kernel Sigmoidal ----------------')
    _testSVM(mejor_modelo, X_validation, Y_validation)
    print('Mejor valor de C: %0.4f' % mejor_modelo.get_params()['C'])
    print('Mejor valor de gamma: %0.4f' % mejor_modelo.get_params()['gamma'])
    return C,gamma

