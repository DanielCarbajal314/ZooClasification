from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
from data.GetAnimalType import GetAnimalType
import numpy as np

NUMBER_OF_CLASES = 7

def runSVM(X_train,X_test,Y_train,Y_test):
    #Se seleccionó svm con kernel gaussiano
    _testSvmGaussian(X_train,X_test,Y_train,Y_test)
    #Pruebas con otros kernels
    #SvmKernelTest(X_train,X_test,Y_train,Y_test)

def _testSvmGaussian(X_train,X_test,Y_train,Y_test):
    #Se elige los mejores parámetros C y Gamma
    Cs = np.logspace(-2, 2, 9)  # ~ [0.01, 0.03, 0.1, ..., 100]
    gammas = np.logspace(-4, 4, 9)  # [0.0001, 0.001, ..., 10000]

    mejor_modelo = None
    mejor_score = 0
    for C in Cs:
        for gamma in gammas:

            model = model = SVC(kernel='rbf', C=C, gamma=gamma)
            model.fit(X_train, Y_train)

            score_val = model.score(X_test, Y_test)

            if score_val > mejor_score:
                mejor_score = score_val
                mejor_modelo = model

    predictions = mejor_modelo.predict(X_test)
    #Se imprime un resumen de la predicción.
    print('---------------- Kernel Gaussiano----------------')
    _testSVM(predictions=predictions,Y_test=Y_test)
    #print('Mejor valor de C: %0.4f' % mejor_modelo.get_params()['C'])
    #print('Mejor valor de gamma: %0.4f' % mejor_modelo.get_params()['gamma'])
    return C,gamma

def _testSVM(predictions,Y_test):
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
def SvmKernelTest(X_train,X_test,Y_train,Y_test):
    _SvmLinear(X_train,X_test,Y_train,Y_test)
    _testSvmGaussian(X_train,X_test,Y_train,Y_test)
    _SvmPolinomial(X_train,X_test,Y_train,Y_test)
    _SvmSigmoidal(X_train,X_test,Y_train,Y_test)

def _SvmLinear(X_train,X_test,Y_train,Y_test):
    #hiperplanos
    Cs = np.logspace(-2, 2, 9) #Se elige el parámetro C
    mejor_modelo = None
    mejor_score = 0
    for C in Cs:
        model = model = SVC(kernel='linear', C=C)
        model.fit(X_train, Y_train)

        score_val = model.score(X_test, Y_test)

        if score_val > mejor_score:
            mejor_score = score_val
            mejor_modelo = model

    print('---------------- Kernel Lineal -----------------')
    predictions = mejor_modelo.predict(X_test)
    # Se imprime un resumen de la predicción.
    _testSVM(predictions=predictions, Y_test=Y_test)
    print('Mejor valor de C: %0.4f' % mejor_modelo.get_params()['C'])
    return C

def _SvmPolinomial(X_train,X_test,Y_train,Y_test):
    #Curvas polinomiales de grado d
    Cs = np.logspace(-2, 2, 9)  # Se elige el parámetro C
    Ds = [1,2,3,4,5] #Se prueban varios grados polinomiales
    mejor_modelo = None
    mejor_score = 0
    for C in Cs:
        for D in Ds:
            model = model = SVC(kernel='poly', C=1, degree = D)
            model.fit(X_train, Y_train)

            score_val = model.score(X_test, Y_test)

            if score_val > mejor_score:
                mejor_score = score_val
                mejor_modelo = model

    predictions = mejor_modelo.predict(X_test)
    # Se imprime un resumen de la predicción.
    print('---------------- Kernel Polinomial -----------------')
    _testSVM(predictions=predictions, Y_test=Y_test)
    print('Mejor valor de C: %0.4f' % mejor_modelo.get_params()['C'])
    print('Mejor valor de grado: %d' % mejor_modelo.get_params()['degree'])
    return C, D


def _SvmSigmoidal(X_train,X_test,Y_train,Y_test):
    #Se elige los mejores parámetros C y Gamma
    Cs = np.logspace(-2, 2, 9)  # ~ [0.01, 0.03, 0.1, ..., 100]
    gammas = np.logspace(-4, 4, 9)  # [0.0001, 0.001, ..., 10000]

    mejor_modelo = None
    mejor_score = 0
    for C in Cs:
        for gamma in gammas:
            model = model = SVC(kernel='sigmoid', C=C, gamma=gamma)
            model.fit(X_train, Y_train)

            score_val = model.score(X_test, Y_test)

            if score_val > mejor_score:
                mejor_score = score_val
                mejor_modelo = model

    predictions = mejor_modelo.predict(X_test)
    #Se imprime un resumen de la predicción.
    print('---------------- Kernel Sigmoidal ----------------')
    _testSVM(predictions=predictions,Y_test=Y_test)
    print('Mejor valor de C: %0.4f' % mejor_modelo.get_params()['C'])
    print('Mejor valor de gamma: %0.4f' % mejor_modelo.get_params()['gamma'])
    return C,gamma

