from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
from data.GetAnimalType import GetAnimalType
import numpy as np

NUMBER_OF_CLASES = 7

def runSVM(X_train,X_test,Y_train,Y_test):
    #Se seleccionó svm con kernel gaussiano
    _testSvmGaussian(X_train,X_test,Y_train,Y_test)

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
    _testSVM(predictions=predictions,Y_test=Y_test)
    #print('Mejor valor de C: %0.4f' % mejor_modelo.get_params()['C'])
    #print('Mejor valor de gamma: %0.4f' % mejor_modelo.get_params()['gamma'])
    return Cs,gammas

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

#def _SvmKernelTest(X_train,X_test,Y_train,Y_test):
    #_SvmLinear()
    #_SvmGaussian()
    #_SvmPolinomial()
    #_SvmSigmoidal()

#def _SvmLinear(X_train,X_test,Y_train,Y_test):
    #hiperplanos
    #svm_classifier = SVC(kernel='linear', C=10)
    #svm_classifier.fit(X_train, Y_train)
    #score = svm_classifier.score(X_test,Y_test)
    #print('Exactitud en el conjunto de pruebas (kernel linear): %0.4f' % score)

#def _SvmGaussian(X_train,X_test,Y_train,Y_test):
    #hiperesferas
    #svm_classifier = SVC(kernel='rbf', C=1, gamma = 'auto')
    #svm_classifier.fit(X_train, Y_train)
    #score = svm_classifier.score(X_test,Y_test)
    #print('Exactitud en el conjunto de pruebas (kernel gaussiano): %0.4f' % score)

#def _SvmPolinomial(X_train,X_test,Y_train,Y_test):
    #Curvas polinomiales de grado d
    #d = 2
    #svm_classifier = SVC(kernel='poly', C=1, coef0 = 0, degree = d)
    #svm_classifier.fit(X_train, Y_train)
    #score = svm_classifier.score(X_test,Y_test)
    #print('Exactitud en el conjunto de pruebas (kernel polinomial): %0.4f' % score)

#def _SvmSigmoidal(X_train,X_test,Y_train,Y_test):
    #svm_classifier = SVC(kernel='sigmoid', C=1, gamma = 'auto', coef0= 0)
    #svm_classifier.fit(X_train, Y_train)
    #score = svm_classifier.score(X_test,Y_test)
    #print('Exactitud en el conjunto de pruebas (kernel sigmoidal): %0.4f' % score)