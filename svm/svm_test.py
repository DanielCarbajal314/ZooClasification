from data.GetGeneralData import GetGeneralData
from sklearn.svm import SVC
from sklearn import metrics

import numpy as np

X_train, X_test, Y_train, Y_test = GetGeneralData()

def _SvmKernelTest():
    #_SvmLinear()
    #_SvmGaussian()
    #_SvmPolinomial()
    #_SvmSigmoidal()
    c, gamma = _getParametersGaussian()

def _SvmLinear():
    #hiperplanos
    svm_classifier = SVC(kernel='linear', C=10)
    svm_classifier.fit(X_train, Y_train)
    score = svm_classifier.score(X_test,Y_test)
    print('Exactitud en el conjunto de pruebas (kernel linear): %0.4f' % score)

def _SvmGaussian():
    #hiperesferas
    svm_classifier = SVC(kernel='rbf', C=1, gamma = 'auto')
    svm_classifier.fit(X_train, Y_train)
    score = svm_classifier.score(X_test,Y_test)
    print('Exactitud en el conjunto de pruebas (kernel gaussiano): %0.4f' % score)

def _SvmPolinomial():
    #Curvas polinomiales de grado d
    d = 2
    svm_classifier = SVC(kernel='poly', C=1, coef0 = 0, degree = d)
    svm_classifier.fit(X_train, Y_train)
    score = svm_classifier.score(X_test,Y_test)
    print('Exactitud en el conjunto de pruebas (kernel polinomial): %0.4f' % score)

def _SvmSigmoidal():
    svm_classifier = SVC(kernel='sigmoid', C=1, gamma = 'auto', coef0= 0)
    svm_classifier.fit(X_train, Y_train)
    score = svm_classifier.score(X_test,Y_test)
    print('Exactitud en el conjunto de pruebas (kernel sigmoidal): %0.4f' % score)

def SvmMain():
    _SvmKernelTest()

def _getParametersGaussian():
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

    print('Mejor valor de C: %0.4f' % mejor_modelo.get_params()['C'])
    print('Mejor valor de gamma: %0.4f' % mejor_modelo.get_params()['gamma'])
    print('Exactitud en el conjunto de entrenamiento: %0.4f' % mejor_modelo.score(X_test, Y_test))
    print('Exactitud en el conjunto de validación: %0.4f' % mejor_modelo.score(X_train, Y_train))
