import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from data.GetGeneralData import GetGeneralData

def getLabeledResult(predictions):
    animalTypeNames = []
    for result in predictions:
        animalTypeName = GetAnimalType(result)
        animalTypeNames.append(animalTypeName)
    return animalTypeNames

def trainRFTuning(X_train, Y_train, X_test, Y_test):
    best_score = 0
    best_n_estimators = None
    best_rfModel = None
    rf_error_train = []
    rf_error_test = []

    n_estimators_grid = np.linspace(2,80,40).astype(int)

    for n_estimators in n_estimators_grid:
        rfModel = RandomForestClassifier(n_estimators=n_estimators, random_state=0)
        rfModel.fit(X_train, Y_train)

        score_train = rfModel.score(X_train, Y_train)
        rf_error_train.append(1 - score_train)

        score_test = rfModel.score(X_test, Y_test)
        rf_error_test.append(1 - score_test)

        if score_test > best_score:
            best_score = score_test
            best_n_estimators = n_estimators
            best_rfModel = rfModel

    rfModel = best_rfModel
    rfModel.fit(X_train, y_train)

    print ("Mejor valor de n_estimators :", best_n_estimators)
    print ("Exactitud de RandomForest en conjunto de entrenamiento :", rfModel.score(X_train, y_train))
    print ("Exactitud de RandomForest en conjunto de validación    :", rfModel.score(X_test, y_test))
return rfModel

def trainAdaTuning(X_train, Y_train, X_test, Y_test):
    n_estimators=600
    min_samples_leaf_grid = np.linspace(1,10,10).astype(int)

    best_score = 0
    best_min_samples_leaf = None

    for min_samples_leaf in min_samples_leaf_grid:
    base_estimator = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf, random_state=0)
    adaModel = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=n_estimators, random_state=0)
    adaModel.fit(X_train, y_train)

    score_test = adaModel.score(X_test, y_test)

    if score_test > best_score:
        best_score = score_test
        best_min_samples_leaf = min_samples_leaf
        best_adaModel = adaModel

    adaModel = best_adaModel
    adaModel.fit(X_train, y_train)

    print ("Mejor valor de min_samples_leaf :", best_min_samples_leaf)
    print ("Exactitud de AdaBoost en conjunto de entrenamiento :", adaModel.score(X_train, y_train))
    print ("Exactitud de AdaBoost en conjunto de validación    :", best_score)
return adaModel

def test(tested_model, X_test, Y_test):
    predictions = tested_model.predict(X_test)
    predictedNames = getLabeledResult(predictions)
    testNames = getLabeledResult(Y_test)
    print(confusion_matrix(testNames, predictedNames))
    print(classification_report(testNames, predictedNames))

def runAdaRF(X_train, X_test, Y_train, Y_test):
    rfModel = trainRFTuning(X_train, Y_train, X_test, Y_test)
    test(rfModel, X_test, Y_test)
    adaModel = trainAdaTuning(X_train, Y_train, X_test, Y_test)
    test(adaModel, X_test, Y_test)
