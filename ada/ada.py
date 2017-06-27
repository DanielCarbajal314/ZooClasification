import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from data.GetAnimalType import GetAnimalType

def getLabeledResult(predictions):
    animalTypeNames = []
    for result in predictions:
        animalTypeName = GetAnimalType(result)
        animalTypeNames.append(animalTypeName)
    return animalTypeNames

def trainRFTuning(X_train, Y_train, X_validation, Y_validation):
    best_score = 0
    best_n_estimators = None
    best_rfModel = None
    rf_error_train = []
    rf_error_val = []

    n_estimators_grid = np.linspace(2,80,40).astype(int)

    for n_estimators in n_estimators_grid:
        rfModel = RandomForestClassifier(n_estimators=n_estimators, random_state=0)
        rfModel.fit(X_train, Y_train)

        score_train = rfModel.score(X_train, Y_train)
        rf_error_train.append(1 - score_train)

        score_val= rfModel.score(X_validation, Y_validation)
        rf_error_val.append(1 - score_val)

        if score_val > best_score:
            best_score = score_val
            best_n_estimators = n_estimators
            best_rfModel = rfModel

    rfModel = best_rfModel
    rfModel.fit(X_validation, Y_validation)

    print ("Mejor valor de n_estimators :", best_n_estimators)
    print ("Exactitud de RandomForest en conjunto de entrenamiento :", rfModel.score(X_train, Y_train))
    print ("Validacion", rfModel.score(X_validation, Y_validation))
    return rfModel

def trainAdaboost(X_train, Y_train):
    param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "n_estimators": [1, 2, 50, 150, 200]
             }

    dtc = DecisionTreeClassifier(random_state = 11, max_features = "auto", class_weight = "auto",max_depth = None)

    adaboostModel = AdaBoostClassifier(base_estimator = dtc)
    
    grid_search_adaboost = GridSearchCV(adaboostModel, param_grid=param_grid, scoring = 'precision_macro')

    grid_search_adaboost.estimator.fit(X_train,Y_train)

    return grid_search_adaboost.estimator

def test(tested_model, X_test, Y_test):
    predictions = tested_model.predict(X_test)
    predictedNames = getLabeledResult(predictions)
    testNames = getLabeledResult(Y_test)
    print(confusion_matrix(testNames, predictedNames))
    print(classification_report(testNames, predictedNames))

def runAdaRF(X_train,X_test,X_validation,Y_train,Y_test,Y_validation):
    print('==RandomForest==')
    rfModel = trainRFTuning(X_train, Y_train, X_validation, Y_validation)
    test(rfModel, X_test, Y_test)
    print('==Adaboost==')
    adaboostModel = trainAdaboost(np.concatenate((X_train,X_validation),axis=0), np.concatenate((Y_train,Y_validation),axis=0))
    test(adaboostModel, X_test, Y_test)
