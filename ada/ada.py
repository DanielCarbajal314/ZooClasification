import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from data.GetGeneralData import GetGeneralData
from sklearn.grid_search import GridSearchCV

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

def trainAdaboost(X_train, Y_train):
    param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "n_estimators": [1, 2, 50, 150, 200]
             }

    dtc = DecisionTreeClassifier(random_state = 11, max_features = "auto", class_weight = "auto",max_depth = None)

    adaboostModel = AdaBoostClassifier(base_estimator = dtc)

    grid_search_adaboost = GridSearchCV(ABC, param_grid=param_grid, scoring = 'roc_auc')

    estimator=grid_search_adaboost.estimator

    adaboostModel = estimator.fit(X_train, Y_train)

return adaboostModel

def test(tested_model, X_test, Y_test):
    predictions = tested_model.predict(X_test)
    predictedNames = getLabeledResult(predictions)
    testNames = getLabeledResult(Y_test)
    print(confusion_matrix(testNames, predictedNames))
    print(classification_report(testNames, predictedNames))

def runAdaRF(X_train, X_test, Y_train, Y_test):
    rfModel = trainRFTuning(X_train, Y_train, X_test, Y_test)
    test(rfModel, X_test, Y_test)
    adaboostModel = trainAdaboost(X_train, Y_train)
    test(adaboostModel, X_test, Y_test)
