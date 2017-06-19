import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from data.GetAnimalType import GetAnimalType
from sklearn.metrics import classification_report,confusion_matrix
NUMBER_OF_CLASES = 7

def runNaiveBayes(X_train,X_test,Y_train,Y_test):
    NaiveBayesModel = TrainNaiveBayes(X_train,Y_train)
    TestNaiveBayes(NaiveBayesModel,X_test,Y_test)

def TrainNaiveBayes(X_train,Y_train):
    bernoulliNB = BernoulliNB()
    bernoulliNB.fit(X_train, Y_train)
    return bernoulliNB

def getLabeledResult(predictions):
    animalTypeNames = []
    for result in predictions: 
        animalTypeName = GetAnimalType(result)
        animalTypeNames.append(animalTypeName)
    return animalTypeNames    

def TestNaiveBayes(NaiveBayesModel,X_test,Y_test):
    predictions = NaiveBayesModel.predict(X_test)
    predictedNames = getLabeledResult(predictions)
    testNames = getLabeledResult(Y_test)
    print(confusion_matrix(testNames,predictedNames))
    print(classification_report(testNames,predictedNames))

