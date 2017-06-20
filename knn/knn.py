from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from data.GetAnimalType import GetAnimalType

NUMBER_OF_CLASES = 7

def getLabeledResult(predictions):
    animalTypeNames = []
    for result in predictions: 
        animalTypeName = GetAnimalType(result)
        animalTypeNames.append(animalTypeName)
    return animalTypeNames

def trainKNN(X_train, Y_train, k):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, Y_train)
    return knn

def testKNN(knnModel, X_test, Y_test):
    predictions = knnModel.predict(X_test)
    predictedNames = getLabeledResult(predictions)
    testNames = getLabeledResult(Y_test)
    print(confusion_matrix(testNames, predictedNames))
    print(classification_report(testNames, predictedNames))

def runKNN(X_train, X_test, Y_train, Y_test, k = 1):
    knnModel = trainKNN(X_train, Y_train, k)
    testKNN(knnModel, X_test, Y_test)