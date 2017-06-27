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

def findBestKNN(X_train,X_validation,Y_train,Y_validation):
    mejor_modelo = None
    mejor_score = 0

    #Se prueba con K = 1 hasta K = SquareRoot(n)
    for i in range(1, int(len(Y_train)**(.5)) + 1):
        print("Evaluando K = %d" % i)
        model = trainKNN(X_train, Y_train, i)
        score = model.score(X_validation, Y_validation)

        if score > mejor_score:
            mejor_score = score
            mejor_modelo = model

    return mejor_modelo

def runKNN(X_train, X_test, X_validation, Y_train, Y_test, Y_validation):
    knnModel = findBestKNN(X_train, X_validation, Y_train, Y_validation)
    print('Mejor valor de K: %d' % knnModel.get_params()['n_neighbors'])
    testKNN(knnModel, X_test, Y_test)