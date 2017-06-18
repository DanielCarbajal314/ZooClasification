from sklearn.neural_network import MLPClassifier as neuralNetworkClasifierConstructor
from sklearn.metrics import classification_report,confusion_matrix
from data.GetAnimalType import GetAnimalType
import numpy as np
NUMBER_OF_CLASES = 7

def __transformToNeuralNetResultVector(YResultData):
    transformedResult = []
    for result in YResultData: 
        zeros = np.zeros(NUMBER_OF_CLASES)
        zeros[result-1]=1
        transformedResult.append(zeros)
    return transformedResult

def __trainNetwork(X_train,Y_train):
    neuralNetwork = neuralNetworkClasifierConstructor(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(11), random_state=1)
    AdaptedToNeuralNet_Y_train = __transformToNeuralNetResultVector(Y_train)
    neuralNetwork.fit(X_train,AdaptedToNeuralNet_Y_train)
    return neuralNetwork

def __TestNeuralNet(neuralNetwork,X_test,Y_test):
    AdaptedToNeuralNet_Y_test = __transformToNeuralNetResultVector(Y_test)
    predictions = neuralNetwork.predict(X_test)
    predictedNames = __getLabeledResult(predictions)
    testNames = __getLabeledResult(AdaptedToNeuralNet_Y_test)
    print(confusion_matrix(testNames,predictedNames))
    print(classification_report(testNames,predictedNames))

def __getLabeledResult(AdaptedResult):
    animalTypeNames = []
    for result in AdaptedResult: 
        animalTypeIndex = next(i for i,v in enumerate(result) if v > 0)+1
        animalTypeName = GetAnimalType(animalTypeIndex)
        animalTypeNames.append(animalTypeName)
    return animalTypeNames

def runNeuralNet(X_train,X_test,Y_train,Y_test):
    neuralNetwork = __trainNetwork(X_train,Y_train)
    __TestNeuralNet(neuralNetwork,X_test,Y_test)



