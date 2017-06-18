import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.cross_validation import train_test_split

def __getDataFromFile(fileName):
    datafile = pd.read_csv(fileName)
    Yvalues=datafile['class_type']
    XColumnNames = datafile.columns.difference(['class_type','animal_name'])
    Xvalues=datafile[XColumnNames]
    return Xvalues,Yvalues

def __overSampleData(Xvalues,Yvalues):
    randomOverSampler = RandomOverSampler()
    XResampled, YResampled = randomOverSampler.fit_sample(Xvalues, Yvalues)
    return XResampled,YResampled

def GetGeneralData():
    Xvalues,Yvalues=__getDataFromFile('./data/dataset/zoo.csv')
    XResampled,YResampled = __overSampleData(Xvalues,Yvalues)
    X_train,X_test,Y_train,Y_test = train_test_split(XResampled, YResampled)
    return X_train,X_test,Y_train,Y_test


