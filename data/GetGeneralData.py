import pandas as pd
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
    NonBalances_X_train,X_ForTesting,NonBalances_Y_train,Y_ForTesting = train_test_split(Xvalues, Yvalues,test_size=0.30,random_state=42)
    X_train,Y_train = __overSampleData(NonBalances_X_train,NonBalances_Y_train)
    X_ForTesting,Y_ForTesting = __overSampleData(X_ForTesting,Y_ForTesting)
    X_test,X_validation,Y_test,Y_validation = train_test_split(X_ForTesting, Y_ForTesting,test_size=0.50,random_state=42)
    return X_train,X_test,X_validation,Y_train,Y_test,Y_validation


