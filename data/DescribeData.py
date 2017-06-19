
import pandas as pd
from matplotlib import pyplot
import seaborn as sns


def DescribeData():
    sns.set()
    datafile = pd.read_csv('./data/dataset/zoo.csv')
    Yvalues=datafile['class_type']
    XColumnNames = datafile.columns.difference(['class_type','animal_name'])
    Xvalues=datafile[XColumnNames]
    
    print(datafile.shape)

    print(datafile.dtypes)

    print(datafile.describe)

    print(datafile.groupby('class_type').size())

    datafile.plot(kind='density', subplots=True, layout=(4,5), figsize=(13,20), sharex=False, sharey=False)
    pyplot.show()