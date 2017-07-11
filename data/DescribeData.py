
import pandas as pd
from matplotlib import pyplot
import seaborn as sns


def describeClases():
    datafile = pd.read_csv('./data/dataset/zoo.csv')
    Yvalues = datafile['class_type']
    percents = ((Yvalues.value_counts() / Yvalues.count())*100).round(2)
    names = pd.DataFrame(['Unknown','Mammal','Bird','Reptile','Fish','Amphibian','Bug','Invertebrat'],columns=["class_names"])
    percentTable = pd.concat([names,percents],axis=1)
    print("=====Unbalanced Data=====")
    print(percentTable)

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