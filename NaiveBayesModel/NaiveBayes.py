get_ipython().magic('matplotlib inline')
import seaborn
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold

x = []
y = []

def NaiveBayes(X_train,Y_train):
    x = X_train
    y = Y_train

    listgnb = []
    listmnb = []
    listbnb = []

    gnb = GaussianNB()
    listgnb = score_seed(gnb,10,listgnb,"GNB completed")

    mnb = MultinomialNB()
    listmnb = score_seed(mnb,10,listmnb,"MNB completed")

    bnb = BernoulliNB()
    listbnb = score_seed(bnb,10,listbnb,"BNB completed")

    list_model = ["Gausian NB", "Multinomial NB", "Bernoulli NB"]
    list_pref  = [np.mean(listgnb),np.mean(listmnb),np.mean(listbnb)]
    return 

def plot_correlation_map( df ):
    corr = df.corr()
    _ , ax = plt.subplots( figsize =( 14 , 12 ) )
    cmap = seaborn.diverging_palette( 220 , 10 , as_cmap = True )
    _ = seaborn.heatmap(
        corr, 
        cmap = cmap,
        square=True, 
        cbar_kws={ 'shrink' : .9 }, 
        ax=ax, 
        annot = True, 
        annot_kws = { 'fontsize' : 12 }
    )

def score(clf, train_np, random_state, folds, target):
    kf = StratifiedKFold(n_splits = folds, shuffle = False, random_state = random_state)
    list_perf = []
    for itrain, itest in kf.split(train_np,target):
        Xtr, Xte = train_np[itrain], train_np[itest]
        ytr, yte = target[itrain], target[itest]
        clf.fit(Xtr, ytr.ravel())
        pred = pd.DataFrame(clf.predict(Xte)) 
        list_perf.append(metrics.accuracy_score(yte,pred))
    return list_perf

def score_seed(clf,nbseed,listout,printoutput):
    listout = []
    listoutfold = []

    for i in range(nbseed):
        list1 = score(clf,np.array(x),random_state = i, folds = 4, target = y )
        listout.append(list1)
    print(printoutput) 
    return listout


