
# coding: utf-8

# In[7]:

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


# Import data
data = pd.read_csv("./zoo.csv",sep=",")


# In[8]:

data


# In[2]:

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
plot_correlation_map(data)


# In[9]:

data.drop('animal_name',axis=1,inplace=True)
# Data and Target
y = data['class_type']
x = data.drop('class_type',axis = 1)

# Functions
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


# In[10]:

# Modelisation
# 10 seed differents. for each classifications we'll take 4 folds
gnb = GaussianNB()
listgnb = []
listgnb = score_seed(gnb,10,listgnb,"GNB completed")

mnb = MultinomialNB()
listmnb = []
listmnb = score_seed(mnb,10,listmnb,"MNB completed")

bnb = BernoulliNB()
listbnb = []
listbnb = score_seed(bnb,10,listbnb,"BNB completed")

print("GNB perf:",   np.mean(listgnb))
print("MNB perf:",   np.mean(listmnb))
print("BNB perf:",   np.mean(listbnb))


# In[11]:

list_model = ["Gausian NB", "Multinomial NB", "Bernoulli NB"]
list_pref  = [np.mean(listgnb),np.mean(listmnb),np.mean(listbnb)]


# In[6]:

print(list_pref)


# In[14]:

# Performance of models
fg,ax = plt.subplots(nrows=1,ncols=1,figsize=(6, 2.5))
sns.barplot(list_model, list_pref)
plt.ylim(0.80, 0.97)
ax.set_ylabel("Performance")
ax.set_xlabel("Name")
ax.set_xticklabels(list_model,rotation=35)
plt.title('Battle of Algorithms')


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



