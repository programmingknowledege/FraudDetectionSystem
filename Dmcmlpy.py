#!/usr/bin/env python
# coding: utf-8

# In[ ]:



# Load libraries
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score


# In[ ]:


inputfile="train.csv"
#if columns are not specified in dataset
#cols = ["trustLevel","totalScanTimeInSeconds","grandTotal","lineItemVoids","scansWithoutRegistration","quantityModifications","scannedLineItemsPerSecond","valuePerSecond","lineItemVoidsPerPosition","fraud"]
#df = pd.read_csv(inputfile, sep="|", header=None, names=cols)

dataset = pd.read_csv(inputfile, sep="|", header=0)
print(dataset.head(5))
print(dataset.describe())


# In[ ]:


# scatter plot matrix
scatter_matrix(dataset)
plt.show()


# In[ ]:


# Split-out validation dataset
array = dataset.values
X = array[:,0:8]
Y = array[:,9]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


# In[ ]:


# Test options and evaluation metric
seed = 7
scoring = 'accuracy'


# In[ ]:


# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
models.append(('RF',RandomForestClassifier(n_estimators=10)))
models.append(('ADA',AdaBoostClassifier(n_estimators=100)))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# In[ ]:


# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# In[ ]:


# Make predictions on validation dataset
ada = AdaBoostClassifier(n_estimators=100)
ada.fit(X_train, Y_train)
predictions = ada.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


# In[48]:


# Make predictions on validation dataset
ada = AdaBoostClassifier(n_estimators=100)
ada.fit(X, Y)
#predictions = ada.predict(X)


# In[49]:


testset = pd.read_csv("test.csv", sep="|", header=0)
predictions = ada.predict(testset)

