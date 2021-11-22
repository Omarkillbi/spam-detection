# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 10:29:28 2021
@author: anisP

"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from matplotlib import pyplot as plt
import numpy as np


# * import des données
data =  pd.read_csv('spambase.data', sep=",", header=None)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# * filtrage par variance
var_thre = VarianceThreshold(0.2) # instanciation
X_thresh = var_thre.fit_transform(X)

# * standarisation 
scalar = StandardScaler()
X_scaled = scalar.fit_transform(X)

# * split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# * regression logistique
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
conf_matrix = metrics.confusion_matrix(y_test, y_pred)

fpr, tpr, thresh = metrics.roc_curve(y_test, y_pred, pos_label=1)
plt.style.use('seaborn')
plt.plot(fpr, tpr, linestyle=':', color='red')
auc = metrics.roc_auc_score(y_test, y_pred) 
print(f"AUC = {auc}")

# * K fold
accuracy = []
kf = KFold(n_splits=10)
for train_index, test_index in kf.split(X_scaled):
    # les K folds
    XX_train = X_scaled[train_index]
    yy_train = y[train_index]
    XX_test = X_scaled[test_index]
    yy_test = y[test_index]
    # apprentissage et validation
    logreg.fit(XX_train,yy_train)
    y_pred=logreg.predict(XX_test)
    acc = metrics.accuracy_score(yy_test, y_pred)
    accuracy.append(acc)
    
accuracy_finale = np.mean(accuracy)
print(f"accuracy finale = {accuracy_finale}")
accuracy_finale_var = np.var(accuracy)
print(f"variance de l'accuracy = {accuracy_finale_var}")

# * prédiction test
y_pred=logreg.predict(X_test)
accuracy_test = metrics.accuracy_score(y_test, y_pred)
print(f"accuracy test = {accuracy_test}")
    
    
    





