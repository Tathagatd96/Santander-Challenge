# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 16:47:26 2018

@author: Tathagat Dasgupta
"""


import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


santander_data=pd.read_csv("train.csv")
test_df=pd.read_csv('test.csv')
print(santander_data.keys())  #4992 features except target
#print(santander_data.head())

x_data=santander_data.drop(["target",'ID'],axis=1)
y_data=santander_data["target"]

#print(len(x_data.keys()))
#print(x_data.isnull().sum().sort_values(ascending=False))

from sklearn.decomposition import PCA
pca = PCA(n_components=1000)
x_data = pd.DataFrame(pca.fit_transform(x_data))
test_df=pd.DataFrame(pca.fit_transform(test_df))
#print(x_data.head(5))

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.70, random_state=1)

import sklearn.feature_selection
select = sklearn.feature_selection.SelectKBest(k=100)
selected_features = select.fit(X_train, y_train)
indices_selected = selected_features.get_support(indices=True)
colnames_selected = [x_data.columns[i] for i in indices_selected]

X_train_selected = X_train[colnames_selected]
X_test_selected = X_test[colnames_selected]
test_df_selected=test_df[colnames_selected]

from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

regr = RandomForestRegressor(max_depth=5, random_state=0)
rfr=regr.fit(X_train_selected, y_train)
y_pred=rfr.predict(X_test_selected)

def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))


err = rmsle(y_pred,y_test)

print('RMSLE: {:.3f}'.format(err))


y_pred=rfr.predict(test_df_selected)

df=pd.read_csv("sample_submission.csv")
submission = pd.DataFrame({
        "ID": df["ID"],
        "target": y_pred
    })
submission.to_csv('santander1.csv', index=False)