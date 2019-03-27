#!/usr/bin/env python
import sklearn.datasets
from sklearn.datasets import load_wine
wine_data = load_wine()
import numpy as np
import pandas as pd
wine_df = pd.DataFrame(data=np.c_[wine_data['data'],wine_data['target']],columns=wine_data['feature_names']+['Class'])
wine_df = wine_df.rename(columns = {'od280/od315_of_diluted_wines':'od280_od315_of_diluted_wines'})
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn import preprocessing

#preprocessing
scaler = preprocessing.StandardScaler()
Scaled_data = scaler.fit_transform(wine_data.data)

# Split dataset into training set and test set

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedKFold

kf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=None) 

accuracy_score_list =[]
confusion_matrix_list = []

for train_index, test_index in kf.split(Scaled_data):
      X_train, X_test = Scaled_data[train_index], Scaled_data[test_index] 
      y_train, y_test = wine_data.target[train_index], wine_data.target[test_index]
      rfc = RandomForestClassifier(n_estimators=200)
      rfc.fit(X_train, y_train)
      pred_rfc = rfc.predict(X_test)
      train_rfc = rfc.predict(X_train)
      accuracy_score_list.append(accuracy_score(y_test,pred_rfc))
      cm = confusion_matrix(y_test,pred_rfc)
      confusion_matrix_list.append(cm)

##print((accuracy_score_list))
##print((confusion_matrix_list))
##print(np.mean(accuracy_score_list))
##print(np.mean(confusion_matrix_list, axis = 0))

#mycviterator = []
#for i in range(10):
#	X_train, X_test, y_train, y_test = train_test_split(Scaled_data, wine_data.target, test_size=0.3) # 70% training and 30% test
#	rfc = RandomForestClassifier(n_estimators=200)
#	rfc.fit(X_train, y_train)
#	pred_rfc = rfc.predict(X_test)
#	rfc_eval = cross_val_score(estimator = rfc , X= X_train ,y = y_train , cv= 10)
#	accuracy_score = (rfc_eval.mean())
#	print(accuracy_score)
#	cm = confusion_matrix(y_test , pred_rfc)

##rfc = RandomForestClassifier(n_estimators=200)
##rfc.fit(X_train, y_train)
##pred_rfc = rfc.predict(X_test)
##rfc_eval = cross_val_score(estimator = rfc , X= X_train ,y = y_train , cv= 10)
##accuracy_score = (rfc_eval.mean())
##print(accuracy_score)
##cm = confusion_matrix(y_test , pred_rfc)
#print(rfc_eval)

plt.figure(figsize=(7,7))
sns.heatmap(np.mean(confusion_matrix_list, axis = 0), annot=True, fmt=".3f",linewidths=.5, square = True, cmap = 'Blues_r')
cmap = ('Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.title('Accuracy: '+ '{:.0%}'.format(np.mean(accuracy_score_list)), size = 15)
plt.savefig('RFC_Accuracy'+'.png')
plt.clf()

importances = rfc.feature_importances_

X_train, X_test, y_train, y_test = train_test_split(Scaled_data, wine_data.target, test_size=0.3) # 70% training and 30% test

plt.figure(figsize=(20,10))
plt.title("Feature importances")
plt.barh(range(X_train.shape[1]), importances,
       color="b", align="center")
# If you want to define your own labels,
# change indices to a list of labels on the following line.
plt.yticks(range(X_train.shape[1]),wine_df.columns)
plt.ylim([-1, X_train.shape[1]])
plt.savefig('Feature_Importances_Accuracy'+'.png')
plt.clf()
