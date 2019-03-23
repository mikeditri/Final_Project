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
X_train, X_test, y_train, y_test = train_test_split(Scaled_data, wine_data.target, test_size=0.3) # 70% training and 30% test

#Import Gaussian Naive Bayes Classifier model
#from sklearn.naive_bayes import GaussianNB
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
from sklearn.metrics import confusion_matrix

#Create Gaussian Naive Bayes Classifier
#gnb = GaussianNB()

#gnb.fit(X_train,y_train)

#predictions = gnb.predict(X_test)

#gnb_eval = cross_val_score(estimator = gnb , X= X_train ,y = y_train , cv= 10)
#accuracy_score = (gnb_eval.mean())
#print(gnb_eval)


#cm = confusion_matrix(y_test , predictions)
#print(cm)

#plt.figure(figsize=(7,7))
#sns.heatmap(cm, annot=True, fmt=".3f",linewidths=.5, square = True, cmap = 'Blues_r')
#cmap = ('Blues_r')
#plt.ylabel('Actual label')
#plt.xlabel('Predicted label')
#plt.title('Accuracy: '+ '{:.0%}'.format(accuracy_score), size = 15)
#plt.savefig('Gaussian_NB_with_Accuracy'+'.png')
#plt.clf()

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)
rfc_eval = cross_val_score(estimator = rfc , X= X_train ,y = y_train , cv= 10)
accuracy_score = (rfc_eval.mean())
print(accuracy_score)

predictions = rfc.predict(X_test)
cm = confusion_matrix(y_test , predictions)
#print(cm)

plt.figure(figsize=(7,7))
sns.heatmap(cm, annot=True, fmt=".3f",linewidths=.5, square = True, cmap = 'Blues_r')
cmap = ('Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.title('Accuracy: '+ '{:.0%}'.format(accuracy_score), size = 15)
plt.show()
plt.savefig('RFC_Accuracy'+'.png')
plt.clf()

importances = rfc.feature_importances_

plt.figure()
plt.title("Feature importances")
plt.barh(range(X_train.shape[1]), importances,
       color="b", align="center")
# If you want to define your own labels,
# change indices to a list of labels on the following line.
plt.yticks(range(X_train.shape[1]),wine_df.columns)
plt.ylim([-1, X_train.shape[1]])
plt.show()
plt.savefig('Feature_Importances_Accuracy'+'.png')
plt.clf()
