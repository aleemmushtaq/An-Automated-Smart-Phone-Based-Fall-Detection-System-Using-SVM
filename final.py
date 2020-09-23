from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
df_T = loadmat('Dataset_App_T_new.mat')
df_W = loadmat('Dataset_App_W_new2.mat')
X = df_T['Xnew']
y = df_T['ynew']

X[0:5]
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

from sklearn import svm
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train) 

yhat = clf.predict(X_test)

from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(y_test, yhat)
