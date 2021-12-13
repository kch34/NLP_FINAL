# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 17:45:24 2021

@author: Hostl
"""
#these are the required packages for model testing
import tqdm, pandas, re, time, pickle, keras
import pandas as pd
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, model_from_json
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import cross_val_score, KFold, cross_val_predict, RepeatedKFold, cross_val_score
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.preprocessing.text import Tokenizer

#testable pickle files
"""
"covid.pickle"
"election_fraud.pickle"
"election_fraud_5yr.pickle"
"gun.pickle"
"gun_5yr.pickle"
"police.pickle"
"police_5yr.pickle"
"2021_data.pickle"
"2020_data.pickle"
"vaccine.pickle"
"vaccine_5yr.pickle"
"""

#the trained dataset
name  = 'gun'
#the testing dataset
test  = 'police'
#load in the trained model
estimator =  keras.models.load_model('{}_'.format(name))
#load in the dataset to be tested
df_test = pd.read_pickle("{}.pickle".format(test))

#split the data 
test_x = df_test['X'].to_list()
test_y = df_test['Y'].to_list()
for i in tqdm.trange(len(test_x)):
    test_x[i] = re.sub('\s+',' ',test_x[i])
#load in the tokenizer
with open('{}_token.pickle'.format(name), 'rb') as handle:
    t = pickle.load(handle)
# integer encode documents
test_x = t.texts_to_matrix(test_x, mode='count')


print(" ")
print("***********{} Testing***********".format(test))
y_pred = estimator.predict_classes(test_x)
matrix = confusion_matrix(test_y, y_pred,normalize='true')
matrix2 = confusion_matrix(test_y, y_pred)
score = 0
for i in range(len(test_y)):
    if test_y[i] == y_pred[i]:
        score = score+1
score = score/len(test_y)
score = round(score*100,2)
print(matrix2)
print("Test Accuracy: {}%".format(score))
print("Train Length: {}".format(len(test_x)))
num = 5

for i in range(num):
    for j in range(num):
        temp = matrix[i][j] *100
        temp = round(temp,2)
        matrix[i][j] = temp
    
df_cm = pd.DataFrame(matrix, range(num), range(num))
plt.figure(figsize=(10,7))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16},fmt='g') # font size
plt.show()
