# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 16:57:20 2021

@author: Hostl
"""
#these are the required packages for model training
import tqdm, pandas, re, time, pickle 
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


#trainable pickle files
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

#the dataframe to be trained
#when using a pickle file make sure the name is changed as well as the following for gun
df_train= pd.read_pickle("gun.pickle")
name  = 'gun'

#we prepare the data
raw_x = df_train['X'].to_list()
raw_y = df_train['Y'].to_list()
for i in tqdm.trange(len(raw_x)):
    raw_x[i] = re.sub('\s+',' ',raw_x[i])
    
# create the tokenizer
t = Tokenizer()
# fit the tokenizer on the documents
t.fit_on_texts(raw_x)

#finalize the data to be used
length     = len(raw_x)
test_limit = round(length*.10)
test_x = raw_x[0:test_limit]
test_y = raw_y[0:test_limit]
raw_x  = raw_x[test_limit:length]
raw_y  = raw_y[test_limit:length]

#integer encode documents
encoded_x = t.texts_to_matrix(raw_x, mode='count')
test_x = t.texts_to_matrix(test_x, mode='count')
small_x   = encoded_x
small_y   = raw_y

# Save the tokenizer to be used
with open('{}_token.pickle'.format(name), 'wb') as handle:
    pickle.dump(t, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
print(" ")
print("Input Dimensions: {}".format(len(encoded_x[1])))
print('MAKE SURE THE MODEL INPUT_DIM IS {}. If it is fine then simply press'.format(len(encoded_x[1])))
x = input()

#model parameters
batch_size = 1
epochs     = 100

#the model creation function
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(8, input_dim=93287, activation='relu'))
	model.add(Dense(5, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
#initialze the model
estimator = KerasClassifier(build_fn=baseline_model, epochs=epochs, batch_size=batch_size, verbose=1)
#train the model
estimator.fit(small_x,small_y,epochs=epochs, batch_size=batch_size, verbose=1,validation_split=.01)

print(" ")
print("***********Testing***********")
#This is testing on the test set of input data.
y_pred = estimator.predict(test_x)
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
print("Train Length: {}".format(len(raw_x)))
print("Test Length: {}".format(len(test_x)))
print("Batch_size: {}".format(batch_size))
print("Epochs: {}".format(epochs))
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

#save the model
estimator.model.save("{}_".format(name))


