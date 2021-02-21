# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 16:34:32 2021

@author: loic9
"""


from flappy import FlappyBird
import keras
from keras import layers
import numpy as np
from keras import models
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import time as time 
from sklearn.model_selection import train_test_split


nelt = 6
nbstates = nelt*4

input_m = keras.Input(shape=(nbstates,)) 
x = layers.Dense(40, activation='tanh')(input_m)
x = layers.Dense(20, activation='tanh')(x)
x = layers.Dense(2, activation='tanh')(x)
mlp = keras.Model(input_m,x)
mlp.summary()
mlp.compile(optimizer='adam', loss='mae')
mlp.save('./mlp.h5')   


import pickle
file = open("entryData.pickle.dat", "br")
data = pickle.load(file)

memoire = len(data)
X = np.array([[0. for j in range(nbstates)] for i in range(memoire)])
y = np.array([[0.,0.] for i in range(memoire)])

state = []
for i in range(memoire):
    getstate = data[i][0]
    if state == []:
            state = np.array(getstate.copy()+getstate.copy()+getstate.copy()+getstate.copy())
    else : 
        state[nelt:] = state[:nbstates-nelt]
        state[:nelt] = np.array(getstate.copy()) 
    X[i] = state
    if data[i][1] == 'stay':
        y[i] = np.array([1,-1])
    else :
        y[i] = np.array([-1,1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, shuffle = True)
history  = mlp.fit(X_train, y_train,epochs=40,batch_size = 50,shuffle = True,verbose=1,
                    validation_data=(X_test, y_test)
                    )

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])