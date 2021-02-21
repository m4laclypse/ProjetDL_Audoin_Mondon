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
from sklearn.utils.class_weight import compute_class_weight


nelt = 6
nbstates = nelt*4

input_m = keras.Input(shape=(nbstates,)) 
x = layers.Dense(150, activation='tanh')(input_m)
x = layers.Dropout(0.1)(x)
x = layers.Dense(100, activation='tanh')(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(50, activation='tanh')(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(25, activation='tanh')(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(10, activation='tanh')(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(2, activation='tanh')(x)
mlp = keras.Model(input_m,x)
mlp.summary()
sgd = keras.optimizers.SGD(learning_rate=0.05)
mlp.compile(optimizer=sgd, loss='mae')


import pickle
file = open("entryDataTotal.pickle.dat", "br")
data = pickle.load(file)

memoire = len(data)
X = np.array([[0. for j in range(nbstates)] for i in range(memoire)])
y = np.array([[0.,0.] for i in range(memoire)])
z = np.array([0 for i in range(memoire)])

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
        z[i] = 0
    else :
        y[i] = np.array([-1,1])
        z[i] = 1

class_weight = compute_class_weight('balanced', np.unique(z), z)
class_weight = {0: class_weight[0], 1 : class_weight[1] * 10}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, shuffle = True)
history  = mlp.fit(X_train, y_train,epochs=3,batch_size = 50,shuffle = True,verbose=1
                    , validation_data=(X_test, y_test)
                    , class_weight=class_weight
                    )

plt.plot(np.log(history.history["loss"]), label = 'train')
plt.plot(np.log(history.history["val_loss"]), label = 'test')
plt.legend()
plt.show()

mlp.save('./mlp.h5')
