# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 14:13:32 2021

@author: loic9
"""


import keras
from keras import layers
import numpy as np
from keras import models
import time
import tensorflow as tf

state_0 = np.zeros(42)
nX = np.array([state_0])
ny = np.array([0])

def add(X,val):
    n = len(X)
    i = np.random.randint(0,n)
    X[i] = val
    return X 

input_m = keras.Input(shape=(42,)) 
x = layers.Dense(40, activation='tanh')(input_m)
x = layers.Dense(20, activation='tanh')(x)
x = layers.Dense(2, activation='tanh')(x)
mlp = keras.Model(input_m,x)
mlp.summary()
mlp.compile(optimizer='adam', loss='binary_crossentropy')
mlp.save('./mlp.h5')

def act(nX,ny,mlp,state,score,alea,p):
    state = state.reshape((1,len(state)))
    nX = np.append(nX,state, axis = 0)
    res = mlp.predict(state)[0]
    n_action = len(res)
    action_true = np.argmax(res)
    action = alea(n_action,action_true,p)
    y = res[action]
    ny[-1] += score
    ny = np.append(ny,y)
    return action,nX,ny
    

def alea(n,a,p):
    r = np.random.random()
    if r < p:
        return np.random.randint(0,n)
    else :
        return a

a = np.zeros(42)
ac,nX,ny = act(nX,ny,mlp,a,0,alea,0)
ac,nX,ny = act(nX,ny,mlp,a,1,alea,0)
print(act(nX,ny,mlp,a,1,alea,0))

memoire = 10 
X = [[] for i in range(memoire)]
y = [-1 for i in range(memoire)]

