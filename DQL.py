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

nState = np.array([state_0])
nScore = np.array([])

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

def act(nSt,nSc,mlp,state,score,alea,p):
    state = state.reshape((1,len(state)))
    nSt = np.append(nSt,state, axis = 0)
    nSc = np.append(nSc,score)
    res = mlp.predict(state)[0]
    n_action = len(res)
    action_true = np.argmax(res)
    action = alea(n_action,action_true,p)
    return action,nSt,nSc
    

def alea(n,a,p):
    r = np.random.random()
    if r < p:
        return np.random.randint(0,n)
    else :
        return a
    
a,nState,nScore = act(nState,nScore,mlp,state_0,0,alea,1)
a,nState,nScore = act(nState,nScore,mlp,state_0,1,alea,1)
a,nState,nScore = act(nState,nScore,mlp,state_0,2,alea,1)


memoire = 10000
X = [[] for i in range(memoire)]
y = [-1 for i in range(memoire)]



