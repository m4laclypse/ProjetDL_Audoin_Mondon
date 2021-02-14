from flappy import FlappyBird
import keras
from keras import layers
import numpy as np
from keras import models
import time
import tensorflow as tf


#%% Deep Q-Learning

def alea(n,a,p):
    r = np.random.random()
    if r < p:
        return np.random.randint(0,n)
    else :
        return a
    
def act(nSt,nSc,nP,mlp,state,score,p):
    state = state.reshape((1,len(state)))
    nSt = np.append(nSt,state, axis = 0)
    nSc = np.append(nSc,score)
    res = mlp.predict(state)[0]
    n_action = len(res)
    action_true = np.argmax(res)
    action = alea(n_action,action_true,p)
    nP = np.append(nP,res[action])
    return action,nSt,nSc,nP



def add(X,val):
    n = len(X)
    i = np.random.randint(0,n)
    X[i] = val
    return X 
    
    

#%% Main 

state_0 = np.zeros(6)

nState = np.array([state_0])
nScore = np.array([0])
nPred = np.array(0)

input_m = keras.Input(shape=(6,)) 
x = layers.Dense(40, activation='tanh')(input_m)
x = layers.Dense(20, activation='tanh')(x)
x = layers.Dense(2, activation='tanh')(x)
mlp = keras.Model(input_m,x)
mlp.summary()
mlp.compile(optimizer='adam', loss='binary_crossentropy')
mlp.save('./mlp.h5')


memoire = 10000
X = [[] for i in range(memoire)]
y = [-1 for i in range(memoire)]


flappy = FlappyBird()
while True:
    # state = np.array(flappy.getState())
    # score = flappy.getScore()
    # p = 1
    # action,nState,nScore,nPred = act(nState,nScore,nPred,mlp,state,score,p)
    # if action == 1:
    #     entry = "jump"
    # else :
    #     entry = "stay"
    entry = None
    gameend = flappy.nextFrame(manual=False, entry=entry)
    if not gameend :
        flappy.exit()
        
