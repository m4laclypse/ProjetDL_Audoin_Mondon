# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 14:47:03 2021

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

def alea(n,a,p):
    r = np.random.random()
    if r < p:
        nr = np.random.random()
        if nr < 0.07 : return 1
        else : return 0
    else :
        return a

def act(nSt,nSc,nP,nA,mlp,state,score,p):
    state = state.reshape((1,len(state)))
    nSt = np.append(nSt,state, axis = 0)
    nSc = np.append(nSc,score)
    # res = mlp.predict(state)[0]
    res = mlp(state)[0].numpy() #♣c'est 50 fois plus rapide
    n_action = len(res)
    action_true = np.argmax(res)
    action = alea(n_action,action_true,p)
    if nP.shape == (0,):
        nP  = np.array([res]) 
    else :
        nP = np.append(nP,[res],axis = 0)
    nA = np.append(nA,action)
    return action,nSt,nSc,nP,nA


def normalize(getstate):
    Lmean = [5.870e-01, 2.158e-01, 2.890e-02, 2.235e-01, 4.622e-01, -4.000e-04]
    Lstd = [0.1286, 0.1291, 0.13, 0.1465, 0.2003, 0.388 ]
    if getstate[2] >= 0.45 :
        getstate[2] = 0.0
    for i in range(6):
        getstate[i] = (getstate[i]-Lmean[i])/Lstd[i]
    if getstate[4] <= -1.75 :
        getstate[4] = 0.0
    return getstate


mlp = models.load_model('./mlp.h5')
nelt = 6
nbstates = nelt*4
scores = []


for iterations in range(5):
    step = 0 
    flappy = FlappyBird(graphique = True, FPS = 30)
    maxstep = 1000
    state_0 = np.zeros(nbstates)
    nState = np.array([state_0])
    nScore = np.array([])
    nPred = np.array([])
    nAction = np.array([0])
    survival_points = 0
    
    state = []
    while True:
        ta = time.time()
        getstate = flappy.getState()
        getstate = normalize(getstate)
        tb = time.time()
        if state == []:
            state = np.array(getstate.copy()+getstate.copy()+getstate.copy()+getstate.copy())
        else : 
            state[nelt:] = state[:nbstates-nelt]
            state[:nelt] = np.array(getstate.copy())
            
        score = flappy.getScore()
        this_score = score
        score += survival_points
        action,nState,nScore,nPred,nAction = act(nState,nScore,nPred,nAction,mlp,state,score,0)
        survival_points += 0.00
        tc = time.time()
        if action == 1:
            entry = "jump"
        else :
            entry = "stay"
        gameend = flappy.nextFrame(manual=True, entry=entry)
        if gameend != 0 :
            nScore = np.append(nScore,-2)
            break
        if step > maxstep :
            nScore = np.append(nScore,2)
            break
    scores.append(this_score)
        
plt.plot(scores)
flappy.exit()