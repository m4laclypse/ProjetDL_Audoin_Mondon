# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 09:07:07 2021

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


Lstate = []
LstateNorm = []


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

for iterations in range(3):
    flappy = FlappyBird(graphique = True, FPS = 30)
    while True:
        getstate = flappy.getState()
        Lstate.append(getstate.copy())
        getstate = normalize(getstate)
        LstateNorm.append(getstate.copy())

        gameend = flappy.nextFrame(manual=False, entry=None)
        if gameend != 0 :
            break

 

#%%

Lstate = np.array(Lstate)
LstateNorm = np.array(LstateNorm)

Lmean = [LstateNorm[:,i].mean() for i in range(6)]
Lstd = [LstateNorm[:,i].std() for i in range(6)]

#%%

plt.figure()
plt.subplot(3,2,1)
# plt.plot(Lstate[:,0])
plt.plot(LstateNorm[:,0])
plt.title("val1")

plt.subplot(3,2,2)
# plt.plot(Lstate[:,1])
plt.plot(LstateNorm[:,1])

plt.title("val2")

plt.subplot(3,2,3)
# plt.plot(Lstate[:,2])
plt.plot(LstateNorm[:,2])
plt.title("val3")

plt.subplot(3,2,4)
# plt.plot(Lstate[:,3])
plt.plot(LstateNorm[:,3])

plt.title("val4")

plt.subplot(3,2,5)
# plt.plot(Lstate[:,4])
plt.plot(LstateNorm[:,4])

plt.title("val5")

plt.subplot(3,2,6)
# plt.plot(Lstate[:,5])
plt.plot(LstateNorm[:,5])

plt.title("val6")


flappy.exit()