from flappy import FlappyBird
import keras
from keras import layers
import numpy as np
from keras import models
import time
import tensorflow as tf
import matplotlib.pyplot as plt

#%% Deep Q-Learning

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
    res = mlp.predict(state)[0]
    n_action = len(res)
    action_true = np.argmax(res)
    action = alea(n_action,action_true,p)
    if nP.shape == (0,):
        nP  = np.array([res]) 
    else :
        nP = np.append(nP,[res],axis = 0)
    nA = np.append(nA,action)
    return action,nSt,nSc,nP,nA



def add(X,y,valx,valy):
    n = len(X)
    i = np.random.randint(0,n)
    X[i] = valx
    y[i] = valy
    return X ,y

def update(nSt,nSc,nP,nA,mlp,X,y,alpha,mu):
    n = len(nA)-1
    #on commence par gérer l'état final
    Q = nSc[n]
    Q = normalize(Q)
    state = nSt[n]
    action = nA[n]
    #on doit faire la prédiction du dernier état car elle n'a pas été faite 
    preds = mlp.predict(np.reshape(state,(1,len(state))))[0]
    preds[action] = Q
    X,y = add(X,y,state,preds)
    for j in range(n-1):
        i = n - j - 1
        action = nA[i]
        preds = nP[i]
        state = nSt[i]
        result = nSc[i]
        Q = (1-alpha)*preds[action] + alpha*(result + mu*Q)
        Q = normalize(Q)
        preds[action] = Q
        X,y = add(X,y,state,preds)
    return X,y
    
def normalize(Q):
    if Q < 0 :
        Q = 4*Q/1000
    else : 
        Q = 4*Q/100
    return np.tanh(Q)

def fit(X,y):
    X_train = []
    y_train = []
    n = len(X)
    for i in range(n):
        if not (y[i] == [0,0]).all():
            if X_train == []:
                X_train = X[i:i+1]
                y_train = y[i:i+1]
            else :
                X_train = np.append(X_train,X[i:i+1], axis = 0)
                y_train = np.append(y_train,y[i:i+1], axis = 0)
    return X_train,y_train

def decrease(p):
    return 0.95*p

#%% Main 



input_m = keras.Input(shape=(6,)) 
x = layers.Dense(20, activation='tanh')(input_m)
x = layers.Dense(10, activation='tanh')(x)
x = layers.Dense(2, activation='tanh')(x)
mlp = keras.Model(input_m,x)
mlp.summary()
mlp.compile(optimizer='adam', loss='binary_crossentropy')



memoire = 1000
X = np.array([[0. for j in range(6)] for i in range(memoire)])
y = np.array([[0.,0.] for i in range(memoire)])

p = 0.9
loss = []

for iterations in range(20):
    step = 0 
    maxstep = 10000
    state_0 = np.zeros(6)
    nState = np.array([state_0])
    nScore = np.array([])
    nPred = np.array([])
    nAction = np.array([0])
    
    flappy = FlappyBird()
    while True:
        state = np.array(flappy.getState())
        score = flappy.getScore()
        action,nState,nScore,nPred,nAction = act(nState,nScore,nPred,nAction,mlp,state,score,p)
        if action == 1:
            entry = "jump"
        else :
            entry = "stay"
        gameend = flappy.nextFrame(manual=True, entry=entry)
        if not gameend :
            nScore = np.append(nScore,-1000)
            break
        if step > maxstep :
            nScore = np.append(nScore,1000)
            break
    X,y = update(nState,nScore,nPred,nAction,mlp,X,y,0.5,0.8)
    X_train,y_train = fit(X,y)
    history  = mlp.fit(X_train, y_train,epochs=20,batch_size = 20,shuffle = True,verbose=1)
    loss += history.history['loss']
    p = decrease(p)

plt.plot(loss)
mlp.save('./mlp.h5')     
flappy.exit()