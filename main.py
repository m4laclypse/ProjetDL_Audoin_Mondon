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
    # res = mlp.predict(state)[0]
    res = mlp(state)[0].numpy() #c'est 50 fois plus rapide
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
    state = nSt[n]
    action = nA[n]
    #on doit faire la prédiction du dernier état car elle n'a pas été faite 
    preds = mlp.predict(np.reshape(state,(1,len(state))))[0]
    preds[action] = Q
    X,y = add(X,y,state,preds)
    predsprec = preds
    for j in range(n-1):
        i = n - j - 1
        action = nA[i]
        preds = nP[i]
        state = nSt[i]
        result = nSc[i]
        Q = (1-alpha)*preds[action] + alpha*(result/3 + mu*max(predsprec))
        preds[action] = Q
        X,y = add(X,y,state,preds)
        predsprec = preds
    return X,y


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

def modp(p):
    if p > 0.5 :
        return 0.99*p
    else :
        return 0

def modalpha(a):
    val = 0.993
    lim = 0.2
    return val*a+(1-val)*lim


nelt = 6

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


#%% Reseau

nbstates = nelt*4

input_m = keras.Input(shape=(nbstates,)) 
x = layers.Dense(40, activation='tanh')(input_m)
x = layers.Dense(20, activation='tanh')(x)
x = layers.Dense(2, activation='tanh')(x)
mlp = keras.Model(input_m,x)
mlp.summary()
mlp.compile(optimizer='adam', loss='mae')
mlp.save('./mlp.h5')   

memoire = 5000
X = np.array([[0. for j in range(nbstates)] for i in range(memoire)])
y = np.array([[0.,0.] for i in range(memoire)])
loss = []
valloss = []
scores = []


#%% Main


mlp = models.load_model('./mlp.h5')

p = 0.99
alpha = 0.99
mu = 0.9

for iterations in range(200):
    tjeu = 0
    tpred = 0
    tentr = 0
    t1 = time.time()
    step = 0 
    flappy = FlappyBird(graphique = False, FPS = 300)
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
        action,nState,nScore,nPred,nAction = act(nState,nScore,nPred,nAction,mlp,state,score,p)
        survival_points += 0.02
        tc = time.time()
        if action == 1:
            entry = "jump"
        else :
            entry = "stay"
        gameend = flappy.nextFrame(manual=True, entry=entry)
        if gameend != 0 :
            print("session: Echec.")
            nScore = np.append(nScore,-2)
            break
        if step > maxstep :
            print("session: Succès !")
            nScore = np.append(nScore,2)
            break
        td = time.time()
        tpred += tc - tb
        tjeu = td - tc + ta - tb
    te = time.time()
    X,y = update(nState,nScore,nPred,nAction,mlp,X,y,alpha,mu)
    X_f,y_f = fit(X,y)
    # X_train, X_test, y_train, y_test = train_test_split(X_f, y_f, test_size = 0.3, shuffle = True)
    history  = mlp.fit(X_f, y_f,epochs=5,batch_size = 50,shuffle = True,verbose=0,
                       # validation_data=(X_test, y_test)
                       )
    loss += history.history['loss']
    # valloss += history.history['val_loss']
    scores.append(this_score)
    t2 = time.time()
    tentr = t2 -te
    tot = t2 - t1
    print("session n°"+str(iterations+1))
    print("session de "+str(t2-t1)+" secondes")
    print("dont "+str(round(tjeu/tot*100))+"% de jeu")
    print("dont "+str(round(tpred/tot*100))+"% de prédiction")
    print("dont "+str(round(tentr/tot*100))+"% d'entrainement")
    print("le score est de "+str(score))
    print("p = "+str(p))
    print("aplpha = "+str(alpha))
    print("")
    p = modp(p)
    alpha = modalpha(alpha)


plt.figure()
plt.plot(loss)
plt.plot(valloss)
plt.figure()
plt.plot(scores)
mlp.save('./mlp.h5')     
flappy.exit()
