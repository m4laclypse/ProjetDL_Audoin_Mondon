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
import random
import pickle

#%% Deep Q-Learning

def alea(listeProbaAction,p):
    """
    Permet de forcer aléatoirement des mouvements pour donner des données à l'apprentissage
    """
    r = np.random.random()
    if r < p:
        nr = np.random.random()
        if nr <= 0.055:
            return 1
        else : return 0
    else :
        choice = random.choices([0, 1], weights=listeProbaAction, k=1)[0]
        return choice


def act(nSt,nSc,nP,nA,mlp,state,score,p):
    """
    Ajoute l'état actuel à l'historique et donne la prochaine action à effectuer.
    """
    state = state.reshape((1,len(state)))
    nSt = np.append(nSt,state, axis = 0)
    nSc = np.append(nSc,score)
    res = mlp(state)[0].numpy()  # Plus rapide que predict
    action = alea(res, p)
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


def update(nSt, nSc, nP, nA, mlp, X, y, alpha, mu, indice):
    n = len(nA)-1
    # On commence par gérer l'état final
    Q = nSc[n]
    state = nSt[n]
    action = nA[n]
    # On doit faire la prédiction du dernier état car elle n'a pas été faite 
    preds = mlp.predict(np.reshape(state,(1,len(state))))[0]
    preds[action] = Q
    X, y = add(X,y,state,preds)
    predsprec = preds
    for j in range(n-1):
        i = n - j - 1
        action = nA[i]
        preds = nP[i]
        state = nSt[i]
        result = nSc[i]
        # La formule centrale Q, objectif que l'on cherche à prédire
        Q = (1-alpha)*preds[action] + alpha*(result + mu*max(predsprec))
        preds[action] = Q
        X[indice] = state
        y[indice] = preds
        predsprec = preds
        indice = (indice + 1) % len(X)
    return X, y, indice


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
    val = 0.997
    lim = 0.9
    return val*a+(1-val)*lim


def normalize(getstate):  # Normalise les données empiriquement
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

nelt = 6
nbstates = nelt*4

def getNetwork():
    """
    Génère un nouveau réseau à entraîner
    """
    input_m = keras.Input(shape=(nbstates,)) 
    x = layers.Dense(20, activation='swish')(input_m)
    x = layers.Dense(10, activation='swish')(x)
    x = layers.Dense(20, activation='swish')(x)
    x = layers.Dense(10, activation='swish')(x)
    x = layers.Dense(2, activation="softmax")(x)
    mlp = keras.Model(input_m,x)
    mlp.summary()
    sgd = keras.optimizers.SGD(learning_rate=0.005)
    mlp.compile(optimizer=sgd, loss='mse')
    return mlp


#%% pre data

def pretraining(mlp):
    """
    Fonction de pré-entraînement prenant des données provenant d'une partie réelle pour
    initialiser le réseau. Non utilisé finalement car non probant.
    """
    file = open("entryData.pickle.dat", "br")
    data = pickle.load(file)
    file.close()

    memoire = len(data)
    X = np.array([[0. for _ in range(nbstates)] for _ in range(memoire)])
    y = np.array([[0.,0.] for _ in range(memoire)])

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
    return mlp, history


#%% Main

def getPrevious():
    """
    Permet de charger l'état final du dernier entraînement effectué
    """
    mlp = models.load_model('./mlp.h5')
    file = open("./x.pickle.dat", "br")
    X = pickle.load(file)
    file.close()
    y = np.array([[0.,0.] for _ in X])
    return mlp, X, y

if __name__ == "__main__":
    continuerEntrainement = False
    NOMBREITERATIONS = 1000
    maxstep = 5000

    if continuerEntrainement:
        mlp, X, y = getPrevious()
        p = 0

    else:
        mlp = getNetwork()
        memoire = 1000
        X = np.array([[0. for _ in range(nbstates)] for _ in range(memoire)])
        y = np.array([[0.,0.] for _ in range(memoire)])
        p = 0.9

    loss = []
    valloss = []
    scores = []

    alpha = 0.99
    mu = 0.3

    maxScore = 0

    for iterations in range(NOMBREITERATIONS):
        tjeu = 0
        tpred = 0
        tentr = 0
        t1 = time.time()
        step = 0 
        flappy = FlappyBird(graphique = False, FPS = 300)
        state_0 = np.zeros(nbstates)
        nState = np.array([state_0])
        nScore = np.array([])
        nPred = np.array([])
        nAction = np.array([0])
        survival_points = 0

        state = []
        indice = 0
        while True:
            ta = time.time()
            getstate = flappy.getState()  # Donne la situation actuelle du jeu
            getstate = normalize(getstate)
            tb = time.time()

            if state == []:
                state = np.array(getstate.copy()+getstate.copy()+getstate.copy()+getstate.copy())
            else : 
                state[nelt:] = state[:nbstates-nelt]
                state[:nelt] = np.array(getstate.copy())
                
            score = flappy.getScore()  # Donne le score
            this_score = score
            score += survival_points  # On ajoute un bonus si on reste en vie
            action, nState, nScore, nPred, nAction = act(nState,nScore,nPred,nAction,mlp,state,score,p)
            survival_points += 0.1
            tc = time.time()
            if action == 1:
                entry = "jump"
            else :
                entry = "stay"
            gameend = flappy.nextFrame(manual=True, entry=entry)  # Calcule l'étape suivante du jeu
            if gameend != 0 :  # Si on a perdu le jeu
                print("session: Echec.")
                nScore = np.append(nScore,-2)
                break
            if step > maxstep :  # Si on a joué suffisamment longtemps
                print("session: Succès !")
                nScore = np.append(nScore,2)
                break
            td = time.time()
            tpred += tc - tb
            tjeu = td - tc + ta - tb
            step += 1
        te = time.time()

        if this_score > maxScore:  # On sauvegarde toujours le meilleur modèle pour compenser l'instabilité
            maxScore = this_score
            mlp.save('./best_last.h5')

        X, y, indice = update(nState,nScore,nPred,nAction,mlp,X,y,alpha,mu, indice)
        # On modifie la mémoire X en ajoutant les nouvelles valeurs de Q qu'on a obtenu durant cette partie

        X_f, y_f = fit(X,y)
        
        history  = mlp.fit(X_f, y_f, epochs=2, batch_size = 50, shuffle=True, verbose=0)

        loss += history.history['loss']
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
        print("alpha = "+str(alpha))
        print("")
        p = modp(p)
        alpha = modalpha(alpha)

        if iterations % 200 == 0:
            # On sauvegarde périodiquement les données produites
            mlp.save('./mlp.h5')
            file = open("./loss.pickle.dat", "bw+")
            pickle.dump(loss, file)
            file.close()
            file = open("./valloss.pickle.dat", "bw+")
            pickle.dump(valloss, file)
            file.close()
            file = open("./scores.pickle.dat", "bw+")
            pickle.dump(scores, file)
            file.close()

    file = open("./x.pickle.dat", "bw+")
    pickle.dump(X, file)
    file.close()
    print(maxScore)
    plt.figure()
    plt.plot(loss)
    plt.figure()
    plt.plot(scores)
    plt.show()
    mlp.save('./mlp.h5')
    flappy.exit()
