# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 15:51:27 2017

@author: canda
"""

import numpy as np
import matplotlib.pyplot as mpl
import random
from math import sqrt
from sklearn import datasets
data = datasets.load_iris()
print(data['DESCR'])

# Parametre : data les donnees a diviser, n un entier compris entre [0;100] representant la part de l'ensemble initiale dans l'ensemble d'apprentissage
def diviser_data_set(data, n) :
   # Nbre de donnee que l'on met dans l'ensemble d'apprentissage
    nb = int((50*n)/100)
    dataApp = []
    targetApp = []
    dataTest = []
    targetTest = []
    dataTemp = {}
    for i in range(0,3):
        dataTemp['data'] = data['data'][i*50:i*50+50]
        dataTemp['target'] = data['target'][i*50:i*50+50]
        random.shuffle(dataTemp['data'])
        dataApp.extend(dataTemp['data'][0:nb])
        targetApp.extend(dataTemp['target'][0:nb])
        dataTest.extend(dataTemp['data'][nb:50])
        targetTest.extend(dataTemp['target'][nb:50])
    
    app={'data':dataApp,'target':targetApp}
    test={'data':dataTest,'target':targetTest}
    return app,test

def distance(x,y):
    res = 0
    for i in range(0,len(x)):
        res += ((y[i] - x[i])**2)
    
    return sqrt(res)

# Parametre : data l'ensemble de donnee utilisee, x l'instance que l'on considere et k le nombre de voisins que l'on considere.
def kppv(data,x,k):
    
    dist = []
    for i in range(0,len(data['data'])):
        res = distance(x,data['data'][i])
        dist.append(res)
        
    pos = np.argsort(dist)
    print(pos)
    classe = []
    
    for i in range(0,k):
        idx = np.where(pos == i)
        classe.append(data['target'][idx[0][0]])
    
    print(classe)
    
    c = np.argmax(np.bincount(classe))
    print(c)
    return c
    
def taux_classification(dataApp,dataTest,k):
    cpt = 0
    for i in range(0,len(dataTest['data'])):
        classe = kppv(dataApp,dataTest['data'][i],k)
        
        if classe == dataTest['target'][i]:
            cpt = cpt+1

    res = cpt / len(dataTest['data'])
    return res * 100
    
def repeat():
    taux = []
    dataApp,dataTest = diviser_data_set(data,67)

    for i in range(1,100):
        val = taux_classification(dataApp,dataTest,i)
        taux.append(val)
         
   # for i, elt in enumerate(taux):
    #    print("Pour k = {} le taux de classification vaut {}.".format(i+1, elt))
        
    x = []
    for i in range(1,100):
        x.append(i)
    
    mpl.plot(x,taux)
    mpl.axis([1,100,0,100])
    mpl.show()
        
    return taux
