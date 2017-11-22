# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 15:51:27 2017

@author: canda
"""

import numpy as np
import matplotlib as mpl
import random
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
        dataApp.concat(dataTemp['data'][0:nb])
        targetApp.concat(dataTemp['target'][0:nb])
        dataTest.append(dataTemp['data'][nb:50])
        targetTest.append(dataTemp['target'][nb:50])
    
    app={'data':dataApp,'target':targetApp}
    test={'data':dataTest,'target':targetTest}
    return [app,test]

