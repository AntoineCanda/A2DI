# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 15:51:27 2017

@author: canda
"""

import numpy as np
import matplotlib.pyplot as mpl
import random
import time
from sklearn import datasets

# Question 1 : chargement des donnees 

data = datasets.load_iris()

# Question 2 : Obtenir informations sur le dataset : 150 exemples, 3 classes , dimension = 4 et type(d) est numeric.

print(data['DESCR'])

# Question 3 : Fonction qui permet de diviser le dataset en 2 dataset d'apprentissage et de teste avec la proportion n dans le dataset d'apprentissage. Les exemples sont divisés en 3 blocs de 50 exemples représentant à chaque fois une classe. Pour avoir de l'aléatoire, on les divise en 3 datasets de 50 que l'on mélange aléatoirement avec une fonction shuffle. Ensuite on récupère les n premiers exemples correspondants au pourcentage souhaité pour l'ensemble d'apprentissage et le reste pour l'ensemble de test. On a donc un partage aléatoire et équivalent des exemples dans les deux sets.

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

# Question 4 : Pour la fonction kppv on commence par définir une fonction calculant la distance euclidienne entre deux points en tenant compte de chaque attribut pour la calculer. Ensuite on va trier la liste des distances (après copie) et ensuite on va chercher les indices dans la liste des distances non triés pour avoir les k premieres distances et on va chercher la classe associee. On calcule ensuite l'histogramme des distributions de classe obtenus pour les k plus proche voisins et on selectionne la classe la plus presente.
# Remarque : En cas de distance équale on a pas les indices suivant le premier : on peut donc se retrouver avec une erreur. En revanche en utilisant les fonctions argsort pour calculer l'ordre de chaque distance et ensuite where pour trouver les k premieres valeurs on a difficilement plus de 75% de bonne classification. On a generalement une classe qui n'apparait quasiment pas (la 1) alors que en procedant ainsi on obtient plus de 90% de bonne classification selon k.

def distance(x,y):
    res = 0
    for i in range(0,len(x)):
        res += ((y[i] - x[i])**2)
    
    return (res**(1/2))

# Parametre : data l'ensemble de donnee utilisee, x l'instance que l'on considere et k le nombre de voisins que l'on considere.
def kppv(data,x,k):
    
    dist = []
    for i in range(0,len(data['data'])):
        res = distance(x,data['data'][i])
        dist.append(res)
    
    dist_triee = list(dist)
    dist_triee.sort()
        
    pos = []
   
    for i in range(0,len(dist)):
        idx = dist.index(dist_triee[i])
        pos.append(idx)
       
    classe = []
    
    for i in range(0,k):
        classe.append(data['target'][pos[i]])
        
    b = np.bincount(classe)
    c = np.argmax(b)
    return c

# Question 5 : On calcule le taux en faisant nombre de bonne classification sur nombre totale de classification. 
    
def taux_classification(dataApp,dataTest,k):
    cpt = 0
    for i in range(0,len(dataTest['data'])):
        classe = kppv(dataApp,dataTest['data'][i],k)
        
        if classe == dataTest['target'][i]:
            cpt = cpt+1

    res = cpt / len(dataTest['data'])
    return res * 100

# Question 6 : On fait varier k de 1 à 100 en tenant compte du fait du nombre d'exemples dans l'ensemble d'apprentissage. 
# Généralement pour des k entre 5 et 10 on a des taux très intéressants. Il arrive qu'avec k très faible (< 3) on ait parfois 100% de bonne classification mais attention vu le tres faible nombre de voisins considere il peut s'agir juste d'une coincidence.
    
def classifier_var_k(l):
    taux = []
    dataApp,dataTest = diviser_data_set(data,l)

    n = min(len(dataApp['data'])+1,101)
    
    for i in range(1,n):
        val = taux_classification(dataApp,dataTest,i)
        taux.append(val)
        
    
    for i, elt in enumerate(taux):
        print("Pour k = {} le taux de classification vaut {}.".format(i+1, elt))
        
    x = []
    for i in range(1,n):
        x.append(i)
    
    mpl.plot(x,taux)
    mpl.axis([1,n,0,100])
    mpl.xlabel('Nombre de voisins considérés')
    mpl.ylabel('Pourcentage de bonnes classifications')
    mpl.show()
        
    return taux

# Question 7 : On remarque qu'on a un pic aux alentours de 55%/60% dans le temps d'execution ce qui peut être interprete comme suit : c'est entre les proportions entre les dataset les plus proches qu'on est le plus penalise (a l'instar du fait que plus deux valeurs sont proche et plus leur produit est improtant (5*5 > 1*9)).
# Apres le temps mis pour classifier n'est pas non plus tres eleves, il peut donc avoir un certain intérêt.
def my_range(start, end, step):
    while start <= end:
        yield start
        start += step
        
def classifier_var_dataApp():
    t = []

    for i in my_range(20,95,5):
        t0 = 0
        t1 = 0
        tm = 0
        for j in range(0,100):
            t0 = time.time()
            dataApp,dataTest = diviser_data_set(data,i)
            taux_classification(dataApp,dataTest,15)
            t1 = time.time()
            tm = tm +(t1-t0)
        t.append(tm/15)
        
    x = []
    for i in my_range(20,95,5):
        x.append(i)
        
    mpl.plot(x,t)
    mpl.axis([20,95,0,max(t)])
    mpl.xlabel('Pourcentage du dataset dans le dataset d\'apprentissage')
    mpl.ylabel('Temps moyen pour un calcul de taux')
    mpl.show()