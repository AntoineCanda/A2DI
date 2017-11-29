"""
Created on Wed Nov 15 15:51:56 2017

@author: canda
"""

import numpy as np
import matplotlib.pyplot as mpl
import random


# Question 1 : on genere n points aleatoirement dans l'interval [0.0,1.0] 

def generer_points(n):
    points = []
    for i in range(0,n):
        x = np.random.uniform(0.0,1.0)
        y = np.random.uniform(0.0,1.0)
        points.append((x,y))
    
    return points

# Question 2 : On applique l'attribution des classes à chaque point en considerant la formule donnee.
def attribution_classe(points):
    classes = []
    
    for x, y in points:
        if -0.5 * x + 0.75 <= y:
            classes.append(1)
        else :
            classes.append(-1)
    
    return classes

# Question 2 / 3 : Permet la separation des points selon chaque classe A et B 
def separate_classe_point(points,classe):
    classeA = [] 
    classeB = []
    
    for i in range(0,len(points)):
        if classe[i] == 1 :
            classeA.append(points[i])
        else :
            classeB.append(points[i])
    return classeA,classeB

# Question 2 : La fonction permet d'afficher le graphe avec les points ainsi qu'une droite separatrice a partir de theta
def vision_point(points,classe,theta,n):
    classeA,classeB = separate_classe_point(points,classe)
            
    classeAx = []
    classeAy = []
    classeBx = []
    classeBy = []
    
    for i in range(0 , len(classeA)):
        classeAx.append(classeA[i][0])
        classeAy.append(classeA[i][1])
        
    for i in range(0 , len(classeB)):
        classeBx.append(classeB[i][0])
        classeBy.append(classeB[i][1])

        
    
    mpl.plot(classeAx,classeAy,'ro')
    mpl.plot(classeBx,classeBy,'bo')
    mpl.axis([0.0,1.0,0.0,1.0])
    if(not(theta)):
        mpl.show()

    if theta :
        Dapp , targetApp , Dtest, targetTest = separate_dataset(points,classe,n)
        
        w1,w2,b = ptrain(Dapp,targetApp)
        x1 = []
        for i in range(0,len(Dapp)):
            x1.append(Dapp[i][1])
        
        f = lambda x : -((w1*x+b)/w2)
        x2 = [f(x) for x in x1]
        mpl.plot(x1,x2,'r')
        mpl.show()
        
# Question 3 : Separe le dataset en dataset d'apprentissage et de test avec les target associes.
def separate_dataset(point,classe,n):
    classeA,classeB = separate_classe_point(point,classe)
    
    nbA = int((len(classeA)*n)/100)
    nbB = int((len(classeB)*n)/100)
    
    random.shuffle(classeA)
    random.shuffle(classeB)
    
    dataApp = []
    dataTest = []
    targetApp = []
    targetTest = []
    
    dataApp.extend(classeA[0:nbA])
    dataApp.extend(classeB[0:nbB])
    
    for i in range(0,len(classeA)):
        if(i < nbA):
            targetApp.append(1)
        else:
            targetTest.append(1)
            
    for i in range(0,len(classeB)):
        if(i < nbB):
            targetApp.append(-1)
        else:
            targetTest.append(-1)
            
    dataTest.extend(classeA[nbA:])
    dataTest.extend(classeB[nbB:])
    
    return dataApp,targetApp, dataTest, targetTest




# Question 4 : Fonction ptrain prenant ensemble apprentissage (donnee + classe) et calculant la bonen valeur de theta pour toute les valeurs de l'ensemble d'apprentissage selon formule presente dans le poly
def ptrain(data,target):
    theta = np.array([np.random.random(),np.random.random(),np.random.random()])
    end = False
    while(not(end)):
        end = True
        for i in range(0,len(data)):
            vec = np.array(list(data[i])+[1])
            c = ptest(vec,theta)
            if not(c == target[i]):
                theta = theta + (target[i] * vec)
                end = False
    return theta
# Question 5 : Voir fonction vision_point de la question 2

# Question 6 : On utilise la fonction dot pour faire le calcule montré plus tôt dans le sujet.
def ptest(x,theta):
    sign = lambda x : -1 if x < 0 else 1
    return sign(np.dot(theta,x))

# Question 7 : Pas faite
     
# Question 8 : Pas faite