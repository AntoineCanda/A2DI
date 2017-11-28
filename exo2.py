"""
Created on Wed Nov 15 15:51:56 2017

@author: canda
"""

import numpy as np
import matplotlib as mpl
from math import sqrt
from sklearn import datasets

def generer_points(n):
    points = []
    for i in range(0,n):
        x = np.random.uniform(0.0,1.0)
        y = np.random.uniform(0.0,1.0)
        points.append((x,y))
    
    return points

def attribution_classe(points):
    classes = []
    
    for x, y in points:
        if -0.5 * x + 0.75 <= y:
            classes.append(1)
        else :
            classes.append(-1)
    
    return classes


def ptrain(data):
    theta = np.array([np.random.random(),np.random.random(),np.random.random()])
    end = False
    
    while not(end):
        end = True
        for


def ptest(x,theta):
    sign = lambda x : -1 if x < 0 else 1
    return sign(np.dot(theta,x))
