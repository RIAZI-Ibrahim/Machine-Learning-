#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 09:11:31 2020

@author: Cecile Capponi
ONLY BINARY CLASSIFICATION IS CONCERNED, WITHIN REAL 2D INPUT SPACE
WEAK LEARNERS MUST BE STUMPS
"""

from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import numpy as np

# Surface of decision is a rectagle defined by learned stumps (at least 4)
# each rectangle will receive a color adapted to the class predicted for
# points in it.
class Rectangle:
    def __init__(self, xmin, xmax, ymin, ymax): # limits of the rectangle 
        self.ptopleft_ = (xmin,ymin)
        self.ptopright_ = (xmax, ymin)
        self.pbotleft_ = (xmin, ymax)
        self.pbotright_ = (xmax, ymax)
        self.center_ = np.array([(xmax+xmin)/2, (ymax+ymin)/2])
        
    def __str__(self):
        return str(self.center_)+' -- '+str(self.class_)
        
    
    def set_class(self, c):
        self.class_ = c
        
    def tox(self):
        return [self.ptopleft_[0], self.pbotleft_[0], 
                self.pbotright_[0], self.ptopright_[0]]
        
    def toy(self):
        return [self.ptopleft_[1], self.pbotleft_[1], 
                self.pbotright_[1], self.ptopright_[1]]
        
# if considering the stump learned at iteration t of classifier clf, 
# this function returns on which component (0 or 1) the test was made
# and what is the threshold on that component
def getStump(clf, t):
    # to be implemented
    pass
    
# Generates the rectangles of decisions
def generateZones(clf,limitsx, limitsy, T, process_all=1):
    # A list of two lists (thresholds on first then second component)
    TX =[] 
    TX.append([limitsx[0]])
    TX.append([limitsy[0]])
    # getting the weak separators given by stumps
    if process_all:
        for ite in range(T):
            stump = getStump(clf,ite)
            TX[stump[0]].append(stump[1])
    else:
        stump = getStump(clf,T)
        TX[stump[0]].append(stump[1])
    TX[0].append(limitsx[1])
    TX[1].append(limitsy[1])
    # sorting
    for i in [0,1]: 
        TX[i] = np.array(TX[i])
        TX[i].sort()
    # list of rectangles to be colored
    R = []
    for yt in range(TX[1].shape[0]-1):
        for xt in range(TX[0].shape[0]-1):
            r = Rectangle(TX[0][xt], TX[0][xt+1], TX[1][yt], TX[1][yt+1])
            R.append(r)
    return R
