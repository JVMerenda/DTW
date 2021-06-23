#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 04:39:31 2021

@author: lab
"""
import cv2
import scipy, scipy.ndimage
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  
from numpy import mean
from numpy import std

def image_reader(name):
    img = cv2.imread(name, 0) #since the image is grayscale, we need only one channel and the value '0' indicates just that
    return img

def neighborhood(matrix, pt):
    x = pt[0]   #linha
    y = pt[1]   #coluna
    neighbors = []
    if(x-1 >= 0):
        N = (x-1, y, matrix[x-1][y])
        neighbors.append(N)
    if(x-1 >= 0 and y+1 < len(matrix)):
        NE = (x-1, y+1, matrix[x-1][y+1])
        neighbors.append(NE)
    if(y+1 < len(matrix)):
        E = (x, y+1, matrix[x][y+1])
        neighbors.append(E)
    if(x+1 < len(matrix) and y+1 < len(matrix)):
        SE = (x+1, y+1, matrix[x+1][y+1])
        neighbors.append(SE)
    if(x+1 < len(matrix)):
        S = (x+1, y, matrix[x+1][y])
        neighbors.append(S)
    if(x+1 < len(matrix) and y-1>=0):
        SW = (x+1, y-1, matrix[x+1][y-1])
        neighbors.append(SW)
    if(y-1 >= 0):
        W = (x, y-1, matrix[x][y-1])
        neighbors.append(W)
    if(x-1 >= 0 and y-1 >=0):
        NW = (x-1, y-1, matrix[x-1][y-1])
        neighbors.append(NW)
        
    return neighbors

def memory(pt, u, visited):
    memoria = []
    if(len(visited) <= u):
        memoria = visited
    else:
        memoria = visited[-u:]
    return memoria 

def step(pt, neighbors, memoria, rule):
    L = []
    for i in neighbors:
        L.append(abs(pt[2] - i[2]))
    if(rule =="min"):
        minimus = min(L)
        indice = L.index(minimus)
        if(neighbors[indice] not in memoria):
            pt = neighbors[indice]
        else:
            sorte = L
            sorte.sort()
            indice = L.index(sorte[1])
            pt = neighbors[indice]
    if(rule == "max"):
        maximus = max(L)
        indice = L.index(maximus)
        if(neighbors[indice] not in memoria):
            pt = neighbors[indice]
        else:
            sorte = L
            sorte.sort(reverse=True)
            indice = L.index(sorte[1])
            pt = neighbors[indice]
    return pt


    
def walk(matrix, bg, u, r):
    visited = []
    visited.append(bg)
    pt = bg
    stop = True
    while stop:
        memoria = memory(pt, u, visited)
        neighbors = neighborhood(matrix, pt)
        pt = step(pt, neighbors, memoria, r)
        visited.append(pt)
        for i in visited:
            if (visited.count(i) >= 3):
                stop = False
                j = visited.index(i)
                transient = j
                A = [k for k, n in enumerate(visited) if n == i][1]
                p = A - j
                break
    return (visited, (transient, p))

def all_image(img_matrix):
    pp = []
    for i in range(len(img_matrix)):
        for j in range(len(img_matrix)):
            pp.append((i, j, img_matrix[i][j]))
    return pp

def tourist(initial_points, matrix, u, r):
    trajectories = []
    sizes = []
    vec = []
    psi = []
    for bg in initial_points:
        
        tupla = walk(matrix, bg, u, r)
        trajectories.append(tupla[0])
        sizes.append(tupla[1])
        
    for transient in range(0, 5):
        for p in range(u+1, u+6):
            S = (transient, p, sizes.count((transient, p)))
            vec.append(S)
    for e in vec:
        if(e[0] == 0 and e[1] == u+1):
            psi.append(e[2])
        if(e[0] == 1 and e[1] == u+1):
            psi.append(e[2])
        if(e[0] == 2 and e[1] == u+1):
            psi.append(e[2])
        if(e[0] == 3 and e[1] == u+1):
            psi.append(e[2])
        if(e[0] == 4 and e[1] == u+1):
            psi.append(e[2])
    
    return psi

def name_of_class(n):
    if(n < 10):
        name = 'c00'+str(n)+'_'
    elif(n<100):
        name = 'c0'+str(n)+'_'
    else:
        name = str(n)+'_'
    return name

def name_of_image(n, m):
    classe = name_of_class(n)
    if(m < 10):
        name = classe+'00'+str(m)+'.png'
    else:
        name = classe+'0'+str(m)+'.png'
    return name

def LDA(train, w, test, expected):
    X = np.array(train)
    y = np.array(w)
    test_vec = np.array(test)
    exp_vec = np.array(expected)
    # define model
    model = LinearDiscriminantAnalysis()
    # define model evaluation method
    model.fit(X, y)
    pred = model.predict(test_vec)
    count = 0
    for i in range(len(pred)):
        if(pred[i] == exp_vec[i]):
            count += 1
    precision = count/len(exp_vec)
    print("precision: %.3f" %precision)
        
        
    
train = []
test = []
w = []
expected = []
for n in range(1, 112):
    print('Class'+str(n))
    for m in range(1, 17):
        name = name_of_image(n, m)
        images = image_reader(name)
        print(name)
        initial_points = all_image(images)
        phi_min = []
        for u in range(1,5):
            psi_min = tourist(initial_points, images, u, "min")
            phi_min.extend(psi_min)
        if(m <= 10):
            train.append(phi_min)
            w.append(n)
        else:
            test.append(phi_min)
            expected.append(n)
arq = open('savemax.txt', 'w')
arq.write('train:')
arq.write(str(train))
arq.write("classification: ")
arq.write(str(w))
arq.write("test: ")
arq.write(str(test))
arq.write("Expected: ")
arq.write(str(expected))
arq.close()

LDA(train, w, test, expected)