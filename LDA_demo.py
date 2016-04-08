#!/usr/local/bin/python3.5
# encoding: utf-8

import sys
import os
import numpy as NP
from numpy import ma as MA
from scipy import linalg as LA
from matplotlib import pyplot as MPL
from mpl_toolkits import axes_grid1 as AG
from mpl_toolkits.mplot3d import Axes3D
from LDA import LDA as LDA

NP.set_printoptions(precision=3, linewidth=85, suppress=True)

# fname = os.path.expanduser("~/SpiderOak Hive/LDA/authorship.csv")
fname = os.path.expanduser("~/Documents/authorship.csv")

with open(fname, 'rt') as fh:
    data = [ row.strip().split(',') for row in fh.readlines() ]

col_headers = data.pop(0)
unique_class_labels = {row[-1] for row in data[1:]}
tx = [ label for label in enumerate(unique_class_labels)]
LuT = dict([ (v, k) for k, v in tx ])
fnx = lambda k : LuT[k]
labels = NP.array([fnx(row[-1]) for row in data], dtype=int)

# remove BookID & class label columns
data = NP.array([row[:-2] for row in data], dtype=float)


assert data.shape[0] == labels.shape[0]

# shuffle the data & labels
idx= NP.arange(data.shape[0])
NP.random.shuffle(idx)
data = data[idx,]
labels = labels[idx]


# set number of dimensions in rescaled data
dim_rescale = 3

# {'London': 0, 'Austen': 1, 'Milton': 2, 'Shakespeare': 3}
ndx0 = labels==0
ndx1 = labels==1
ndx2 = labels==2
ndx3 = labels==3

rescaled_data, w = LDA(data, labels, dim_rescale)

assert NP.sum(ndx0) + NP.sum(ndx1) + NP.sum(ndx2) + NP.sum(ndx3) == data.shape[0]

class0 = rescaled_data[ndx0,]
class1 = rescaled_data[ndx1,]
class2 = rescaled_data[ndx2,]
class3 = rescaled_data[ndx3,]

#----------------------- plotting ----------------------#

x0, y0, z0 = data[:,0], data[:,1], data[:,2]
x1, y1, z1 = class0[:,0], class0[:,1], class0[:,2]
x2, y2, z2 = class1[:,0], class1[:,1], class1[:,2]
x3, y3, z3 = class2[:,0], class2[:,1], class2[:,2]
x4, y4, z4 = class3[:,0], class3[:,1], class3[:,2]

x0lo, x0hi = NP.floor(NP.min(data[:,0]))-.25, NP.ceil(NP.max(data[:,0]))+.25
y0lo, y0hi = NP.floor(NP.min(data[:,1]))-.25, NP.ceil(NP.max(data[:,1]))+.25

x1lo = NP.floor(NP.min(rescaled_data[:,0]))-.25
x1hi = NP.ceil( NP.max(rescaled_data[:,0]))+.25
y1lo = NP.floor(NP.min(rescaled_data[:,1]))-.25
y1hi = NP.ceil(NP.max(rescaled_data[:,1]))+.25

clrs = "#006A4E #2E5894 #CC5500 #91A3B0".split()

fig = MPL.figure(1)
ax1 = Axes3D(fig)

ax1.plot(x1, y1, z1, '.', mfc=clrs[0], mec=clrs[0])
ax1.plot(x2, y2, z2, '.', mfc=clrs[1], mec=clrs[1])
ax1.plot(x3, y3, z3, '.', mfc=clrs[2], mec=clrs[2])
ax1.plot(x4, y4, z4, '.', mfc=clrs[3], mec=clrs[3])


MPL.show()
