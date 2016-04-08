#!/usr/local/bin/python3.5
# encoding: utf-8

#TODO: demo #1 using iris data

'''
#TODO:
if num_eigenvalues:
    num_eigenvalues = (m - num_eigenvalues, m-1)
eva, evc = LA.eigh(R, eigvals=num_eigenvalues)

find a suitable projection to the ratio s_bc / s_wc, as large as possible;
in practice,
    (i)  calculate s_bc & s_wc, then
    (ii) calculate w


'''


import sys
import os

import numpy as NP
from scipy import linalg as LA


NP.set_printoptions(precision=3, linewidth=110, suppress=True)




def LDA(data, labels, dim_rescale):
    '''
    Linear Discriminant Analysis
    pass in:
        (i) a raw data array--features encoded in the cols;
            one data instance per row;
        (ii) EV, explanatory variable, is included in D as last column;
        (iii) the LDA flag is set to False so PCA is the default techique;
            if both LDA & EV are set to True then LDA is performed
            instead of PCA
    returns:
        (i) eigenvalues (1D array);
        (ii) eigenvectors (2D array)
        (iii) covariance matrix

    some numerical assertions:

    >>> # sum of the eigenvalues is equal to trace of R
    >>> x = R.trace()
    >>> x1 = eva.sum()
    >>> NP.allclose(x, x1)
    True

    >>> # determinant of R is product of eigenvalues
    >>> q = LA.det(R)
    >>> q1 = NP.prod(eva)
    >>> NP.allclose(q, q1)
    True
    '''
    assert data.shape[0] == labels.shape[0]
    # mean center the data array
    data -= data.mean(axis=0)
    nrow, ndim = data.shape
    # pre-allocate sw, sb arrays (both same shape as covariance matrix)
    # s_wc: array encoding 'within class' scatter
    # s_bc: array encoding 'between class' scatter
    s_wc = NP.zeros((ndim, ndim))
    s_bc = NP.zeros((ndim, ndim))
    R = NP.cov(data.T)
    classes = NP.unique(labels)
    for c in range(len(classes)):
        # create an index only for data rows whose class label = classes[c]
        idx = NP.squeeze(NP.where(labels == classes[c]))
        d = NP.squeeze(data[idx,:])
        class_cov = NP.cov(d.T)
        s_wc += float(idx.shape[0]) / nrow * class_cov
    s_bc = R - s_wc
    # now solve for w then compute the mapped data
    evals, evecs = LA.eig(s_wc, s_bc)
    NP.ascontiguousarray(evals)
    NP.ascontiguousarray(evecs)
    # sort the eigenvectors based on eigenvalues sort order
    idx = NP.argsort(evals)
    idx = idx[::-1]
    evecs = evecs[:,idx]
    # take just number of eigenvectors = dim_rescale
    evecs_dr = evecs[:,:dim_rescale]
    # multiply data array & remaining set of eigenvectors
    rescaled_data = NP.dot(data, evecs_dr)
    return rescaled_data, evecs_dr
