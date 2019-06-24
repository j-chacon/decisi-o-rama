# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 12:42:31 2019

@author: jchaconhurtado
"""

import numpy as np

UTILITY_THRESHOLD = 0.00001

def pareto_fronts(M, minimize=True):
    '''function to calculate the pareto fronts'''
    if minimize is True:
        i_dominates_j = np.all(M[:,None] <= M, axis=-1) & np.any(M[:,None] < M, axis=-1)
    else:
        i_dominates_j = np.all(M[:,None] >= M, axis=-1) & np.any(M[:,None] > M, axis=-1)
    remaining = np.arange(len(M))
    fronts = np.empty(len(M), int)
    frontier_index = 0
    while remaining.size > 0:
        dominated = np.any(i_dominates_j[remaining[:,None], remaining], axis=0)
        fronts[remaining[~dominated]] = frontier_index

        remaining = remaining[dominated]
        frontier_index += 1
    return fronts

def pareto_front_i(M, minimize=True, i=0):
    '''Function to get a specific pareto set. i=0 means optimal'''
    pfs = pareto_fronts(M, minimize)
    return np.where(pfs == i)[0]

def core_index(sols_inp, pf):
    '''calculate the core index. takes solutions and position of pareto-solutions'''
    return np.mean(sols_inp[pf,:], axis=0)

#def pref_additive(sols, w, w_norm=False):
#    '''model preferences using additive model. takes two vectors'''
#    if np.sum(w) != 1.0 and w_norm is False:
#        raise ValueError('weights have to be normalised to 1')
#    elif w_norm:
#        w = w / np.sum(w, axis=0)
#    
#    out = np.zeros([sols.shape[1], sols.shape[2]])
#    for i in range(sols.shape[1]):
#        out[i,:] = np.sum(sols[:,i,:]*w, axis=0)
#        
#    return out
#
#def pref_cobb_douglas(sols, w, w_norm=False):
#    '''model preferences using cobb-douglas method. takes two vectors'''
#    # Test this function
#    if np.sum(w) != 1.0 and w_norm is False:
#        raise ValueError('weights have to be normalised to 1')
#    elif w_norm:
#        w = w / np.sum(w, axis=0)
#    
#    out = np.zeros([sols.shape[1], sols.shape[2]])
#    for i in range(sols.shape[1]):
#        out[i,:] = np.prod(sols[:,i,:]**w, axis=0)
#        
#    return out
#
#def util_exponential(v, r):
#    '''calculate exponential utility'''
#    if r == 0.0:
#        out = v
#    else:
#        out = (1.0 - np.exp(-r*v)) / (1.0 - np.exp(-r))
#    return out
#
#def agg_hierarchical(sols, w, alpha):
#    '''makes hierarchical aggregation'''
#    add = pref_additive(sols, w)
#    cd = pref_cobb_douglas(sols, w)
#    return alpha*(add) + (1.0 - alpha)*cd


