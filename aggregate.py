# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:08:56 2019

@author: jchaconhurtado

Aggregation models

http://www.informatica.si/ojs-2.4.3/index.php/informatica/article/view/1321/972

"""
import numpy as np

def additive(sols, w=None, w_norm=False):
    '''aggregate preferences using additive model. takes two vectors'''
    if w is None:
        w = np.ones(sols.shape[1]) / sols.shape[1]
    
    if np.sum(w) != 1.0 and w_norm is False:
        raise ValueError('weights have to be normalised to 1')
    elif w_norm:
        w = w / np.sum(w, axis=0)
    
    out = np.zeros([sols.shape[1], sols.shape[2]])
    for i in range(sols.shape[1]):
        out[i,:] = np.sum(sols[:,i,:]*w, axis=0)
        
    return out

# also known as the geometric mean operator
def cobb_douglas(sols, w, w_norm=False):
    '''aggregate preferences using cobb-douglas method. takes two vectors'''
    # Test this function
    if np.sum(w) != 1.0 and w_norm is False:
        raise ValueError('weights have to be normalised to 1')
    elif w_norm:
        w = w / np.sum(w, axis=0)
    
    out = np.zeros([sols.shape[1], sols.shape[2]])
    for i in range(sols.shape[1]):
        out[i,:] = np.prod(sols[:,i,:]**w, axis=0)
        
    return out

def harmonic(sols, w=None, w_norm=False):
    if w is None:
        w = np.ones(sols.shape[1]) / sols.shape[1]
    
    if np.sum(w) != 1.0 and w_norm is False:
        raise ValueError('weights have to be normalised to 1')
    elif w_norm:
        w = w / np.sum(w, axis=0)
    
    # to implement    
    out = None
    
    return out


def bonferroni(sols, w=None, w_norm=False):
    if w is None:
        w = np.ones(sols.shape[1]) / sols.shape[1]
    
    if np.sum(w) != 1.0 and w_norm is False:
        raise ValueError('weights have to be normalised to 1')
    elif w_norm:
        w = w / np.sum(w, axis=0)
    
    # to implement    
    out = None
    
    return out


def power(sols, w=None, w_norm=False):
    if w is None:
        w = np.ones(sols.shape[1]) / sols.shape[1]
    
    if np.sum(w) != 1.0 and w_norm is False:
        raise ValueError('weights have to be normalised to 1')
    elif w_norm:
        w = w / np.sum(w, axis=0)
    
    # to implement    
    out = None
    
    return out

def choquet(sols, w=None, w_norm=False):
    if w is None:
        w = np.ones(sols.shape[1]) / sols.shape[1]
    
    if np.sum(w) != 1.0 and w_norm is False:
        raise ValueError('weights have to be normalised to 1')
    elif w_norm:
        w = w / np.sum(w, axis=0)
    
    # to implement    
    out = None
    
    return out

def sugeno(sols, w=None, w_norm=False):
    if w is None:
        w = np.ones(sols.shape[1]) / sols.shape[1]
    
    if np.sum(w) != 1.0 and w_norm is False:
        raise ValueError('weights have to be normalised to 1')
    elif w_norm:
        w = w / np.sum(w, axis=0)
    
    # to implement    
    out = None
    
    return out


def mix_linear(m1, m2, alpha):
    '''makes hierarchical aggregation'''
    return alpha*(m1) + (1.0 - alpha)*m2
