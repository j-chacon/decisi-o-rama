# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:08:56 2019

@author: jchaconhurtado

Aggregation models

http://www.informatica.si/ojs-2.4.3/index.php/informatica/article/view/1321/972

"""
import numpy as np
#from random_instance import check_numeric


#%%
def additive(sols, w=None, w_norm=True):
    '''aggregate preferences using additive model. takes two vectors'''
    if w is None:
        w = np.ones(sols.shape[1]) / sols.shape[1]
    elif callable(w[0]):  # check if its an iterable with a generator
        w = np.array([wi() for wi in w])
        w = w / np.sum(w, axis=0)
    
    if np.sum(w) != 1.0 and w_norm is False:
        raise ValueError('weights have to be normalised to 1')
    elif w_norm:
        w = w / np.sum(w, axis=0)
    
    return np.dot(sols, w)

#s = np.array([[0,1], [1,0], [0.5, 0.5]])
#w = np.array([0.8, 0.2])
#
#additive(s,w)
#%%

# also known as the geometric mean operator
def cobb_douglas(sols, w, w_norm=True):
    '''aggregate preferences using cobb-douglas method. takes two vectors'''
    if w is None:
        w = np.ones(sols.shape[1]) / sols.shape[1]
    elif callable(w[0]):  # check if its an iterable with a generator
        w = np.array([wi() for wi in w])
        w = w / np.sum(w, axis=0)
    
    if np.sum(w) != 1.0 and w_norm is False:
        raise ValueError('weights have to be normalised to 1')
    elif w_norm:
        w = w / np.sum(w, axis=0)
    
    out = np.zeros([sols.shape[0]])
#    print(sols.ndim == 1)
    if sols.ndim == 1:
        out = np.prod(sols**w)
        
    elif sols.ndim == 2:
        for i in range(sols.shape[0]):
            out[i] = np.prod(sols[i,:]**w)
    else:
        raise ValueError('Dimension of the sols vector is not supported')
    return out

#s = np.array([[0,1], [1,0], [0.5, 0.5]])
#w = np.array([0.8, 0.2])
#
#cobb_douglas(s,w)
#%%
def mix_linear_cobb(sols, w, pars=[0.5,], w_norm=True):
    
    if callable(pars[0]):
        alpha = pars[0]()
    else:
        alpha = pars[0]
    add_model = additive(sols, w, w_norm)
    cd_model = cobb_douglas(sols, w, w_norm)
    return alpha*(add_model) + (1.0 - alpha)*cd_model


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


#def mix_linear(m1, m2, alpha):
#    '''makes hierarchical aggregation'''
#    return alpha*(m1) + (1.0 - alpha)*m2
