# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:08:56 2019

@author: jchaconhurtado

Aggregation models

http://www.informatica.si/ojs-2.4.3/index.php/informatica/article/view/1321/972

"""
import numpy as np

#%%
def additive(sols, w=None, w_norm=True):
    '''aggregate preferences using additive model. takes two vectors'''
    if not sols.ndim == 2:
#        if not sols.ndim == 1:
        msg = ('The dimensions of sols have to be (1, ) or (2, ) '
               'got {0}'.format(sols.ndim))
        raise ValueError(msg)
    
    if w is None:
        w = np.ones(sols.shape[1]) / sols.shape[1]
        
    elif callable(w[0]):  # check if its an iterable with a generator
        w = np.array([wi() for wi in w])
        w = w / np.sum(w, axis=0)
    
    if w.ndim == 1:
        if sols.shape[1] != w.shape[0]:
            msg = ('Weights and solutions do not match. the shape of '
                   'solutions are (n, {0}) and {1}, and the indices should '
                   'match'.format(sols.shape[1], w.shape)
                   )
            raise ValueError(msg)
    
    elif w.ndim == 2:
        if sols.shape != w.shape:
            msg = ('Weights and solutions do not match. the shape of '
                   'solutions are {0} and {1}, and the indices should '
                   'match'.format(sols.shape, w.shape)
                   )
            raise ValueError(msg)
        
        
    if np.sum(w) != 1.0 and w_norm is False:
        msg = 'weights have to be normalised to 1, or w_norm set to True'
        raise ValueError(msg)
    elif w_norm:
        if w.ndim == 1:
            w = w / np.sum(w, axis=0)
        else:
            w = np.array([wi / np.sum(wi) for wi in w])
    
    if w.shape == sols.shape:
        out = np.sum(sols * w, axis=1)
    else:
        out = np.dot(sols, w)
        
    return out

#s = np.array([[0,1], [1,0], [0.5, 0.5]])
#w = np.array([0.8, 0.2])
#w2 = np.array([[0.8, 0.2], [0.8, 0.2], [0.8, 0.2]])
#print(additive(s,w))
#print(additive(s,w2))
#%%

# also known as the geometric mean operator
def cobb_douglas(sols, w, w_norm=True):
    '''aggregate preferences using cobb-douglas method. takes two vectors'''
    if not sols.ndim == 2:
#        if not sols.ndim == 1:
        msg = ('The dimensions of sols have to be (1, ) or (2, ) '
               'got {0}'.format(sols.ndim))
        raise ValueError(msg)
    
    if w is None:
        w = np.ones(sols.shape[1]) / sols.shape[1]
    elif callable(w[0]):  # check if its an iterable with a generator
        w = np.array([wi() for wi in w])
        w = w / np.sum(w, axis=0)
    
    if w.ndim == 1:
        if sols.shape[1] != w.shape[0]:
            msg = ('Weights and solutions do not match. the shape of '
                   'solutions are (n, {0}) and {1}, and the indices should '
                   'match'.format(sols.shape[1], w.shape)
                   )
            raise ValueError(msg)
    
    elif w.ndim == 2:
        if sols.shape != w.shape:
            msg = ('Weights and solutions do not match. the shape of '
                   'solutions are {0} and {1}, and the indices should '
                   'match'.format(sols.shape, w.shape)
                   )
            raise ValueError(msg)
    
    if np.sum(w) != 1.0 and w_norm is False:
        raise ValueError('weights have to be normalised to 1')
    elif w_norm:
        if w.ndim == 1:
            w = w / np.sum(w, axis=0)
        else:
            w = np.array([wi / np.sum(wi) for wi in w])
        
    out = np.zeros([sols.shape[0]])
#    print(sols.ndim == 1)
    if sols.ndim == 1:
        out = np.prod(sols**w)
        
    else:# sols.ndim == 2:
#        print(np.min(sols))
        out = np.prod(sols**w, axis=1)
#        for i in range(sols.shape[0]):
#            out[i] = np.prod(sols[i,:]**w)
#    else:
#        raise ValueError('Dimension of the sols vector is not supported')
    return out

#s = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [0.0, -0.1]])
#w = np.array([0.8, 0.2])
#s = np.array([[0,1], [1,0], [0.5, 0.5]])
#w = np.array([0.8, 0.2])
#w2 = np.array([[0.8, 0.2], [0.8, 0.2], [0.8, 0.2]])
#print(cobb_douglas(s, w))
#print(cobb_douglas(s, w2))
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
    raise NotImplementedError('Not implemented yet')

def bonferroni(sols, w=None, w_norm=False):
    raise NotImplementedError('Not implemented yet')

def power(sols, w=None, w_norm=False):
    raise NotImplementedError('Not implemented yet')

def choquet(sols, w=None, w_norm=False):
    raise NotImplementedError('Not implemented yet')

def sugeno(sols, w=None, w_norm=False):
    raise NotImplementedError('Not implemented yet')
