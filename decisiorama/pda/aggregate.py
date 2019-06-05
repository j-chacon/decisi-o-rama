# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:08:56 2019

@author: jchaconhurtado

Aggregation models

http://www.informatica.si/ojs-2.4.3/index.php/informatica/article/view/1321/972

"""
import numpy as np
from scipy.optimize import newton
OFFSET = 1e-6
#%%
@np.vectorize
def _rerange(u, offset=OFFSET):
    '''function rto re-range utilities so they are not either 0 or 1'''
    return u*(1.0 - 2.0*offset) + offset

def _dimcheck(sols, w):
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
#%%
def _w_normalize(w):
    if w.ndim == 1:
        w[:] = w / np.sum(w, axis=0)
    else:
        w[:] = np.array([wi / np.sum(wi) for wi in w])
    #%%
def additive(sols, w=None, w_norm=True):
    '''aggregate preferences using additive model. takes two vectors'''
    _dimcheck(sols, w)
    if w_norm:
        _w_normalize(w)
    
    if w.shape == sols.shape:
        out = np.sum(sols * w, axis=1)
    else:
        out = np.dot(sols, w)
        
    return out
#
#s = np.array([[0,1], [1,0], [0.5, 0.5]])
#w = np.array([0.8, 0.2])
#w2 = np.array([[0.8, 0.2], [0.8, 0.2], [0.8, 0.2]])
#print(additive(s,w))
#print(additive(s,w2))
#%%

# also known as the geometric mean operator
def cobb_douglas(sols, w, w_norm=True):
    '''aggregate preferences using cobb-douglas method. takes two vectors'''
    _dimcheck(sols, w)
    if w_norm:
        _w_normalize(w)
        
    if sols.ndim == 1:
        out = np.prod(sols**w)
        
    else:
        out = np.prod(sols**w, axis=1)

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

#%%
def reverse_harmonic(sols, w, w_norm=True):
    _dimcheck(sols, w)
    if w_norm:
        _w_normalize(w)
        
    if sols.ndim == 1:
        out = 1.0 - 1.0 / (np.sum(w / (1.0 - sols)))
    else:
        out = 1.0 - 1.0 / (np.sum(w / (1.0 - sols), axis=1))
    
    return out

s = np.array([[0,1], [1,0], [0.5, 0.5]])
w = np.array([0.8, 0.2])
w2 = np.array([[0.8, 0.2], [0.8, 0.2], [0.8, 0.2]])
print(reverse_harmonic(s,w))
print(reverse_harmonic(s,w2))

#%%
#https://www.rdocumentation.org/packages/utility/versions/1.4.3/topics/utility.aggregate.revaddpower
def reverse_power(sols, w, w_norm=True, alpha=1.0):
    _dimcheck(sols, w)
    if w_norm:
        _w_normalize(w)
        
    if sols.ndim == 1:
        out = 1.0 - np.power(np.sum(w*(1.0 - sols)**alpha), 1.0/alpha)
    else:
        out = 1.0 - np.power(np.sum(w*(1.0 - sols)**alpha, axis=1), 1.0/alpha)
    
    return out

s = np.array([[0,1], [1,0], [0.5, 0.5]])
w = np.array([0.8, 0.2])
w2 = np.array([[0.8, 0.2], [0.8, 0.2], [0.8, 0.2]])
print(reverse_power(s,w))
print(reverse_power(s,w2))
#%%
## https://www.rdocumentation.org/packages/utility/versions/1.4.3/topics/utility.aggregate.mult
#def multiplicative(sols, w, w_norm=True):
#    _dimcheck(sols, w)
#    if w_norm:
#        _w_normalize(w)
#        
#    
#    
#    if sols.ndim == 1:
#        # Get values of k using newton rhapson
#        _f = lambda k: (k+1) - np.prod(1 + k*w)
#        out = 1.0 - np.power(np.sum(w*(1.0 - sols)**alpha), 1.0/alpha)
#    else:
#        out = 1.0 - np.power(np.sum(w*(1.0 - sols)**alpha, axis=1), 1.0/alpha)
#    
#    return out
#
#s = np.array([[0,1], [1,0], [0.5, 0.5]])
#w = np.array([0.8, 0.2])
#w2 = np.array([[0.8, 0.2], [0.8, 0.2], [0.8, 0.2]])
#print(reverse_power(s,w))
#print(reverse_power(s,w2))

#%%
# https://www.rdocumentation.org/packages/utility/versions/1.4.3/topics/utility.aggregate.addsplitpower

def split_power(sols, w, pars=[0.5, 0.5], w_norm=True):
    _dimcheck(sols, w)
    if w_norm:
        _w_normalize(w)
    
    alpha = pars[0]
    s = pars[1]
    
    @np.vectorize
    def _g(u, s, alpha):
        if u <= s:
            out = s*(u/s)**alpha
        else:
            out = 1.0 - (1.0 - s)*((1.0 - u)/(1.0 - s))**alpha
        return out
    
    @np.vectorize
    def _g_inv(u, s, alpha):
        if u <= s:
            out = s*(u/s)**(1.0/alpha)
        else:
            out = 1.0 - (1.0 - s)*((1.0 - u)/(1.0 - s))**(1.0/alpha)
        return out
    
    if sols.ndim == 1:
        out = _g_inv(np.sum(w*_g(sols, s, alpha)), s, alpha)
    else:
        out = _g_inv(np.sum(w*_g(sols, s, alpha), axis=1), s, alpha)
    
    return out

s = np.array([[0,1], [1,0], [0.5, 0.5]])
w = np.array([0.8, 0.2])
w2 = np.array([[0.8, 0.2], [0.8, 0.2], [0.8, 0.2]])
print(split_power(s,w))
print(split_power(s,w2))

#%%

def harmonic(sols, w=None, w_norm=True):
    _dimcheck(sols, w)
    if w_norm:
        _w_normalize(w)
    
    sols = _rerange(sols, OFFSET)
    
    if sols.ndim == 1:
        out = 1.0 / np.sum(w/sols)
    else:
        out = 1.0 / np.sum(w/sols, axis=1)
    
    return out

s = np.array([[0,1], [1,0], [0.5, 0.5]])
w = np.array([0.8, 0.2])
w2 = np.array([[0.8, 0.2], [0.8, 0.2], [0.8, 0.2]])
print(harmonic(s,w))
print(harmonic(s,w2))

#%%

def bonferroni(sols, w=None, w_norm=False):
    raise NotImplementedError('Not implemented yet')

def power(sols, w=None, w_norm=False):
    raise NotImplementedError('Not implemented yet')

def choquet(sols, w=None, w_norm=False):
    raise NotImplementedError('Not implemented yet')

def sugeno(sols, w=None, w_norm=False):
    raise NotImplementedError('Not implemented yet')
