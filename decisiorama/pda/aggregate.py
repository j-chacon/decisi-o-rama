# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:08:56 2019

@author: jchaconhurtado

Aggregation models

http://www.informatica.si/ojs-2.4.3/index.php/informatica/article/view/1321/972

"""
import numpy as np
#from scipy.optimize import newton
OFFSET = 1e-6

@np.vectorize
def _rerange(utils, offset=OFFSET):
    '''Re-range utilities so they are in the open interval (0,1)'''
    return utils*(1.0 - 2.0*offset) + offset

def _dimcheck(utils, w):
    '''Check the dimension consistency of inputs and weights'''
    if not utils.ndim == 2:
#        if not utils.ndim == 1:
        msg = ('The dimensions of utils have to be (1, ) or (2, ) '
               'got {0}'.format(utils.ndim))
        raise ValueError(msg)
    
    if w is None:
        w = np.ones(utils.shape[1]) / utils.shape[1]
        
    elif callable(w[0]):  # check if its an iterable with a generator
        w = np.array([wi() for wi in w])
        w = w / np.sum(w, axis=0)
    
    if w.ndim == 1:
        if utils.shape[1] != w.shape[0]:
            msg = ('Weights and solutions do not match. The shape of '
                   'solutions is {0} and of weights is {1}'.format(utils.shape[1], 
                                 w.shape)
                   )
            raise ValueError(msg)
    
    elif w.ndim == 2:
        if utils.shape != w.shape:
            msg = ('Weights and solutions do not match. The shape of '
                   'solutions is {0} and of weights is {1}'.format(utils.shape, 
                                 w.shape)
                   )
            raise ValueError(msg)

def _w_normalize(w):
    '''Normalise the weights so the um is equal to 1'''
    if w.ndim == 1:
        w[:] = w / np.sum(w, axis=0)
    else:
        w[:] = np.array([wi / np.sum(wi) for wi in w])
        
    
#%%
def additive(utils, w, w_norm=True, *args, **kwargs):
    '''Additive utility aggregation function
    
    Aggregate preferences using a weighted average
    
    Parameters
    ----------
    utils : ndarray [n, u]
        Two-dimensional array with the provided utilities to aggregate. The 
        dimensions corresponds to the number of random samples (n) and the 
        number of utilities (u)
    w : ndarray [u], [n, u]
        Array with the provided weights to each of the utilities. If passed 
        as a 1D-array, the same weights are used for of all the random samples.
        In case it is a 2D-array, w requires the same dimensions as `utils`
    w_norm : Bool, optional
        If True, the sum of the weights will be equal to 1
    
    Returns
    -------
    out : ndarray [n]
        Vector with the aggregated values
    
    Example
    -------
    .. highlight:: python
    .. code-block:: python

        s = np.array([0.0, 1.0])
        w = np.array([0.8, 0.2])
        print(additive(s,w))
        
        >>> [0.2]
        
        s = np.array([[0.0, 1.0], 
                      [1.0, 0.0], 
                      [0.5, 0.5]])
        w = np.array([0.8, 0.2])
        print(additive(s,w))
        
        >>> [0.2 0.8 0.5]
        
        s = np.array([[0.0, 1.0], 
                      [1.0, 0.0], 
                      [0.5, 0.5]])
        w = np.array([[0.8, 0.2], 
                      [0.8, 0.2], 
                      [0.8, 0.2]])
        print(additive(s,w))
        
        >>> [0.2 0.8 0.5]
    '''
    if utils.ndim == 1:
        utils = np.reshape(utils, [1,-1])
        
    _dimcheck(utils, w)
    if w_norm:
        _w_normalize(w)
    
    if w.shape == utils.shape:
        out = np.sum(utils * w, axis=1)
    else:
        out = np.dot(utils, w)
        
    return out

#s = np.array([0.0, 1.0])
#w = np.array([0.8, 0.2])
#print(additive(s,w))
#
#s = np.array([[0.0, 1.0], 
#              [1.0, 0.0], 
#              [0.5, 0.5]])
#w = np.array([0.8, 0.2])
#print(additive(s,w))
#
#s = np.array([[0.0, 1.0], 
#              [1.0, 0.0], 
#              [0.5, 0.5]])
#w = np.array([[0.8, 0.2], 
#              [0.8, 0.2], 
#              [0.8, 0.2]])
#print(additive(s,w))
    
#%%

# also known as the geometric mean operator
def cobb_douglas(utils, w, w_norm=True, *args, **kwargs):
    '''Cobb-Douglas utility aggregation function
    
    Aggregate preferences using the cobb-douglas aggregation function. This 
    method is also known as the weighted geometric average
    
    Parameters
    ----------
    utils : ndarray [n, u]
        Two-dimensional array with the provided utilities to aggregate. The 
        dimensions corresponds to the number of random samples (n) and the 
        number of utilities (u)
    w : ndarray [u], [n, u]
        Array with the provided weights to each of the utilities. If passed 
        as a 1D-array, the same weights are used for of all the random samples.
        In case it is a 2D-array, w requires the same dimensions as `utils`
    w_norm : Bool, optional
        If True, the sum of the weights will be equal to 1
    
    Returns
    -------
    out : ndarray [n]
        Vector with the aggregated values
    
    Example
    -------
    .. highlight:: python
    .. code-block:: python

        s = np.array([0.0, 1.0])
        w = np.array([0.8, 0.2])
        print(cobb_douglas(s,w))
        
        >>> [0.]
        
        s = np.array([[0.0, 1.0], 
                      [1.0, 0.0], 
                      [0.5, 0.5]])
        w = np.array([0.8, 0.2])
        print(cobb_douglas(s,w))
        
        >>> [0. 0. 0.5]
        
        s = np.array([[0.0, 1.0], 
                      [1.0, 0.0], 
                      [0.5, 0.5]])
        w = np.array([[0.8, 0.2], 
                      [0.8, 0.2], 
                      [0.8, 0.2]])
        print(cobb_douglas(s,w))
        
        >>> [0. 0. 0.5]

    '''
    if utils.ndim == 1:
        utils = np.reshape(utils, [1,-1])
        
    _dimcheck(utils, w)
    if w_norm:
        _w_normalize(w)

    return np.prod(utils**w, axis=1)


#s = np.array([0.0, 1.0])
#w = np.array([0.8, 0.2])
#print(cobb_douglas(s,w))
#
#s = np.array([[0.0, 1.0], 
#              [1.0, 0.0], 
#              [0.5, 0.5]])
#w = np.array([0.8, 0.2])
#print(cobb_douglas(s,w))
#
#s = np.array([[0.0, 1.0], 
#              [1.0, 0.0], 
#              [0.5, 0.5]])
#w = np.array([[0.8, 0.2], 
#              [0.8, 0.2], 
#              [0.8, 0.2]])
#print(cobb_douglas(s,w))


#%%
def mix_linear_cobb(utils, w, pars=[0.5,], w_norm=True, *args, **kwargs):
    
    #print('The mix_linear_cobb function has been deprecated. rather use mix')
    if callable(pars[0]):
        alpha = pars[0]()
    else:
        alpha = pars[0]
    add_model = additive(utils, w, w_norm)
    cd_model = cobb_douglas(utils, w, w_norm)
    return alpha*(add_model) + (1.0 - alpha)*cd_model

#%%
def reverse_harmonic(utils, w, w_norm=True, *args, **kwargs):
    '''Reverse harmonic utility aggregation function
    
    Aggregate preferences using the cobb-douglas aggregation function. This 
    method is also known as the weighted geometric average
    
    Parameters
    ----------
    utils : ndarray [n, u]
        Two-dimensional array with the provided utilities to aggregate. The 
        dimensions corresponds to the number of random samples (n) and the 
        number of utilities (u)
    w : ndarray [u], [n, u]
        Array with the provided weights to each of the utilities. If passed 
        as a 1D-array, the same weights are used for of all the random samples.
        In case it is a 2D-array, w requires the same dimensions as `utils`
    w_norm : Bool, optional
        If True, the sum of the weights will be equal to 1
    
    Returns
    -------
    out : ndarray [n]
        Vector with the aggregated values
    
    Example
    -------
    .. highlight:: python
    .. code-block:: python

        s = np.array([0.0, 1.0])
        w = np.array([0.8, 0.2])
        print(reverse_harmonic(s,w))
        
        >>> [1.]
        
        s = np.array([[0.0, 1.0], 
                      [1.0, 0.0], 
                      [0.5, 0.5]])
        w = np.array([0.8, 0.2])
        print(reverse_harmonic(s,w))
        
        >>> [1. 1. 0.5]
        
        s = np.array([[0.0, 1.0], 
                      [1.0, 0.0], 
                      [0.5, 0.5]])
        w = np.array([[0.8, 0.2], 
                      [0.8, 0.2], 
                      [0.8, 0.2]])
        print(reverse_harmonic(s,w))
        
        >>> [1. 1. 0.5]
    '''
    if utils.ndim == 1:
        utils = np.reshape(utils, [1,-1])
        
    _dimcheck(utils, w)
    if w_norm:
        _w_normalize(w)
    
    return 1.0 - 1.0 / (np.sum(w / (1.0 - utils), axis=1))

s = np.array([0.0, 1.0])
w = np.array([0.8, 0.2])
print(reverse_harmonic(s,w))


s = np.array([[0.0, 1.0], 
              [1.0, 0.0], 
              [0.5, 0.5]])
w = np.array([0.8, 0.2])
print(reverse_harmonic(s,w))

s = np.array([[0.0, 1.0], 
              [1.0, 0.0], 
              [0.5, 0.5]])
w = np.array([[0.8, 0.2], 
              [0.8, 0.2], 
              [0.8, 0.2]])
print(reverse_harmonic(s,w))


#%%
#https://www.rdocumentation.org/packages/utility/versions/1.4.3/topics/utility.aggregate.revaddpower
def reverse_power(utils, w, alpha, w_norm=True, *args, **kwargs):
    '''Reverse power utility aggregation function
    
    Aggregate preferences using the reverse power aggregation function. 
    
    Parameters
    ----------
    utils : ndarray [n, u]
        Two-dimensional array with the provided utilities to aggregate. The 
        dimensions corresponds to the number of random samples (n) and the 
        number of utilities (u)
    w : ndarray [u], [n, u]
        Array with the provided weights to each of the utilities. If passed 
        as a 1D-array, the same weights are used for of all the random samples.
        In case it is a 2D-array, w requires the same dimensions as `utils`
    w_norm : Bool, optional, default True
        If True, the sum of the weights will be equal to 1
    alpha : float, ndarray [n], default 1.0
        power coefficient. If passed as a float, the values will remain the 
        same over the whole computation. Otherwise, it is possible to pass a 
        vector with a value for each random sample
    
    Returns
    -------
    out : ndarray [n]
        Vector with the aggregated values
    
    Example
    -------
    .. highlight:: python
    .. code-block:: python

        s = np.array([0.0, 1.0])
        w = np.array([0.8, 0.2])
        alpha = 1.0
        print(reverse_power(s, w, alpha))
        
        >>> [1.]
        
        s = np.array([[0.0, 1.0], 
                      [1.0, 0.0], 
                      [0.5, 0.5]])
        w = np.array([0.8, 0.2])
        alpha = np.array([1.0],
                         [1.0],
                         [1.0])
        print(reverse_power(s, w, alpha))
        
        >>> [1. 1. 0.5]
        
        s = np.array([[0.0, 1.0], 
                      [1.0, 0.0], 
                      [0.5, 0.5]])
        w = np.array([[0.8, 0.2], 
                      [0.8, 0.2], 
                      [0.8, 0.2]])
        alpha = np.array([1.0],
                         [1.0],
                         [1.0])
        print(reverse_power(s, w, alpha))
        
        >>> [1. 1. 0.5]
    '''
    if utils.ndim == 1:
        utils = np.reshape(utils, [1,-1])
        
    _dimcheck(utils, w)
    if w_norm:
        _w_normalize(w)
    
    if type(alpha) is np.ndarray:
        if alpha.ndim == 1:
            alpha = np.tile(alpha, (utils.shape[1], 1)).T
        
    return 1.0 - np.power(np.sum(w*(1.0 - utils)**alpha, axis=1), 1.0/alpha)

s = np.array([0.0, 1.0])
w = np.array([0.8, 0.2])
alpha = 1.0
print(reverse_power(s, w, alpha))

#>>> [1.]

s = np.array([[0.0, 1.0], 
              [1.0, 0.0], 
              [0.5, 0.5]])
w = np.array([0.8, 0.2])
alpha = np.array([1.0, 1.0, 1.0])
print(reverse_power(s, w, alpha))

#>>> [1. 1. 0.5]

s = np.array([[0.0, 1.0], 
              [1.0, 0.0], 
              [0.5, 0.5]])
w = np.array([[0.8, 0.2], 
              [0.8, 0.2], 
              [0.8, 0.2]])
alpha = np.array([1.0],
                 [1.0],
                 [1.0])
print(reverse_power(s, w, alpha))

#
#s = np.array([[0,1], [1,0], [0.5, 0.5]])
#w = np.array([0.8, 0.2])
#w2 = np.array([[0.8, 0.2], [0.8, 0.2], [0.8, 0.2]])
#s2 = np.array([0,1])
#print(reverse_power(s,w))
#print(reverse_power(s,w2))
#print(reverse_power(s2,w))
#%%
## https://www.rdocumentation.org/packages/utility/versions/1.4.3/topics/utility.aggregate.mult
def multiplicative(utils, w, w_norm=True, *args, **kwargs):
    raise NotImplementedError('This method has not been implemented yet')
#    _dimcheck(utils, w)
#    if w_norm:
#        _w_normalize(w)
#        
#    if utils.ndim == 1:
#        utils = np.reshape(utils, [1,-1])
#        
#    # Get values of k using newton rhapson
#    _f = lambda k: (k+1) - np.prod(1 + k*w)
#
#    
#    return out
#
#s = np.array([[0,1], [1,0], [0.5, 0.5]])
#w = np.array([0.8, 0.2])
#w2 = np.array([[0.8, 0.2], [0.8, 0.2], [0.8, 0.2]])
#print(multiplicative(s,w))
#print(multiplicative(s,w2))

#%%
# https://www.rdocumentation.org/packages/utility/versions/1.4.3/topics/utility.aggregate.addsplitpower

def split_power(utils, w, alpha, s, w_norm=True, *args, **kwargs):
    '''Split power utility aggregation function
    
    Aggregate preferences using the split power aggregation function. 
    
    Parameters
    ----------
    utils : ndarray [n, u]
        Two-dimensional array with the provided utilities to aggregate. The 
        dimensions corresponds to the number of random samples (n) and the 
        number of utilities (u)
    w : ndarray [u], [n, u]
        Array with the provided weights to each of the utilities. If passed 
        as a 1D-array, the same weights are used for of all the random samples.
        In case it is a 2D-array, w requires the same dimensions as `utils`
    alpha : float, ndarray[n]
        Alpha parameter of the power function. In case a float value is used,
        it will be constant for all of the random samples
    s : float, ndarray[n]
        s parameter of the power function. In case a float value is used,
        it will be constant for all of the random samples
    w_norm : Bool, optional
        If True, the sum of the weights will be equal to 1
    
    Returns
    -------
    out : ndarray [n]
        Vector with the aggregated values
    
    Example
    -------
    s = np.array([[0.0, 1.0], 
                  [1.0, 0.0], 
                  [0.5, 0.5]])
    w = np.array([0.8, 0.2])
    additive(s,w)
    
    s = np.array([[0.0, 1.0], 
                  [1.0, 0.0], 
                  [0.5, 0.5]])
    w = np.array([[0.8, 0.2], 
                  [0.8, 0.2], 
                  [0.8, 0.2]])
    additive(s,w)
    '''
    if utils.ndim == 1:
        utils = np.reshape(utils, [1,-1])
        
    _dimcheck(utils, w)
    if w_norm:
        _w_normalize(w)
    
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

    return _g_inv(np.sum(w*_g(utils, s, alpha), axis=1), s, alpha)

s = np.array([[0,1], [1,0], [0.5, 0.5]])
w = np.array([0.8, 0.2])
w2 = np.array([[0.8, 0.2], [0.8, 0.2], [0.8, 0.2]])
s2 = np.array([0,1])
print(split_power(s,w, 1, 1))
print(split_power(s,w2, 1, 1))
print(split_power(s2,w, 1, 1))
#%%

def harmonic(utils, w, w_norm=True, *args, **kwargs):
    '''Harmonic utility aggregation function
    
    Aggregate preferences using the reverse power aggregation function. 
    
    Parameters
    ----------
    utils : ndarray [n, u]
        Two-dimensional array with the provided utilities to aggregate. The 
        dimensions corresponds to the number of random samples (n) and the 
        number of utilities (u)
    w : ndarray [u], [n, u]
        Array with the provided weights to each of the utilities. If passed 
        as a 1D-array, the same weights are used for of all the random samples.
        In case it is a 2D-array, w requires the same dimensions as `utils`
    w_norm : Bool, optional
        If True, the sum of the weights will be equal to 1
    
    Returns
    -------
    out : ndarray [n]
        Vector with the aggregated values
    
    Example
    -------
    s = np.array([[0.0, 1.0], 
                  [1.0, 0.0], 
                  [0.5, 0.5]])
    w = np.array([0.8, 0.2])
    additive(s,w)
    
    s = np.array([[0.0, 1.0], 
                  [1.0, 0.0], 
                  [0.5, 0.5]])
    w = np.array([[0.8, 0.2], 
                  [0.8, 0.2], 
                  [0.8, 0.2]])
    additive(s,w)
    '''
    if utils.ndim == 1:
        utils = np.reshape(utils, [1,-1])
        
    _dimcheck(utils, w)
    if w_norm:
        _w_normalize(w)
    
    utils = _rerange(utils, OFFSET)
    return 1.0 / np.sum(w/utils, axis=1)

s = np.array([[0,1], [1,0], [0.5, 0.5]])
w = np.array([0.8, 0.2])
w2 = np.array([[0.8, 0.2], [0.8, 0.2], [0.8, 0.2]])
s2 = np.array([0,1])
print(harmonic(s,w))
print(harmonic(s,w2))
print(harmonic(s2,w))

#%%

def maximum(utils, *args, **kwargs):
    '''Maximum utility aggregation function
    
    Aggregate preferences using the maximum aggregation function. 
    
    Parameters
    ----------
    utils : ndarray [n, u]
        Two-dimensional array with the provided utilities to aggregate. The 
        dimensions corresponds to the number of random samples (n) and the 
        number of utilities (u)
    
    Returns
    -------
    out : ndarray [n]
        Vector with the aggregated values
    
    Example
    -------
    s = np.array([[0.0, 1.0], 
                  [1.0, 0.0], 
                  [0.5, 0.5]])
    w = np.array([0.8, 0.2])
    additive(s,w)
    
    s = np.array([[0.0, 1.0], 
                  [1.0, 0.0], 
                  [0.5, 0.5]])
    w = np.array([[0.8, 0.2], 
                  [0.8, 0.2], 
                  [0.8, 0.2]])
    additive(s,w)
    '''
    if utils.ndim == 1:
        utils = np.reshape(utils, [1,-1])
        
    return np.max(utils, axis=1)
    
#s = np.array([[0,1], [1,0], [0.5, 0.5]])
#w = np.array([0.8, 0.2])
#w2 = np.array([[0.8, 0.2], [0.8, 0.2], [0.8, 0.2]])
#print(maximum(s,w))
#print(maximum(s,w2))


#%%
def minimum(utils, *args, **kwargs):
    '''Minimum utility aggregation function
    
    Aggregate preferences using the minimum aggregation function. 
    
    Parameters
    ----------
    utils : ndarray [n, u]
        Two-dimensional array with the provided utilities to aggregate. The 
        dimensions corresponds to the number of random samples (n) and the 
        number of utilities (u)
    w : ndarray [u], [n, u]
        Array with the provided weights to each of the utilities. If passed 
        as a 1D-array, the same weights are used for of all the random samples.
        In case it is a 2D-array, w requires the same dimensions as `utils`
    w_norm : Bool, optional
        If True, the sum of the weights will be equal to 1
    
    Returns
    -------
    out : ndarray [n]
        Vector with the aggregated values
    
    Example
    -------
    s = np.array([[0.0, 1.0], 
                  [1.0, 0.0], 
                  [0.5, 0.5]])
    w = np.array([0.8, 0.2])
    additive(s,w)
    
    s = np.array([[0.0, 1.0], 
                  [1.0, 0.0], 
                  [0.5, 0.5]])
    w = np.array([[0.8, 0.2], 
                  [0.8, 0.2], 
                  [0.8, 0.2]])
    additive(s,w)
    '''
    if utils.ndim == 1:
        utils = np.reshape(utils, [1,-1])
        
    return np.min(utils, axis=1)
    
s = np.array([[0,1], [1,0], [0.5, 0.5]])
w = np.array([0.8, 0.2])
s = np.array([[0,1]])
#w2 = np.array([[0.8, 0.2], [0.8, 0.2], [0.8, 0.2]])
print(minimum(s,w=10))
#print(minimum(s,w2))


#%% 
def mix(utils, w, methods, w_methods, mix_fun, w_norm=True):
    '''mixed utility aggregation function
    
    Aggregate preferences using a mix of aggregation functions. 
    
    Parameters
    ----------
    utils : ndarray [n, u]
        Two-dimensional array with the provided utilities to aggregate. The 
        dimensions corresponds to the number of random samples (n) and the 
        number of utilities (u)
    w : ndarray [u], [n, u]
        Array with the provided weights to each of the utilities. If passed 
        as a 1D-array, the same weights are used for of all the random samples.
        In case it is a 2D-array, w requires the same dimensions as `utils`
    methods : list [m]
        a list of functions that will create each individual member of the 
        model mixture
    w_methods : ndarray [m], [n, m]
        An array for the weights that will be used to mix each of the methods
    mix_fun : function
        Function that will be used to aggregate each of the members of the 
        methods
    w_norm : Bool, optional
        If True, the sum of the weights will be equal to 1
    
    Returns
    -------
    out : ndarray [n]
        Vector with the aggregated values
    
    Example
    -------
    s = np.array([[0.0, 1.0], 
                  [1.0, 0.0], 
                  [0.5, 0.5]])
    w = np.array([0.8, 0.2])
    additive(s,w)
    
    s = np.array([[0.0, 1.0], 
                  [1.0, 0.0], 
                  [0.5, 0.5]])
    w = np.array([[0.8, 0.2], 
                  [0.8, 0.2], 
                  [0.8, 0.2]])
    additive(s,w)
    '''
    if utils.ndim == 1:
        utils = np.reshape(utils, [1,-1])
        
    _dimcheck(utils, w)
    if len(methods) != len(w_methods):
        _msg = ('length of methods ({0}) and w_methods ({1}) are not '
                'the same'.format(len(methods), len(w_methods))
                )
        raise ValueError(_msg)
    
    if w_norm:
        _w_normalize(w)
        _w_normalize(w_methods)
    
    agg_util = np.array([m(utils, w) for m in methods]).T
    return mix_fun(agg_util, w_methods)

#s = np.array([[0,1], [1,0], [0.5, 0.5]])
#w = np.array([0.8, 0.2])
#w2 = np.array([[0.8, 0.2], [0.8, 0.2], [0.8, 0.2]])
#methods = [maximum, minimum]
#w_methods = np.array([0.5, 0.5])
#mix_fun = additive
#print(mix(s, w, methods, w_methods, mix_fun))
##print(linear_mix(s, w2))    
#
#

#%%

def bonferroni(utils, w=None, w_norm=False):
    raise NotImplementedError('Not implemented yet')

def power(utils, w=None, w_norm=False):
    raise NotImplementedError('Not implemented yet')

def choquet(utils, w=None, w_norm=False):
    raise NotImplementedError('Not implemented yet')

def sugeno(utils, w=None, w_norm=False):
    raise NotImplementedError('Not implemented yet')
