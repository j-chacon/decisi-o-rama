# -*- coding: utf-8 -*-
""" Utility Module

This module contains utility functions. Current implementation only has the
exponential utility function.

"""

__author__ = "Juan Carlos Chacon-Hurtado"
__credits__ = ["Juan Carlos Chacon-Hurtado", "Lisa Scholten"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "Juan Carlos Chacon-Hurtado"
__email__ = "j.chaconhurtado@tudelft.nl"
__status__ = "Development"
__last_update__ = "22-08-2019"

import numpy as np
from numba import vectorize

@vectorize('float64(float64, float64)', target='cpu')
def _exponential(v, r):
    '''vectorised engine of the exponential utility'''
    if r == 0:
        out = v
    else:
        out = (1.0 - np.exp(-r*v)) / (1.0 - np.exp(-r))
    return out

def exponential(v, r, *args, **kwargs):
    '''Calculates the exponential utility function

    Parameters
    ----------
    v : float, ndarray
        Array containing the normalised values
    r : float, ndarray
        Exponent parameter

    returns
    out : ndarray
        Utility values

    Note
    ----
    This is executed as a vectorized function

    '''
    if np.max(v) > 1.0 or np.min(v) < 0.0:
        _msj = ('Values passed to the utility function should be ranked '
                'normalised between 0 and 1')
        raise ValueError(_msj)
    return _exponential(v, r)

@vectorize('float64(float64, float64)', target='cpu')
def _power(v, p):
    '''vectorised engine of the exponential utility'''
    return v**p

def power(v, p, *args, **kwargs):
    '''Calculates the power utility function

    Parameters
    ----------
    v : float, ndarray
        Array containing the normalised values
    p : float, ndarray
        power parameter

    returns
    out : ndarray
        Utility values

    Note
    ----
    This is executed as a vectorized function

    '''
    if np.max(v) > 1.0 or np.min(v) < 0.0:
        _msj = ('Values passed to the utility function should be ranked '
                'normalised between 0 and 1')
        raise ValueError(_msj)
    
    if np.min(p) <= 0.0:
        _msj = ('p (power coefficient) value has to be positive')
        raise ValueError(_msj)
    return _power(v,p)

@vectorize('float64(float64, float64, float64, float64, float64)', 
           target='cpu', nopython=True)
def _cpt_pow(v, s_u, s_v, a_win, a_loss):
    '''vectorised engine of the CPT power utility'''
    if v > s_v:  # winning side
        _v = (v - s_v)/(1.0 - s_v)
        u = _v**a_win
        u = s_u + u*(1.0 - s_u) 

    elif v < s_v:  # losing side
        _v = v/s_v
        u = _v**a_loss
        u = u * s_u
    else:
        u = s_u
    return u

def cpt_pow(v, s_u, s_v, r, r_loss=None, gl_ratio=None, *args, **kwargs):
    '''Calculates the CPT power utility function

    Parameters
    ----------
    v : float, ndarray
        Array containing the normalised attribute values
    s_u : float, ndarray
        Array containing the utility for the reference point
    s_v : float, ndarray
        Array containing the attribute value for the reference point
    r : float, ndarray
        Exponent parameter in the gains side
    r_loss : float, ndarray (optional)
        Exponent parameter in the Loss side. By default is the inverse of the
        gains parameter
    gl_ratio : float, ndarray (optional)
        Defines the exponent parameter loss in terms of the gains-loss ratio. 
        This is optinal and does not override the defition of r_loss
        
    returns
    out : ndarray
        Utility values

    Note
    ----
    This is executed as a vectorized function

    '''
    if np.max(v) > 1 or np.min(v) < 0:
        raise ValueError('v has to be in the range [0,1]')
    
    if np.min(s_u) < 0.0 or np.max(s_u) > 1.0:
        _msj = 's_u values have to be in the range [0,1]'
        raise ValueError(_msj)
    
    if np.min(s_v) < 0.0 or np.max(s_v) > 1.0:
        _msj = 's_v values have to be in the range [0,1]'
        raise ValueError(_msj)
    
    if np.min(a) < 0.0:
        _msj = ('alp (Alpha) value has to be positive')
        raise ValueError(_msj)
    
    if gl_ratio is not None and np.min(gl_ratio) <= 0.0:
        _msj = ('wr_ratio has to be positive')
        raise ValueError(_msj)
    
    if r_loss is None and gl_ratio is None:
        r_loss = 1.0/a
    
    elif gl_ratio is not None and r_loss is None:
        r_loss = gl_ratio/a
        
    if np.min(r_loss) < 0.0:
        _msj = ('bet (Beta) value has to be positive')
        raise ValueError(_msj)
        
    return _cpt_pow(v, s_u, s_v, a, r_loss)    


@vectorize('float64(float64, float64, float64, float64, float64)', 
           target='cpu', nopython=True)
def _cpt_exp(v, s_u, s_v, a_win, a_loss):
    '''vectorised engine of the CPT power utility'''
    if v > s_v:  # gains side
        _v = (v - s_v)/(1.0 - s_v)
        if a_win == 0:
            u = _v
        else:
            u = ((1.0 - np.exp(-a_win*_v))/(1.0 - np.exp(-a_win)))
        u = s_u + u*(1.0 - s_u) 

    elif v < s_v:  # losses side
        _v = v/s_v
        if a_loss == 0:
            u = _v
        else:
            u = (1.0 - np.exp(-a_loss*_v))/(1.0 - np.exp(-a_loss))        
        u = u * s_u
    else:
        u = s_u
    return u

def cpt_exp(v, s_u, s_v, a, a_loss=None, wr_ratio=None, *args, **kwargs):
    '''Calculates the CPT power utility function

    Parameters
    ----------
    v : float, ndarray
        Array containing the normalised attribute values
    s_u : float, ndarray
        Array containing the utility for the reference point
    s_v : float, ndarray
        Array containing the attribute value for the reference point
    r : float, ndarray
        Exponent parameter in the gains side
    r_loss : float, ndarray (optional)
        Exponent parameter in the Loss side. By default is the inverse of the
        gains parameter
    gl_ratio : float, ndarray (optional)
        Defines the exponent parameter loss in terms of the gains-loss ratio. 
        This is optinal and does not override the defition of r_loss
        
    returns
    out : ndarray
        Utility values

    Note
    ----
    This is executed as a vectorized function

    '''
    if np.max(v) > 1 or np.min(v) < 0:
        raise ValueError('v has to be in the range [0,1]')
        
    if np.min(s_u) < 0.0 or np.max(s_u) > 1.0:
        _msj = 's_u values have to be in the range [0,1]'
        raise ValueError(_msj)
    
    if np.min(s_v) < 0.0 or np.max(s_v) > 1.0:
        _msj = 's_v values have to be in the range [0,1]'
        raise ValueError(_msj)
        
    if a_loss is None and wr_ratio is None:
        a_loss = -a
    
    elif wr_ratio is not None and a_loss is None:
        a_loss = -a*wr_ratio
        
    return _cpt_exp(v, s_u, s_v, a, a_loss)


if __name__ == '__main__':
    from time import time
    import matplotlib.pyplot as plt
    
    n = 1000
    v = np.linspace(0, 1, n)
    # r = np.random.uniform(-10, 10, n)
    # v = np.linspace(0, 1, n)
    alp = 0.3
    bet = 1/0.3
    lam = 1.0
    p = 0.1
    
    a = time()
    # exponential(v, r)
    # res = cpt(v, alp, bet, lam)
    res = cpt_exp(v, 0.5, 0.5, alp, bet)
#    res = cpt_pow(v, 0.5, 0.5, alp, bet)
    # res = power(v, p)
    print(time() - a)
    
    plt.plot(v, res)
    plt.grid()
    plt.show()
