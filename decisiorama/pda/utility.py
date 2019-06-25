# -*- coding: utf-8 -*-
"""
Created on Thu May  9 11:47:00 2019

@author: jchaconhurtado
"""
import numpy as np

#@np.vectorize
def _exponential(v, r, *args, **kwargs):
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
    if v > 1.0 or v < 0.0:
        _msj = ('Values passed to the utility function should be ranked '
                'normalised between 0 and 1')
        RuntimeWarning(_msj)
    if r == 0:
        out = v
    else:
        out = (1.0 - np.exp(-r*v)) / (1.0 - np.exp(-r))
    return out
exponential = np.vectorize(_exponential)
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    r = 1.2
    v = np.linspace(0,1,200)
    out = exponential(v, r)
    plt.plot(v,v, '--', label='Neutral')
    plt.plot(v, out, label='Exp')
    plt.grid()
    plt.legend()
    plt.show()

