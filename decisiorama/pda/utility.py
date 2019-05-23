# -*- coding: utf-8 -*-
"""
Created on Thu May  9 11:47:00 2019

@author: jchaconhurtado
"""
import numpy as np
OFFSET = 1e-6


def exponential(v, pars):
    '''pars is a list '''
    r = pars[0]
    if type(r) is np.ndarray:
        r[r == 0] = OFFSET
    else:
        if r == 0:
            r = OFFSET
    
    return (1.0 - np.exp(-r*v)) / (1.0 - np.exp(-r))

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    r = [2.0,]
    v = np.linspace(0,1,200)
    out = exponential(v, r)
    plt.plot(v,v, '--', label='Neutral')
    plt.plot(v, out, label='Exp')
    plt.grid()
    plt.legend()
    plt.show()

