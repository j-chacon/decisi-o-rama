# -*- coding: utf-8 -*-
"""
Created on Thu May  9 11:47:00 2019

@author: jchaconhurtado
"""
import numpy as np


# This is the value module. Here all the value funcitons are created
def exponential(v, r):
    '''calculate exponential utility'''
    if r == 0.0:
        out = v
    else:
        out = (1.0 - np.exp(-r*v)) / (1.0 - np.exp(-r))
    return out
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    r = -2
    v = np.linspace(0,1,200)
    out = exponential(v, r)
    plt.plot(v,v, '--', label='Neutral')
    plt.plot(v, out, label='Exp')
    plt.grid()
    plt.legend()
    plt.show()

