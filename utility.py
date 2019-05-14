# -*- coding: utf-8 -*-
"""
Created on Thu May  9 11:47:00 2019

@author: jchaconhurtado
"""
import numpy as np
from random_instance import check_numeric

#OFFSET_VAL = np.random.uniform(-0.00000000001, 0.00000000001)
# This is the value module. Here all the value funcitons are created
def exponential(v, pars):
    '''calculate exponential utility'''
    
    if check_numeric(pars[0]):
        r = pars[0]
    else:
        r = pars[0]()
        
    
    if type(r) is float or type(r) is np.float64 or type(r) is np.float32:
        if r == 0.0:
            out = v
        else:
            out = (1.0 - np.exp(-r*v)) / (1.0 - np.exp(-r))
    else:
        out = []
        for _r in r:
            if _r == 0.0:
                out.append(v)
            else:
                out.append((1.0 - np.exp(-_r*v)) / (1.0 - np.exp(-_r)))
        out = np.array(out)
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

