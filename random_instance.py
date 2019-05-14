# -*- coding: utf-8 -*-
"""
Created on Tue May 14 10:04:20 2019

@author: jchaconhurtado

library to create random numbers on demand
"""

import numpy as np
from numpy import random
from scipy.stats import truncnorm

def check_numeric(x):
    
    if type(x) is float:
        flag = True
    elif type(x) is np.float:
        flag = True
    elif type(x) is np.float32:
        flag = True
    elif type(x) is np.float64:
        flag = True
    else:
        flag = False
        
    return flag

# Define the generator of the tuncated normal distributions
def _tn(mu, sigma, lower=-np.inf, upper=np.inf, size=None):
    out = random.uniform()
    if size is None:
        while lower > out > upper:
            out = random.uniform()

    return out

class normal():
    def __init__(self, mu, std, n=None):
        self.mu = mu
        self.std = std
        self.n = n
    
    def get(self,):
        return random.normal(self.mu, self.std)
    
class beta():
    def __init__(self, a, b, n=None):
        self.a = a
        self.b = b
        self.n = n
    
    def get(self,):
        return random.beta(self.a, self.b, self.n)
    
class uniform():
    def __init__(self, a, b, n=None):
        self.a = a
        self.b = b
        self.n = n
        
    def get(self,):
        return random.uniform(self.a, self.b, self.n)
    
class lognormal():
    def __init__(self, mu, std, n=None):
        self.mu = mu
        self.std = std
        self.n = n
        return
    
    def get(self,):
        return random.lognormal(self.mu, self.std, self.n)

class truncnormal():
    def __init__(self, mu, std, a=-np.inf, b=np.inf, n=None):
        self.mu = mu
        self.std = std
        self.a = a
        self.b = b
        self.n = n
        
        return
    
    def get(self,):
        ## Does not support a size creation method
        return _tn(self.mu, self.std, self.a, self.b, self.n)
if __name__ == '__main__':
    
    print(normal(0, 1).get())
    print(beta(1, 2).get())
    print(uniform(0, 1).get())
    print(lognormal(1, 2).get())
    print(truncnormal(1, 1, 0).get())
#class uniform():
#    def __init__(self):
#        
#        return
#    
#    def get(self,):
#        return
#
#class uniform():
#    def __init__(self):
#        
#        return
#    
#    def get(self,):
#        return    