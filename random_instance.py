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

##%% performance test for trncnorm vs truncnormal
#from time import time
#n = 10000000
#res = np.zeros(n)
#single = truncnormal(0,1,-2, 2).get
#
#a = time()
#for i in range(n):
#    res[i] = single()
#print(time() - a)
#
#
#a = time()
#res = truncnorm.rvs(0, 1,-2, 2, n)
#print(time() - a)

#%%
# Define the generator of the tuncated normal distributions
def _tn(mu, sigma, lower=-np.inf, upper=np.inf, size=None):
    
    if size is None:
        out = random.normal()
        while lower > out > upper:
            out = random.normal()
    else:
        a =  (lower - mu) / sigma
        b =  (upper - mu) / sigma
        out = truncnorm.rvs(a, b, mu, sigma, size)
    return out

class normal():
    def __init__(self, mu, std, n=None):
        self.mu = mu
        self.std = std
        self.n = n
    
    def get(self, n=None):
        if n is None and self.n is not None:
            n = self.n
        return random.normal(self.mu, self.std, n)
    
class beta():
    def __init__(self, a, b, n=None):
        self.a = a
        self.b = b
        self.n = n
    
    def get(self, n=None):
        if n is None and self.n is not None:
            n = self.n
        return random.beta(self.a, self.b, n)
    
class uniform():
    def __init__(self, a, b, n=None):
        self.a = a
        self.b = b
        self.n = n
        
    def get(self, n=None):
        if n is None and self.n is not None:
            n = self.n
        return random.uniform(self.a, self.b, n)
    
class lognormal():
    def __init__(self, mu, std, n=None):
        self.mu = mu
        self.std = std
        self.n = n
        return
    
    def get(self, n=None):
        if n is None and self.n is not None:
            n = self.n
        return random.lognormal(self.mu, self.std, n)

class truncnormal():
    def __init__(self, mu, std, a=-np.inf, b=np.inf, n=None):
        self.mu = mu
        self.std = std
        self.a = a
        self.b = b
        self.n = n
        
        return
    
    def get(self, n=None):
        if n is None and self.n is not None:
            n = self.n
        
        return _tn(self.mu, self.std, self.a, self.b, n)
if __name__ == '__main__':
    
    print(normal(0, 1).get(2))
    print(beta(1, 2).get(2))
    print(uniform(0, 1).get(2))
    print(lognormal(1, 2).get(2))
    print(truncnormal(1, 1, 0).get(2))
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