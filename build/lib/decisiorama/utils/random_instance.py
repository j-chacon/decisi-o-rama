# -*- coding: utf-8 -*-
"""
Created on Tue May 14 10:04:20 2019

@author: jchaconhurtado

library to create random numbers on demand
"""

import numpy as np
from numpy import random
from scipy.stats import truncnorm

def _isnumeric(x, name):
    if not isinstance(x, (float, int, np.float, np.float32, np.float64, 
                          type(None))):
        msg = '''{0} is not of numeric type. got {1}'''.format(name, type(x))
        raise TypeError(msg)
    return 

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

class RandomNumber():
    a = None
    b = None
    n = None
    mu = None
    std = None
    
    def __init__(self,):
        pass

    
    def check_numeric(self):
        # Check if the values are numeric
        tags = [key for key in self.__dict__.keys()]
        for tag in tags:
            _isnumeric(self.__dict__[tag], tag)
        
        

    def check_lims(self):
        if self.a is not None or self.b is not None:
            if self.a > self.b:
                msg = ("The lower limit (a = {0}) cannot be larger than the "
                       "upper limit (b = {1})".format(self.a, self.b))
                raise ValueError(msg)

    
    def __getstate__(self):
        state = dict(mu = self.mu,
                     std = self.std,
                     n = self.n,
                     a = self.a,
                     b = self.b,
                     )
        return state
    
    def __setstate__(self, state):
        self.mu = state['mu']
        self.std = state['std']
        self.n = state['n']
        self.a = state['a']
        self.b = state['b']
        return

class Normal(RandomNumber):
    def __init__(self, mu, std, n=None):

        self.mu = mu
        self.std = std
        self.n = n
        
        self.check_numeric()
        self.check_lims()
    
    def get(self, n=None):
        if n is None and self.n is not None:
            n = self.n
        return random.normal(self.mu, self.std, n)

    
class Beta(RandomNumber):
    def __init__(self, a, b, n=None):
        self.a = a
        self.b = b
        self.n = n
        
        self.check_numeric()
    
    def get(self, n=None):
        if n is None and self.n is not None:
            n = self.n
        return random.beta(self.a, self.b, n)

    
class Uniform(RandomNumber):
    def __init__(self, a, b, n=None):
        # Check if the values are numeric
        self.a = a
        self.b = b
        self.n = n
        
        self.check_numeric()
        self.check_lims()
        
    def get(self, n=None):
        if n is None and self.n is not None:
            n = self.n
        return random.uniform(self.a, self.b, n)

    
class Lognormal(RandomNumber):
    def __init__(self, mu, std, n=None):
        self.mu = mu
        self.std = std
        self.n = n
        
        self.check_numeric()
        return
    
    def get(self, n=None):
        if n is None and self.n is not None:
            n = self.n
        return random.lognormal(self.mu, self.std, n)
    
    
class Truncnormal(RandomNumber):
    def __init__(self, mu, std, a=-np.inf, b=np.inf, n=None):
        self.mu = mu
        self.std = std
        self.a = a
        self.b = b
        self.n = n
        
        # check if arguments are  correctly passed
        self.check_numeric()
        self.check_lims()
        
        return
    
    def get(self, n=None):
        if n is None and self.n is not None:
            n = self.n
        return _tn(self.mu, self.std, self.a, self.b, n)
    
    
if __name__ == '__main__':
    print(Normal(0, 1).get(2))
    print(Beta(1, 2).get(2))
    print(Uniform(0, 1).get(2))
    print(Lognormal(1, 2).get(2))
    print(Truncnormal(1, 1, 0, 2).get(2))
    print(Normal(1, 1).__getstate__())
#    print(Truncnormal(1, 1, 2, 0).get(2))

