# -*- coding: utf-8 -*-
"""
Created on Wed May 15 09:23:17 2019

@author: jchaconhurtado

All of the functions to make a ranking. ranking are made on pda_fun

"""


import numpy as np
#import utils

def iqr(sols, lq=0.25, uq=0.75):
    '''Calculate the interquantile range. shape is [solutions, n]'''    
    return np.quantile(sols, uq, axis=1) - np.quantile(sols, lq, axis=1)

def mean(sols):
    return np.average(sols, axis=1)

def std(sols):
    return np.std(sols, axis=1)

def cov(sols):
    return np.std(sols, axis=1)/np.average(sols, axis=1)


