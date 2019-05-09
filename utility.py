# -*- coding: utf-8 -*-
"""
Created on Thu May  9 11:47:00 2019

@author: jchaconhurtado
"""
import numpy as np

UTILITY_THRESHOLD = 0.00001
# This is the value module. Here all the value funcitons are created

def exponential(v, r):
    '''calculate exponential utility'''
    out = (1.0 - np.exp(-r*v)) / (1.0 - np.exp(-r))
    out[r < UTILITY_THRESHOLD] = 0
    return out