# -*- coding: utf-8 -*-
"""
Created on Thu May 23 14:06:27 2019

@author: jchaconhurtado
"""
from sys import path
path.append('..')

import numpy as np
from decisiorama.pda import utility
import pytest

def test_exponential_single_v_r():
    v = 0.0
    r = 1.0
    res = utility.exponential(v, r)
    assert(np.isclose(res, 0.0))
    
    v = 1.0
    r = 1.0
    res = utility.exponential(v, r)
    assert(np.isclose(res, 1.0))
    
    v = 0.0
    r = 0.0
    res = utility.exponential(v, r)
    assert(np.isclose(res, 0.0))
    
    v = 1.0
    r = 0.0
    res = utility.exponential(v, r)
    assert(np.isclose(res, 1.0))
    
    v = 1.0
    r = -1.5
    res = utility.exponential(v, r)
    assert(np.isclose(res, 1.0))
    
    v = 1.0
    r = 1.5
    res = utility.exponential(v, r)
    assert(np.isclose(res, 1.0))
    
def test_exponential_variable_v():
    v = np.linspace(0, 1)
    r = 0.0
    res = utility.exponential(v, r)
    assert(np.all(np.isclose(v, res)))
    

def test_exponential_variable_r():
    v = np.linspace(0, 1)
    r = np.zeros_like(v)
    res = utility.exponential(v, r)
    assert(np.all(np.isclose(v, res)))
    
