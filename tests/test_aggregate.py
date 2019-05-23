# -*- coding: utf-8 -*-
"""
Created on Thu May 23 14:06:27 2019

@author: jchaconhurtado
"""
from sys import path
path.append('..')

import numpy as np
import decisiorama
import pytest

def test_additive_aggregation_single_weight():
    '''
    
    '''
    s = np.array([[0,1], 
                  [1,0], 
                  [0.5, 0.5]])
    w = np.array([0.8, 0.2])
    res = decisiorama.pda.additive(s, w)
    
    assert(res.ndim == 1)  # dimensions of the results have to be 1
    assert(np.isclose(res[0], 0.2))  # only considering 
    assert(np.isclose(res[1], 0.8))
    assert(np.isclose(res[2], 0.5))
#test_additive_aggregation_single_weight()

def test_additive_aggregation_variable_weight():
    '''
    
    '''
    s = np.array([[0,1], 
                  [1,0], 
                  [0.5, 0.5]])
    w = np.array([[0.8, 0.2], 
                  [0.8, 0.2], 
                  [0.8, 0.2]])
    res = decisiorama.pda.additive(s, w)
    
    assert(res.ndim == 1)  # dimensions of the results have to be 1
    assert(np.isclose(res[0], 0.2))  # only considering 
    assert(np.isclose(res[1], 0.8))
    assert(np.isclose(res[2], 0.5))
    
def test_additive_dimension_error():
    '''
    
    '''
    s = np.array([0, 1, 0.5])
    w = np.array([0.8, 0.2])
    
    with pytest.raises(ValueError):
        decisiorama.pda.additive(s, w)
test_additive_dimension_error()

def test_additive_w_sols_dimension_mismatch():
    '''
    
    '''
    s = np.array([[0,1], 
                  [1,0], 
                  [0.5, 0.5]])
    w = np.array([0.5, 0.2, 0.3])
    
    with pytest.raises(ValueError):
        decisiorama.pda.additive(s, w)
test_additive_w_sols_dimension_mismatch()

def test_cobb_douglas_aggregation_single_weight():
    '''
    
    '''
    s = np.array([[0,1], 
                  [1,0], 
                  [0.5, 0.5]])
    w = np.array([0.8, 0.2])
    res = decisiorama.pda.cobb_douglas(s, w)
    
    assert(res.ndim == 1)  # dimensions of the results have to be 1
    assert(np.isclose(res[0], 0.0))  # only considering 
    assert(np.isclose(res[1], 0.0))
    assert(np.isclose(res[2], 0.5))

def test_cobb_douglas_aggregation_variable_weight():
    '''
    
    '''
    s = np.array([[0,1], 
                  [1,0], 
                  [0.5, 0.5]])
    w = np.array([[0.8, 0.2], 
                  [0.8, 0.2], 
                  [0.8, 0.2]])
    res = decisiorama.pda.cobb_douglas(s, w)
    
    assert(res.ndim == 1)  # dimensions of the results have to be 1
    assert(np.isclose(res[0], 0.0))  # only considering 
    assert(np.isclose(res[1], 0.0))
    assert(np.isclose(res[2], 0.5))
    
def test_cobb_douglas_dimension_error():
    '''
    
    '''
    s = np.array([0, 1, 0.5])
    w = np.array([0.8, 0.2])
    
    with pytest.raises(ValueError):
        decisiorama.pda.cobb_douglas(s, w)

def test_cobb_douglas_w_sols_dimension_mismatch():
    '''
    In this test the weights of the cobb-douglas do not match the solutions
    '''
    s = np.array([[0,1], 
                  [1,0], 
                  [0.5, 0.5]])
    w = np.array([0.5, 0.2, 0.3])
    
    with pytest.raises(ValueError):
        decisiorama.pda.cobb_douglas(s, w)
        
        
def test_mixed_linear_aggregation_single_weight():
    '''
    In the this test we checked on the numerical results of the mixed linear 
    using static weights
    '''
    s = np.array([[0.0, 1.0], 
                  [1.0, 0.0], 
                  [0.5, 0.5]])
    w = np.array([0.8, 0.2])
    
    res = decisiorama.pda.mix_linear_cobb(s, w, pars=[1.0,])
    assert(res.ndim == 1)  # dimensions of the results have to be 1
    
    #all linear additive model
    assert(np.isclose(res[0], 0.2))  # only considering 
    assert(np.isclose(res[1], 0.8))
    assert(np.isclose(res[2], 0.5))
    
    # all cpbb-douglas
    res = decisiorama.pda.mix_linear_cobb(s, w, pars=[0.0,])
    
    assert(np.isclose(res[0], 0.0))  # only considering 
    assert(np.isclose(res[1], 0.0))
    assert(np.isclose(res[2], 0.5))
    
    # half and half
    res = decisiorama.pda.mix_linear_cobb(s, w, pars=[0.5,])
    
    assert(np.isclose(res[0], 0.1))  # only considering 
    assert(np.isclose(res[1], 0.4))
    assert(np.isclose(res[2], 0.5))
    

def test_mixed_linear_aggregation_variable_weight():
    '''
    In the this test we checked on the numerical results of the mixed linear 
    using variable weights
    '''
    s = np.array([[0.0, 1.0], 
                  [1.0, 0.0], 
                  [0.5, 0.5]])
    w = np.array([[0.8, 0.2], 
                  [0.8, 0.2], 
                  [0.8, 0.2]])
    
    res = decisiorama.pda.mix_linear_cobb(s, w, pars=[1.0,])
    assert(res.ndim == 1)  # dimensions of the results have to be 1
    
    #all linear additive model
    assert(np.isclose(res[0], 0.2))  # only considering 
    assert(np.isclose(res[1], 0.8))
    assert(np.isclose(res[2], 0.5))
    
    # all cpbb-douglas
    res = decisiorama.pda.mix_linear_cobb(s, w, pars=[0.0,])
    
    assert(np.isclose(res[0], 0.0))  # only considering 
    assert(np.isclose(res[1], 0.0))
    assert(np.isclose(res[2], 0.5))
    
    # half and half
    res = decisiorama.pda.mix_linear_cobb(s, w, pars=[0.5,])
    
    assert(np.isclose(res[0], 0.1))  # only considering 
    assert(np.isclose(res[1], 0.4))
    assert(np.isclose(res[2], 0.5))
    
def test_mixed_linear_dimension_error():
    '''
    In this test the solution vector is ill formed
    '''
    s = np.array([0, 1, 0.5])
    w = np.array([0.8, 0.2])
    
    with pytest.raises(ValueError):
        decisiorama.pda.mix_linear_cobb(s, w, [1.0,])

def test_mixed_linear_w_sols_dimension_mismatch():
    '''
    In this test the weights of the mix linear do not match the solutions
    '''
    s = np.array([[0,1], 
                  [1,0], 
                  [0.5, 0.5]])
    w = np.array([0.5, 0.2, 0.3])
    
    with pytest.raises(ValueError):
        decisiorama.pda.mix_linear_cobb(s, w, [1.0, ])
        

