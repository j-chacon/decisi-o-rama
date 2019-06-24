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

np.seterr(divide='ignore')

def test_additive_aggregation_single_weight():
    '''
    
    '''
    utils = np.array([[0.0 ,1.0], 
                      [1.0, 0.0], 
                      [0.5, 0.5]])
    w = np.array([0.8, 0.2])
    res = decisiorama.pda.additive(utils, w)
    
    assert(res.ndim == 1)  # dimensions of the results have to be 1
    assert(np.isclose(res[0], 0.2))  # only considering 
    assert(np.isclose(res[1], 0.8))
    assert(np.isclose(res[2], 0.5))
    
#test_additive_aggregation_single_weight()

def test_additive_aggregation_variable_weight():
    '''
    
    '''
    utils = np.array([[0.0 ,1.0], 
                      [1.0, 0.0], 
                      [0.5, 0.5]])
    w = np.array([[0.8, 0.2], 
                  [0.8, 0.2], 
                  [0.8, 0.2]])
    res = decisiorama.pda.additive(utils, w)
    
    assert(res.ndim == 1)  # dimensions of the results have to be 1
    assert(np.isclose(res[0], 0.2))  # only considering 
    assert(np.isclose(res[1], 0.8))
    assert(np.isclose(res[2], 0.5))
    
def test_additive_dimension_error():
    '''
    
    '''
    utils = np.array([0, 1, 0.5])
    w = np.array([0.8, 0.2])
    
    with pytest.raises(ValueError):
        decisiorama.pda.additive(utils, w)
#test_additive_dimension_error()

def test_additive_w_utils_dimension_mismatch():
    '''
    
    '''
    utils = np.array([[0.0 ,1.0], 
                      [1.0, 0.0], 
                      [0.5, 0.5]])
    w = np.array([0.5, 0.2, 0.3])
    
    with pytest.raises(ValueError):
        decisiorama.pda.additive(utils, w)
#test_additive_w_sols_dimension_mismatch()

#%%

def test_cobb_douglas_aggregation_single_weight():
    '''
    
    '''
    utils = np.array([[0.0 ,1.0], 
                      [1.0, 0.0], 
                      [0.5, 0.5]])
    w = np.array([0.8, 0.2])
    res = decisiorama.pda.cobb_douglas(utils, w)
    
    assert(res.ndim == 1)  # dimensions of the results have to be 1
    assert(np.isclose(res[0], 0.0))  # only considering 
    assert(np.isclose(res[1], 0.0))
    assert(np.isclose(res[2], 0.5))

def test_cobb_douglas_aggregation_variable_weight():
    '''
    
    '''
    utils = np.array([[0.0 ,1.0], 
                      [1.0, 0.0], 
                      [0.5, 0.5]])
    w = np.array([[0.8, 0.2], 
                  [0.8, 0.2], 
                  [0.8, 0.2]])
    res = decisiorama.pda.cobb_douglas(utils, w)
    
    assert(res.ndim == 1)  # dimensions of the results have to be 1
    assert(np.isclose(res[0], 0.0))  # only considering 
    assert(np.isclose(res[1], 0.0))
    assert(np.isclose(res[2], 0.5))
    
def test_cobb_douglas_dimension_error():
    '''
    
    '''
    utils = np.array([0, 1, 0.5])
    w = np.array([0.8, 0.2])
    
    with pytest.raises(ValueError):
        decisiorama.pda.cobb_douglas(utils, w)

def test_cobb_douglas_w_utils_dimension_mismatch():
    '''
    In this test the weights of the cobb-douglas do not match the solutions
    '''
    utils = np.array([[0.0 ,1.0], 
                      [1.0, 0.0]])
    w = np.array([0.5, 0.2, 0.3])
    
    with pytest.raises(ValueError):
        decisiorama.pda.cobb_douglas(utils, w)
        
        
def test_mixed_linear_aggregation_single_weight():
    '''
    In the this test we checked on the numerical results of the mixed linear 
    using static weights
    '''
    utils = np.array([[0.0 ,1.0], 
                      [1.0, 0.0], 
                      [0.5, 0.5]])
    w = np.array([0.8, 0.2])
    
    res = decisiorama.pda.mix_linear_cobb(utils, w, pars=[1.0,])
    assert(res.ndim == 1)  # dimensions of the results have to be 1
    
    #all linear additive model
    assert(np.isclose(res[0], 0.2))  # only considering 
    assert(np.isclose(res[1], 0.8))
    assert(np.isclose(res[2], 0.5))
    
    # all cpbb-douglas
    res = decisiorama.pda.mix_linear_cobb(utils, w, pars=[0.0,])
    
    assert(np.isclose(res[0], 0.0))  # only considering 
    assert(np.isclose(res[1], 0.0))
    assert(np.isclose(res[2], 0.5))
    
    # half and half
    res = decisiorama.pda.mix_linear_cobb(utils, w, pars=[0.5,])
    
    assert(np.isclose(res[0], 0.1))  # only considering 
    assert(np.isclose(res[1], 0.4))
    assert(np.isclose(res[2], 0.5))
    

def test_mixed_linear_aggregation_variable_weight():
    '''
    In the this test we checked on the numerical results of the mixed linear 
    using variable weights
    '''
    utils = np.array([[0.0, 1.0], 
                      [1.0, 0.0], 
                      [0.5, 0.5]])
    w = np.array([[0.8, 0.2], 
                  [0.8, 0.2], 
                  [0.8, 0.2]])
    
    res = decisiorama.pda.mix_linear_cobb(utils, w, pars=[1.0,])
    assert(res.ndim == 1)  # dimensions of the results have to be 1
    
    #all linear additive model
    assert(np.isclose(res[0], 0.2))  # only considering 
    assert(np.isclose(res[1], 0.8))
    assert(np.isclose(res[2], 0.5))
    
    # all cpbb-douglas
    res = decisiorama.pda.mix_linear_cobb(utils, w, pars=[0.0,])
    
    assert(np.isclose(res[0], 0.0))  # only considering 
    assert(np.isclose(res[1], 0.0))
    assert(np.isclose(res[2], 0.5))
    
    # half and half
    res = decisiorama.pda.mix_linear_cobb(utils, w, pars=[0.5,])
    
    assert(np.isclose(res[0], 0.1))  # only considering 
    assert(np.isclose(res[1], 0.4))
    assert(np.isclose(res[2], 0.5))
    
def test_mixed_linear_dimension_error():
    '''
    In this test the solution vector is ill formed
    '''
    utils = np.array([0, 1, 0.5])
    w = np.array([0.8, 0.2])
    
    with pytest.raises(ValueError):
        decisiorama.pda.mix_linear_cobb(utils, w, [1.0,])

def test_mixed_linear_w_utils_dimension_mismatch():
    '''
    In this test the weights of the mix linear do not match the solutions
    '''
    utils = np.array([[0.0, 1.0], 
                      [1.0, 0.0], 
                      [0.5, 0.5]])
    w = np.array([0.5, 0.2, 0.3])
    
    with pytest.raises(ValueError):
        decisiorama.pda.mix_linear_cobb(utils, w, [1.0, ])
        
#%%  reverse_harmonic tests
def test_reverse_harmonic_single_w():
    utils = np.array([0.0, 1.0])
    w = np.array([0.8, 0.2])
    res =  decisiorama.pda.reverse_harmonic(utils, w)
    assert(np.isclose(res[0], 1.0))
    
    utils = np.array([[0.0, 1.0], 
                      [1.0, 0.0], 
                      [0.5, 0.5]])
    w = np.array([0.8, 0.2])
    res =  decisiorama.pda.reverse_harmonic(utils, w)
    assert(np.isclose(res[0], 1.0))
    assert(np.isclose(res[1], 1.0))
    assert(np.isclose(res[2], 0.5))
    
def test_reverse_harmonic_variable_w():
    utils = np.array([[0.0, 1.0], 
                      [1.0, 0.0], 
                      [0.5, 0.5]])
    w = np.array([[0.8, 0.2], 
                  [0.8, 0.2], 
                  [0.8, 0.2]])
    res =  decisiorama.pda.reverse_harmonic(utils, w)
    assert(np.isclose(res[0], 1.0))
    assert(np.isclose(res[1], 1.0))
    assert(np.isclose(res[2], 0.5))
    
def test_reverse_harmonic_w_utils_dimension_mismatch():
    '''
    In this test the weights of the mix linear do not match the solutions
    '''
    utils = np.array([[0.0, 1.0], 
                      [1.0, 0.0], 
                      [0.5, 0.5]])
    w = np.array([0.5, 0.2, 0.3])
    
    with pytest.raises(ValueError):
        decisiorama.pda.reverse_harmonic(utils, w, [1.0, ])
        
def test_reverse_harmonic_dimension_error():
    '''
    In this test the solution vector is ill formed
    '''
    utils = np.array([0, 1, 0.5])
    w = np.array([0.8, 0.2])
    
    with pytest.raises(ValueError):
        decisiorama.pda.reverse_harmonic(utils, w, [1.0,])     
        
#%%  reverse_power tests
def test_reverse_power_single_w():
    utils = np.array([0.0, 1.0])
    w = np.array([0.8, 0.2])
    alpha = 1.0
    res =  decisiorama.pda.reverse_power(utils, w, alpha)
    assert(np.isclose(res[0], 0.2))
    
    utils = np.array([[0.0, 1.0], 
                      [1.0, 0.0], 
                      [0.5, 0.5]])
    w = np.array([0.8, 0.2])
    alpha = 1.0
    res =  decisiorama.pda.reverse_power(utils, w, alpha)
    assert(np.isclose(res[0], 0.2))
    assert(np.isclose(res[1], 0.8))
    assert(np.isclose(res[2], 0.5))
    
def test_reverse_power_variable_w():
    utils = np.array([[0.0, 1.0], 
                      [1.0, 0.0], 
                      [0.5, 0.5]])
    w = np.array([[0.8, 0.2], 
                  [0.8, 0.2], 
                  [0.8, 0.2]])
    alpha = 1.0
    res =  decisiorama.pda.reverse_power(utils, w, alpha)
    assert(np.isclose(res[0], 0.2))
    assert(np.isclose(res[1], 0.8))
    assert(np.isclose(res[2], 0.5))
    
def test_reverse_power_w_utils_dimension_mismatch():
    '''
    In this test the weights of the mix linear do not match the solutions
    '''
    utils = np.array([[0.0, 1.0], 
                      [1.0, 0.0], 
                      [0.5, 0.5]])
    w = np.array([0.5, 0.2, 0.3])
    alpha = 1.0
    with pytest.raises(ValueError):
        decisiorama.pda.reverse_power(utils, w, alpha)
        
def test_reverse_power_dimension_error():
    '''
    In this test the solution vector is ill formed
    '''
    utils = np.array([0, 1, 0.5])
    w = np.array([0.8, 0.2])
    alpha = 1.0
    with pytest.raises(ValueError):
        decisiorama.pda.reverse_power(utils, w, alpha)

def test_reverse_power_single_w_variable_alpha():
    utils = np.array([0.0, 1.0])
    w = np.array([0.8, 0.2])
    alpha = np.array([1.0, ])
    res =  decisiorama.pda.reverse_power(utils, w, alpha)
    assert(np.isclose(res[0], 0.2))
    
    utils = np.array([[0.0, 1.0], 
                      [1.0, 0.0], 
                      [0.5, 0.5]])
    w = np.array([0.8, 0.2])
    alpha = np.array([1.0, 1.0, 1.0])
    res =  decisiorama.pda.reverse_power(utils, w, alpha)
    assert(np.isclose(res[0], 0.2))
    assert(np.isclose(res[1], 0.8))
    assert(np.isclose(res[2], 0.5))
    
def test_reverse_power_variable_w_variable_alpha():
    utils = np.array([[0.0, 1.0], 
                      [1.0, 0.0], 
                      [0.5, 0.5]])
    w = np.array([[0.8, 0.2], 
                  [0.8, 0.2], 
                  [0.8, 0.2]])
    alpha = np.array([1.0, 1.0, 1.0])
    res =  decisiorama.pda.reverse_power(utils, w, alpha)
    assert(np.isclose(res[0], 0.2))
    assert(np.isclose(res[1], 0.8))
    assert(np.isclose(res[2], 0.5))
    
def test_reverse_power_w_utils_dimension_mismatch_variable_alpha():
    '''
    In this test the weights of the mix linear do not match the solutions
    '''
    utils = np.array([[0.0, 1.0], 
                      [1.0, 0.0], 
                      [0.5, 0.5]])
    w = np.array([0.5, 0.2, 0.3])
    alpha = np.array([1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        decisiorama.pda.reverse_power(utils, w, alpha)
        
def test_reverse_power_dimension_error_variable_alpha():
    '''
    In this test the solution vector is ill formed
    '''
    utils = np.array([0, 1, 0.5])
    w = np.array([0.8, 0.2])
    alpha = np.array([1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        decisiorama.pda.reverse_power(utils, w, alpha)        

def test_reverse_power_alpha_dimension_error():
    '''
    In this test the alpha vector is ill formed
    '''
    utils = np.array([0, 1, 0.5])
    w = np.array([0.8, 0.2])
    alpha = np.array([1.0, 1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        decisiorama.pda.reverse_power(utils, w, alpha)
        
#%%  split_power tests
def test_split_power_single_w():
    utils = np.array([0.0, 1.0])
    w = np.array([0.8, 0.2])
    alpha = 1.0
    s = 1.0
    res =  decisiorama.pda.split_power(utils, w, alpha, s)
    assert(np.isclose(res[0], 0.2))
    
    utils = np.array([[0.0, 1.0], 
                      [1.0, 0.0], 
                      [0.5, 0.5]])
    w = np.array([0.8, 0.2])
    alpha = 1.0
    s = 1.0
    res =  decisiorama.pda.split_power(utils, w, alpha, s)
    assert(np.isclose(res[0], 0.2))
    assert(np.isclose(res[1], 0.8))
    assert(np.isclose(res[2], 0.5))
    
def test_split_power_variable_w():
    utils = np.array([[0.0, 1.0], 
                      [1.0, 0.0], 
                      [0.5, 0.5]])
    w = np.array([[0.8, 0.2], 
                  [0.8, 0.2], 
                  [0.8, 0.2]])
    alpha = 1.0
    s = 1.0
    res =  decisiorama.pda.split_power(utils, w, alpha, s)
    assert(np.isclose(res[0], 0.2))
    assert(np.isclose(res[1], 0.8))
    assert(np.isclose(res[2], 0.5))
    
def test_split_power_w_utils_dimension_mismatch():
    '''
    In this test the weights of the mix linear do not match the solutions
    '''
    utils = np.array([[0.0, 1.0], 
                      [1.0, 0.0], 
                      [0.5, 0.5]])
    w = np.array([0.5, 0.2, 0.3])
    alpha = 1.0
    s = 1.0
    with pytest.raises(ValueError):
        decisiorama.pda.split_power(utils, w, alpha, s)
        
def test_split_power_dimension_error():
    '''
    In this test the solution vector is ill formed
    '''
    utils = np.array([0, 1, 0.5])
    w = np.array([0.8, 0.2])
    alpha = 1.0
    s = 1.0
    with pytest.raises(ValueError):
        decisiorama.pda.split_power(utils, w, alpha, s)

def test_split_power_single_w_variable_alpha():
    utils = np.array([0.0, 1.0])
    w = np.array([0.8, 0.2])
    alpha = np.array([1.0, ])
    s = 1.0
    res =  decisiorama.pda.split_power(utils, w, alpha, s)
    assert(np.isclose(res[0], 0.2))
    
    utils = np.array([[0.0, 1.0], 
                      [1.0, 0.0], 
                      [0.5, 0.5]])
    w = np.array([0.8, 0.2])
    alpha = np.array([1.0, 1.0, 1.0])
    s = 1.0
    res =  decisiorama.pda.split_power(utils, w, alpha, s)
    assert(np.isclose(res[0], 0.2))
    assert(np.isclose(res[1], 0.8))
    assert(np.isclose(res[2], 0.5))
    
def test_split_power_variable_w_variable_alpha():
    utils = np.array([[0.0, 1.0], 
                      [1.0, 0.0], 
                      [0.5, 0.5]])
    w = np.array([[0.8, 0.2], 
                  [0.8, 0.2], 
                  [0.8, 0.2]])
    alpha = np.array([1.0, 1.0, 1.0])
    s = 1.0
    res =  decisiorama.pda.split_power(utils, w, alpha, s)
    assert(np.isclose(res[0], 0.2))
    assert(np.isclose(res[1], 0.8))
    assert(np.isclose(res[2], 0.5))
    
def test_split_power_w_utils_dimension_mismatch_variable_alpha():
    '''
    In this test the weights of the mix linear do not match the solutions
    '''
    utils = np.array([[0.0, 1.0], 
                      [1.0, 0.0], 
                      [0.5, 0.5]])
    w = np.array([0.5, 0.2, 0.3])
    alpha = np.array([1.0, 1.0, 1.0])
    s = 1.0
    with pytest.raises(ValueError):
        decisiorama.pda.split_power(utils, w, alpha, s)
        
def test_split_power_dimension_error_variable_alpha():
    '''
    In this test the solution vector is ill formed
    '''
    utils = np.array([0, 1, 0.5])
    w = np.array([0.8, 0.2])
    alpha = np.array([1.0, 1.0, 1.0])
    s = 1.0
    with pytest.raises(ValueError):
        decisiorama.pda.split_power(utils, w, alpha, s)

def test_split_power_alpha_dimension_error():
    '''
    In this test the alpha vector is ill formed
    '''
    utils = np.array([0, 1, 0.5])
    w = np.array([0.8, 0.2])
    alpha = np.array([1.0, 1.0, 1.0, 1.0])
    s = 1.0
    with pytest.raises(ValueError):
        decisiorama.pda.split_power(utils, w, alpha, s)
        
   
def test_split_power_variable_s():
    utils = np.array([[0.0, 1.0], 
                      [1.0, 0.0], 
                      [0.5, 0.5]])
    w = np.array([[0.8, 0.2], 
                  [0.8, 0.2], 
                  [0.8, 0.2]])
    alpha = 1.0
    s = np.array([1.0, 1.0, 1.0])
    res =  decisiorama.pda.split_power(utils, w, alpha, s)
    assert(np.isclose(res[0], 0.2))
    assert(np.isclose(res[1], 0.8))
    assert(np.isclose(res[2], 0.5))
    
def test_split_power_s_utils_dimension_mismatch():
    '''
    In this test the weights of the mix linear do not match the solutions
    '''
    utils = np.array([[0.0, 1.0], 
                      [1.0, 0.0], 
                      [0.5, 0.5]])
    w = np.array([0.5, 0.2, 0.3])
    alpha = 1.0
    s = s = np.array([1.0, 1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        decisiorama.pda.split_power(utils, w, alpha, s)
        
def test_split_power_single_w_variable_s():
    utils = np.array([0.0, 1.0])
    w = np.array([0.8, 0.2])
    alpha = np.array([1.0, ])
    s = np.array([1.0, ])
    res =  decisiorama.pda.split_power(utils, w, alpha, s)
    assert(np.isclose(res[0], 0.2))
    
    utils = np.array([[0.0, 1.0], 
                      [1.0, 0.0], 
                      [0.5, 0.5]])
    w = np.array([0.8, 0.2])
    alpha = np.array([1.0, 1.0, 1.0])
    s = np.array([1.0, 1.0, 1.0])
    res =  decisiorama.pda.split_power(utils, w, alpha, s)
    assert(np.isclose(res[0], 0.2))
    assert(np.isclose(res[1], 0.8))
    assert(np.isclose(res[2], 0.5))
    
def test_split_power_variable_w_variable_s():
    utils = np.array([[0.0, 1.0], 
                      [1.0, 0.0], 
                      [0.5, 0.5]])
    w = np.array([[0.8, 0.2], 
                  [0.8, 0.2], 
                  [0.8, 0.2]])
    alpha = np.array([1.0, 1.0, 1.0])
    s = np.array([1.0, 1.0, 1.0])
    res =  decisiorama.pda.split_power(utils, w, alpha, s)
    assert(np.isclose(res[0], 0.2))
    assert(np.isclose(res[1], 0.8))
    assert(np.isclose(res[2], 0.5))
            
def test_split_power_dimension_error_variable_s():
    '''
    In this test the s vector is ill formed
    '''
    utils = np.array([0, 1, 0.5])
    w = np.array([0.8, 0.2])
    alpha = np.array([1.0, 1.0, 1.0])
    s = np.array([1.0, 1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        decisiorama.pda.split_power(utils, w, alpha, s)

#%% 
def test_harmonic_aggregation_single_weight():
    '''
    
    '''
    utils = np.array([[0.0 ,1.0], 
                      [1.0, 0.0], 
                      [0.5, 0.5]])
    w = np.array([0.8, 0.2])
    res = decisiorama.pda.harmonic(utils, w)
    
    assert(res.ndim == 1)  # dimensions of the results have to be 1
    assert(np.isclose(res[0], 0.0))  # only considering 
    assert(np.isclose(res[1], 0.0))
    assert(np.isclose(res[2], 0.5))
    
#test_harmonic_aggregation_single_weight()

def test_harmonic_aggregation_variable_weight():
    '''
    
    '''
    utils = np.array([[0.0 ,1.0], 
                      [1.0, 0.0], 
                      [0.5, 0.5]])
    w = np.array([[0.8, 0.2], 
                  [0.8, 0.2], 
                  [0.8, 0.2]])
    res = decisiorama.pda.harmonic(utils, w)
    
    assert(res.ndim == 1)  # dimensions of the results have to be 1
    assert(np.isclose(res[0], 0.0))  # only considering 
    assert(np.isclose(res[1], 0.0))
    assert(np.isclose(res[2], 0.5))
    
def test_harmonic_dimension_error():
    '''
    
    '''
    utils = np.array([0, 1, 0.5])
    w = np.array([0.8, 0.2])
    
    with pytest.raises(ValueError):
        decisiorama.pda.harmonic(utils, w)
#test_harmonic_dimension_error()

def test_harmonic_w_utils_dimension_mismatch():
    '''
    
    '''
    utils = np.array([[0.0 ,1.0], 
                      [1.0, 0.0], 
                      [0.5, 0.5]])
    w = np.array([0.5, 0.2, 0.3])
    
    with pytest.raises(ValueError):
        decisiorama.pda.harmonic(utils, w)
#test_harmonic_w_sols_dimension_mismatch()
        
#%% maximum

def test_maximum_aggregation_single_weight():
    '''
    
    '''
    utils = np.array([[0.0 ,1.0], 
                      [1.0, 0.0], 
                      [0.5, 0.5]])
    res = decisiorama.pda.maximum(utils)
    assert(res.ndim == 1)  # dimensions of the results have to be 1
    assert(np.isclose(res[0], 1.0))  # only considering 
    assert(np.isclose(res[1], 1.0))
    assert(np.isclose(res[2], 0.5))
    
#test_maximum_aggregation_single_weight()

#%% minimum

def test_minimum_aggregation_single_weight():
    '''
    
    '''
    utils = np.array([[0.0 ,1.0], 
                      [1.0, 0.0], 
                      [0.5, 0.5]])
    res = decisiorama.pda.minimum(utils)
    assert(res.ndim == 1)  # dimensions of the results have to be 1
    assert(np.isclose(res[0], 0.0))  # only considering 
    assert(np.isclose(res[1], 0.0))
    assert(np.isclose(res[2], 0.5))
    
#test_maximum_aggregation_single_weight()


##%%
#def test_mix_cobb_additive_additive():
#    utils = np.array([[0.0, 1.0], 
#                      [1.0, 0.0], 
#                      [0.5, 0.5]])
#    w = np.array([[0.8, 0.2], 
#                  [0.8, 0.2], 
#                  [0.8, 0.2]])
#    methods = [cobb_douglas, 
#               additive,]
#    w_methods = np.array([0.5, 0.5])
#    mix_fun = additive
#    res = mix(utils, w, methods, w_methods, mix_fun)
#    
#    assert(np.isclose(res[0], 0.1))
#    assert(np.isclose(res[1], 0.4))
#    assert(np.isclose(res[2], 0.5))
#    
#    
#    utils = np.array([[0.0, 1.0], 
#                  [1.0, 0.0], 
#                  [0.5, 0.5]])
#    w = np.array([[0.8, 0.2], 
#                  [0.8, 0.2], 
#                  [0.8, 0.2]])
#    methods = [cobb_douglas, 
#               split_power,]
#    methods_args = [{}, 
#                    dict(alpha = 1.0, s = 1.0)]
#    w_methods = np.array([0.5, 0.5])
#    mix_fun = additive
#    res = mix(utils, w, methods, w_methods, mix_fun, methods_args=methods_args)
#    assert(np.isclose(res[0], 0.1))
#    assert(np.isclose(res[1], 0.4))
#    assert(np.isclose(res[2], 0.5))   
#    
#    utils = np.array([[0.0, 1.0], 
#                  [1.0, 0.0], 
#                  [0.5, 0.5]])
#    w = np.array([[0.8, 0.2], 
#                  [0.8, 0.2], 
#                  [0.8, 0.2]])
#    methods = [cobb_douglas, 
#               additive,]
#    mix_args = dict(alpha = 1.0, s = 1.0)
#    w_methods = np.array([0.5, 0.5])
#    mix_fun = split_power
#    
#    res = mix(utils, w, methods, w_methods, mix_fun, mix_args=mix_args)
#    assert(np.isclose(res[0], 0.1))
#    assert(np.isclose(res[1], 0.4))
#    assert(np.isclose(res[2], 0.5))
#    
#    
#    utils = np.array([[0.0, 1.0], 
#                  [1.0, 0.0], 
#                  [0.5, 0.5]])
#    w = np.array([[0.8, 0.2], 
#                  [0.8, 0.2], 
#                  [0.8, 0.2]])
#    methods = [cobb_douglas, 
#               additive,]
#    mix_args = dict(alpha = 1.0, s = 1.0)
#    w_methods = np.array([[0.5, 0.5],
#                          [0.5, 0.5],
#                          [0.5, 0.5]])
#    mix_fun = split_power
#    res = mix(utils, w, methods, w_methods, mix_fun, mix_args=mix_args)
#    
#    #>>> [0.1 0.4 0.5]
#    
#    
#    utils = np.array([[0.0, 1.0], 
#                  [1.0, 0.0], 
#                  [0.5, 0.5]])
#    w = np.array([[0.8, 0.2], 
#                  [0.8, 0.2], 
#                  [0.8, 0.2]])
#    methods = [cobb_douglas, 
#               additive,]
#    mix_args = dict(alpha = np.array([1.0, 1.0, 1.0]), s = 1.0)
#    w_methods = np.array([[0.5, 0.5],
#                          [0.5, 0.5],
#                          [0.5, 0.5]])
#    mix_fun = split_power
#    print(mix(utils, w, methods, w_methods, mix_fun, mix_args=mix_args))
#    #>>> [0.1 0.4 0.5]        