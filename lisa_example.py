# -*- coding: utf-8 -*-
"""
Created on Wed May  1 08:57:15 2019

@author: jchaconhurtado
"""


import numpy as np
import matplotlib.pyplot as plt
from numpy.random import beta, normal, lognormal, uniform
from scipy.stats import truncnorm
import itertools
import utils

import generator


# objectives
objectives = [
        [0.0, 100.0],  #rehab - max
        [0.0, 100.0],  #adapt - max
        [0.0, 180.0],  #gwhh - min
        [0.0, 2.000],  #econs - min
        [0.0, 365.0],  #vol_dw - min - 5
        [0.0, 365.0],  #vol_hw - min
        [500.0, 3600.0],  #vol_ffw - max
        [0.0, 0.25],  #reliab_dw - min
        [0.0, 0.25],  #reliab_hw - min
        [0.0, 0.25],  #reliab_ffw - min - 10
        [0.0, 365.0],  #aes_dw - min
        [0.0, 365.0],  #aes_hw - min
        [0.0, 365.0],  #faecal_dw - min
        [0.0, 365.0],  #faecal_hw - min
        [0.0, 2.0],  #cells_dw - min - 15
        [0.0, 2.0],  #cells_hw - min
        [0.0, 20.0],  #no3_dw - min
        [0.0, 0.02],  #pest - min
        [0.0, 150.0],  #bta_dw - min
        [20.0, 95.0],  #efqm - max - 20
        [0.0, 100.0],  #voice - max
        [0.0, 100.0],  #auton - max
        [0.0, 10.0],  #time - min
        [0.0, 10.0],  #area - min
        [1.0, 6.0],  #collab - max - 25
        [0.01, 5.0],  #costcap - min
        [0.0, 5.0],  #costchange - min
            ]

# always 27 objectives
obj_maximise = [True, True, False, False, False, 
                False, True, False, False, False,
                False, False, False, False, False,
                False, False, False, False, True,
                True, True, False, False, True,
                False, False]

n = 1000
sq_sols = np.array(generator.status_quo(n=n))
#%%
def sol_gen(x):    
    out = np.zeros([sq_sols.shape[0], n])
    for i in range(sq_sols.shape[2]):
        out[:, i] = np.dot(sq_sols[:,:,i], np.array(x))
    return out
    
x =  np.ones(11)
y = sol_gen(x)

inp_comb = itertools.product([0, 1], repeat=len(x))
inps = np.array([i for i in inp_comb])
sols = np.array([sol_gen(inp) for inp in inps])


sorted_solutions = sols[:]
# Turn the problem into a minimisation problem
for i, max_true in enumerate(obj_maximise):
    if max_true:
        sorted_solutions[:, i, :] = sorted_solutions[:,i,:]*-1

# Get the pareto fronts in the mean values
mean_pf = utils.pareto_fronts(np.mean(sorted_solutions, axis=1), minimize=True)
        
        